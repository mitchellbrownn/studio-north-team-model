"""Chronological validation using only observed history.

For each historical origin, every input table is truncated to records available
at that origin (transactions before the cutoff; renewals already due), a grid is
rebuilt from the truncated panel, and the model forecasts the following months.
Forecasts are scored against the later *observed* months with the challenge's own
primary rule (cohort x calendar-month contribution WAPE after summing cells).
The scaffold's constant-hazard example is scored the same way as a baseline.

Usage:
  uv run --frozen python scripts/backtest.py --data-dir PATH_TO_RELEASED_DATA
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
import main as model                                   # noqa: E402
import example_constant_hazard as example              # noqa: E402

ORIGINS = [('2024-12',12),('2025-03',9),('2025-06',6)]


def truncate(tables, origin: pd.Period):
    cutoff = (origin+1).to_timestamp()
    t = {k:v.copy() for k,v in tables.items()}
    t['transactions'] = t['transactions'][pd.to_datetime(t['transactions'].occurred_at) < cutoff]
    t['subscription_periods'] = t['subscription_periods'][pd.to_datetime(t['subscription_periods'].renewal_due_at) < cutoff]
    return t


def make_grid(panel: pd.DataFrame, origin: pd.Period, horizon: int) -> pd.DataFrame:
    last = panel[panel.age == panel.cohort.apply(lambda c: origin.ordinal-c.ordinal)]
    rows = []
    for r in last.itertuples(index=False):
        for h in range(1,horizon+1):
            rows.append({'cohort_month':r.cohort.to_timestamp().strftime('%Y-%m-%d'),'channel':r.channel,
                         'primary_need':r.primary_need,
                         'calendar_month':(origin+h).to_timestamp().strftime('%Y-%m-%d'),
                         'age_month':int(r.age+h),'n_acquired':int(r.n_acquired),
                         'origin_retained':int(r.n_retained)})
    return pd.DataFrame(rows)


def score(pred: pd.DataFrame, actual: pd.DataFrame) -> dict:
    keys = ['cohort_month','channel','primary_need','calendar_month']
    m = pred.merge(actual,on=keys,validate='one_to_one')
    cm = m.groupby(['cohort_month','calendar_month'])[['predicted_contribution','contribution']].sum()
    denom = cm.contribution.abs().sum()
    return {'contribution_wape': float((cm.predicted_contribution-cm.contribution).abs().sum()/denom),
            'retention_cell_wape': float((m.predicted_retained-m.n_retained).abs().sum()/m.n_retained.abs().sum()),
            'signed_contribution_bias': float((m.predicted_contribution.sum()-m.contribution.sum())/denom),
            'score_points': int(len(cm))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'validation'/'backtest.json')
    args = p.parse_args()
    contract = json.loads((ROOT/'submission-contract.json').read_text())
    tables = model.load_tables(args.data_dir,contract)
    full_origin = pd.Period('2025-12',freq='M')
    full = model.build_panel(tables,full_origin)
    full['cohort_month'] = full.cohort.apply(lambda c:c.to_timestamp().strftime('%Y-%m-%d'))
    full['calendar_month'] = full.apply(lambda r:(r.cohort+int(r.age)).to_timestamp().strftime('%Y-%m-%d'),axis=1)
    results = []
    for o,h in ORIGINS:
        origin = pd.Period(o,freq='M')
        tt = truncate(tables,origin)
        grid = make_grid(model.build_panel(tt,origin),origin,h)
        actual = full[['cohort_month','channel','primary_need','calendar_month','n_retained','contribution']]
        row = {'origin':o,'horizon_months':h,'grid_rows':len(grid),
               'max_forecast_age':int(grid.age_month.max())}
        for name,fn in (('cell_sbg',model.predict),('constant_hazard_example',example.predict)):
            pred = model.validate_output(fn(tt,grid.copy(deep=True),seed=20261116),grid)
            row[name] = score(pred,actual)
        results.append(row)
        print(json.dumps(row))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'origins':results,'note':'Observed-history backtests only; '
                                       'no 2026 outcomes are available or used.'},indent=2)+'\n')


if __name__ == '__main__':
    main()
