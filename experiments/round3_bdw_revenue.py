"""Round 3: BdW retention with shrinkage, then pre-specified revenue variants.

Experiment only; observed history only (inputs truncated at each cutoff). Same
harness and scoring as round 2 (contribution WAPE, retained WAPE, retention-only
contribution WAPE, signed biases). All settings below were fixed before scoring.

Usage:
  uv run --frozen python experiments/round3_bdw_revenue.py --data-dir PATH
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts')); sys.path.insert(0, str(ROOT/'experiments'))
import main as M                      # noqa: E402
import backtest as B                  # noqa: E402
from retention_round2 import scores   # noqa: E402

BASE = {'RETENTION_MODEL':'sbg','CASH_METHOD':'recalibrate','RECAL_MONTHS':3,'RECAL_SHRINK':0,'DRIFT_MONTHS':0}
VARIANTS = {
    'sBG + recal 3m (published)':                 {},
    'BdW per-cell + recal 3m':                    {'RETENTION_MODEL':'bdw'},
    'BdW pooled + recal 3m':                      {'RETENTION_MODEL':'bdw_pooled'},
    'BdW pooled + recal 3m shrunk':               {'RETENTION_MODEL':'bdw_pooled','RECAL_SHRINK':300},
    'BdW pooled + recal 3m shrunk + drift 12m':   {'RETENTION_MODEL':'bdw_pooled','RECAL_SHRINK':300,'DRIFT_MONTHS':12},
    'BdW pooled + recal 1m shrunk':               {'RETENTION_MODEL':'bdw_pooled','RECAL_SHRINK':300,'RECAL_MONTHS':1},
    'BdW pooled + recal 1m shrunk + drift 12m':   {'RETENTION_MODEL':'bdw_pooled','RECAL_SHRINK':300,'RECAL_MONTHS':1,'DRIFT_MONTHS':12},
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data-dir', type=Path, required=True)
    args = ap.parse_args()
    tables = M.load_tables(args.data_dir, json.loads((ROOT/'submission-contract.json').read_text()))
    full = M.build_panel(tables, pd.Period('2025-12', freq='M'))
    full['cohort_month'] = full.cohort.apply(lambda c: c.to_timestamp().strftime('%Y-%m-%d'))
    full['calendar_month'] = full.apply(lambda r: (r.cohort+int(r.age)).to_timestamp().strftime('%Y-%m-%d'), axis=1)
    actual = full[['cohort_month','channel','primary_need','calendar_month','n_retained','contribution']]
    saved = {k: getattr(M, k) for k in BASE}
    out = {n: {'cutoffs': {}, 'runtime_s': [], 'drift': {}} for n in VARIANTS}
    for o, h in B.ORIGINS:
        origin = pd.Period(o, freq='M'); tt = B.truncate(tables, origin)
        grid = B.make_grid(M.build_panel(tt, origin), origin, h)
        for n, s in VARIANTS.items():
            for k, v in {**BASE, **s}.items(): setattr(M, k, v)
            try:
                t0 = time.perf_counter()
                pred = M.validate_output(M.predict(tt, grid.copy(deep=True), seed=20261116), grid)
                out[n]['runtime_s'].append(time.perf_counter()-t0)
                out[n]['cutoffs'][o] = scores(pred, actual)[0]
                out[n]['drift'][o] = M.pooled_drift(M.build_panel(tt, origin), origin)
            finally:
                for k, v in saved.items(): setattr(M, k, v)
        print('done', o, flush=True)
    dest = ROOT/'experiments'/'round3'/'results.json'; dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({'variants': VARIANTS, 'base': BASE, 'results': out}, indent=1)+'\n')
    print('wrote', dest)


if __name__ == '__main__':
    main()
