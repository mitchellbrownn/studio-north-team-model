"""Studio North team forecast: cell-level shifted-Beta-Geometric retention with a
cohort trend, plus age-profiled cash per retained member.

Retention: within each channel x stated-need cell, members have persistent monthly
churn propensities drawn from a Beta distribution (Fader & Hardie sBG). The Beta
mean shifts with a linear acquisition-cohort trend; concentration is shared within
the cell. Parameters are fitted by penalized maximum likelihood on observed renewal
decisions only. Cash: expected net revenue and contribution per retained member by
cell and customer age, smoothed across neighbouring ages; ages beyond the observed
range reuse the cell's late-age average.

This is NOT the released hierarchical sBG + Gamma reference model. Model choice and
settings are justified by chronological validation in scripts/backtest.py.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

KEYS = ['cohort_month','channel','primary_need','calendar_month','age_month']
EXPOSURES = ['n_acquired','origin_retained']
PREDICTIONS = ['predicted_retained','predicted_net_revenue','predicted_contribution']
CELL = ['channel','primary_need']
FIRST_COHORT = pd.Period('2024-01',freq='M')
# Fixed modelling settings (chosen before private evaluation; see README).
TREND_SCALE = 11.5          # cohort index centring/scale, as in the lab's reference
PRIOR_TREND_SD = 1.0        # weak ridge on the cohort trend (logit scale)
PRIOR_LOGPHI = (np.log(4.0), 2.0)  # weak prior on log concentration
CASH_WINDOW = 1             # +/- ages pooled when estimating cash per member
LATE_AGES = 6               # last observed ages averaged for extrapolation


def build_panel(tables: dict[str,pd.DataFrame], origin: pd.Period) -> pd.DataFrame:
    """Observed cohort x cell x age panel through `origin` (inclusive), using only
    records dated before the first day of the following month."""
    cutoff = (origin+1).to_timestamp()
    tx = tables['transactions'].copy()
    tx['occurred'] = pd.to_datetime(tx.occurred_at)
    tx = tx[tx.occurred < cutoff]
    pay = tx[(tx.transaction_type=='payment') & ((tx.gross_amount-tx.discount_amount)>0)]
    first = (pay.sort_values(['occurred','transaction_id']).groupby('person_id',as_index=False).first()
             [['person_id','occurred']])
    first['cohort'] = first.occurred.dt.to_period('M')
    people = tables['people'][['person_id','primary_need']]
    campaigns = tables['campaign_spend'][['campaign_id','channel']].drop_duplicates()
    if campaigns.campaign_id.duplicated().any():
        raise ValueError('A campaign has conflicting channels')
    attr = tables['acquisition_attribution'].merge(campaigns,left_on='credited_campaign_id',
                                                   right_on='campaign_id',how='left',validate='many_to_one')
    attr['channel'] = attr.channel.fillna(attr.source_type)
    subjects = (first.merge(people,on='person_id',validate='one_to_one')
                     .merge(attr[['person_id','channel']],on='person_id',validate='one_to_one'))
    subjects = subjects[subjects.channel.isin(['search','partner'])]
    # Contractual retention: a paid renewal for a period starting in that month,
    # counting only renewals already due before the cutoff.
    sp = tables['subscription_periods'].copy()
    sp = sp[pd.to_datetime(sp.renewal_due_at) < cutoff]
    sp['month'] = pd.to_datetime(sp.period_start).dt.to_period('M')
    ret = (sp[sp.renewal_outcome=='paid'].groupby(['person_id','month']).size()
             .rename('retained').clip(upper=1).reset_index())
    tx['month'] = tx.occurred.dt.to_period('M')
    tx['net'] = tx.gross_amount-tx.discount_amount-tx.refund_amount
    tx['contribution'] = tx.net-tx.variable_cost
    cash = tx.groupby(['person_id','month'])[['net','contribution']].sum().reset_index()
    rows = []
    origin_ord = origin.ordinal
    for s in subjects.itertuples(index=False):
        for o in range(s.cohort.ordinal, origin_ord+1):
            rows.append((s.person_id,s.cohort,s.channel,s.primary_need,pd.Period(ordinal=o,freq='M')))
    cm = pd.DataFrame(rows,columns=['person_id','cohort','channel','primary_need','month'])
    cm = cm.merge(ret,on=['person_id','month'],how='left').merge(cash,on=['person_id','month'],how='left')
    cm[['retained','net','contribution']] = cm[['retained','net','contribution']].fillna(0)
    cm['age'] = cm.month.apply(lambda m:m.ordinal) - cm.cohort.apply(lambda c:c.ordinal)
    panel = (cm.groupby(['cohort','channel','primary_need','age'])
               .agg(n_acquired=('person_id','size'),n_retained=('retained','sum'),
                    net=('net','sum'),contribution=('contribution','sum')).reset_index())
    return panel


def _cohort_z(cohort: pd.Series) -> np.ndarray:
    idx = np.array([c.ordinal-FIRST_COHORT.ordinal for c in cohort],dtype=float)
    return (idx-TREND_SCALE)/TREND_SCALE


def fit_sbg(cell_panel: pd.DataFrame) -> np.ndarray:
    """Penalized MLE of (a, b, log phi): mu_k = expit(a + b z_k), phi shared.
    Churn hazard at renewal t>=1 is mu*phi/(phi+t-1); renewals are Binomial
    in the previous survivors, each decision counted once."""
    p = cell_panel.sort_values(['cohort','age'])
    prev = p.groupby('cohort').n_retained.shift(1)
    d = p.assign(prev=prev).dropna(subset=['prev'])
    d = d[d.prev>0]
    t = d.age.to_numpy(float); s_prev = d.prev.to_numpy(float); s = d.n_retained.to_numpy(float)
    z = _cohort_z(d.cohort)
    def nll(theta):
        a,b,lphi = theta
        phi = np.exp(lphi); mu = expit(a+b*z)
        h = np.clip(mu*phi/(phi+t-1.0),1e-12,1-1e-12)
        ll = (s*np.log1p(-h) + (s_prev-s)*np.log(h)).sum()
        pen = 0.5*(b/PRIOR_TREND_SD)**2 + 0.5*((lphi-PRIOR_LOGPHI[0])/PRIOR_LOGPHI[1])**2
        return -ll+pen
    best = None
    for start in ([-1.5,0.0,np.log(4.0)],[-2.5,0.0,np.log(1.0)],[-0.5,0.0,np.log(10.0)]):
        r = minimize(nll,np.array(start),method='L-BFGS-B',bounds=[(-10,5),(-5,5),(-5,8)])
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def cash_profile(cell_panel: pd.DataFrame, ages: np.ndarray) -> pd.DataFrame:
    """Net revenue and contribution per retained member by age, pooling
    neighbouring ages; beyond observed ages use the late-age average."""
    by_age = cell_panel.groupby('age')[['n_retained','net','contribution']].sum()
    by_age = by_age[by_age.index>=1]
    max_age = int(by_age.index.max())
    out = {}
    late = by_age[by_age.index>max_age-LATE_AGES].sum()
    late_rate = (late.net/late.n_retained, late.contribution/late.n_retained)
    for a in ages:
        if a > max_age:
            out[a] = late_rate; continue
        w = by_age[(by_age.index>=a-CASH_WINDOW)&(by_age.index<=a+CASH_WINDOW)].sum()
        out[a] = (w.net/w.n_retained, w.contribution/w.n_retained) if w.n_retained>0 else late_rate
    prof = pd.DataFrame.from_dict(out,orient='index',columns=['net_pm','contribution_pm'])
    prof['net_pm'] = prof.net_pm.clip(lower=0)
    prof['contribution_pm'] = np.minimum(prof.net_pm,prof.contribution_pm.clip(lower=0))
    return prof


def predict(tables: dict[str,pd.DataFrame], prediction_grid: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Use only the supplied observed tables; return one forecast per grid row.
    Deterministic: `seed` is accepted for the common interface but unused."""
    grid = prediction_grid.copy(deep=True)
    origin = pd.Period(grid.calendar_month.min(),freq='M') - 1
    panel = build_panel(tables,origin)
    results = []
    for (channel,need),cell in panel.groupby(CELL):
        a,b,lphi = fit_sbg(cell)
        phi = np.exp(lphi)
        rows = grid[(grid.channel==channel)&(grid.primary_need==need)].copy()
        if rows.empty:
            continue
        prof = cash_profile(cell,np.sort(rows.age_month.unique()))
        rows['cohort'] = pd.PeriodIndex(rows.cohort_month,freq='M')
        rows['mu'] = expit(a+b*_cohort_z(rows.cohort))
        rows['origin_age'] = [origin.ordinal-c.ordinal for c in rows.cohort]
        surv = []
        for r in rows.itertuples(index=False):
            ts = np.arange(r.origin_age+1,int(r.age_month)+1,dtype=float)
            h = r.mu*phi/(phi+ts-1.0)
            surv.append(float(r.origin_retained)*float(np.prod(1.0-h)))
        rows['predicted_retained'] = surv
        rows['predicted_net_revenue'] = rows.predicted_retained*prof.loc[rows.age_month,'net_pm'].to_numpy()
        rows['predicted_contribution'] = rows.predicted_retained*prof.loc[rows.age_month,'contribution_pm'].to_numpy()
        results.append(rows)
    out = pd.concat(results)
    missing = len(grid)-len(out)
    if missing:
        raise ValueError(f'{missing} grid rows belong to cells without observed history')
    zero = out.predicted_retained<=0
    out.loc[zero,['predicted_retained','predicted_net_revenue','predicted_contribution']] = 0.0
    return out[KEYS+EXPOSURES+PREDICTIONS]


def load_tables(data_dir: Path, contract: dict) -> dict[str,pd.DataFrame]:
    expected = contract['input_tables']
    if {p.name for p in data_dir.glob('*.csv')} != {name+'.csv' for name in expected}:
        raise ValueError('Supply exactly the eight permitted observed CSVs')
    tables = {}
    for name, fields in expected.items():
        dtypes = {field:('string' if kind in {'VARCHAR','DATE','TIMESTAMP'} else
                         'Int64' if kind=='INTEGER' else 'float64') for field,kind in fields.items()}
        table = pd.read_csv(data_dir/(name+'.csv'),dtype=dtypes)
        if list(table.columns)!=list(fields):
            raise ValueError('Unexpected source schema: '+name)
        tables[name] = table
    return tables


def validate_output(predictions: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(predictions,pd.DataFrame) or set(predictions)!=set(KEYS+EXPOSURES+PREDICTIONS):
        raise ValueError('Return exactly the required grid and prediction columns')
    if predictions.duplicated(KEYS).any():
        raise ValueError('Duplicate prediction keys')
    ordered = predictions.sort_values(KEYS).reset_index(drop=True)
    expected = grid.sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(ordered[KEYS+EXPOSURES],expected[KEYS+EXPOSURES],check_dtype=False)
    values = ordered[PREDICTIONS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values<0).any():
        raise ValueError('Predictions must be finite and nonnegative')
    if (ordered.predicted_retained>ordered.origin_retained+1e-8).any():
        raise ValueError('Predicted retained count exceeds origin survivors')
    if (ordered.groupby(KEYS[:3]).predicted_retained.diff().dropna()>1e-8).any():
        raise ValueError('Expected membership must not increase')
    if ((ordered.predicted_retained==0) & ((ordered.predicted_net_revenue>0)|(ordered.predicted_contribution>0))).any():
        raise ValueError('Zero expected members must have zero expected receipts')
    if (ordered.predicted_contribution>ordered.predicted_net_revenue+1e-8).any():
        raise ValueError('Contribution exceeds net revenue')
    return ordered[KEYS+EXPOSURES+PREDICTIONS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--prediction-grid',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261116)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.receipt.json').exists():
        parser.error('Use a new output path; existing forecasts are preserved')
    contract_path = Path(__file__).with_name('submission-contract.json')
    contract = json.loads(contract_path.read_text())
    tables = load_tables(args.data_dir,contract)
    grid = pd.read_csv(args.prediction_grid)
    started = time.monotonic()
    predictions = validate_output(predict(tables,grid.copy(deep=True),seed=args.seed),grid)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    predictions.to_csv(args.output,index=False,float_format='%.10f',lineterminator='\n')
    sha = lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    receipt = {'model':'Cell-level sBG retention with cohort trend + age-profiled cash per member',
               'rows':len(predictions),'seed':args.seed,'elapsed_seconds':time.monotonic()-started,
               'source_sha256':sha(__file__),'uv_lock_sha256':sha(Path(__file__).with_name('uv.lock')),
               'contract_sha256':sha(contract_path),'prediction_grid_sha256':sha(args.prediction_grid),
               'input_sha256':{name:sha(args.data_dir/(name+'.csv')) for name in tables},
               'prediction_sha256':sha(args.output),'future_outcomes_read':False}
    args.output.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'validated','rows':len(predictions),'output':str(args.output)}))


if __name__ == '__main__':
    main()
