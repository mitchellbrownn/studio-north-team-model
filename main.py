"""Studio North team forecast: hierarchically shrunk Beta-discrete-Weibull (BdW)
retention, plus recalibrated cash per retained member with a pooled drift.

Retention: within each channel x stated-need cell, members have persistent churn
propensities drawn from a Beta distribution, and a discrete-Weibull shape c lets
each member's propensity rise (c > 1) or fall (c < 1) with tenure; c = 1 is the
shifted-Beta-Geometric (sBG) special case. The Beta mean has a linear acquisition-
cohort trend. All four cells are fitted jointly by penalized maximum likelihood
with cell -> channel -> overall shrinkage, on observed renewal decisions only.
Cash: net revenue and contribution per retained member by cell and age (smoothed
across neighbouring ages; late-age average beyond observed ages), recalibrated to
the last three observed months (credibility-shrunk toward channel and overall),
then grown by a monthly drift estimated over the last 12 months pooled across
all cells and capped at +/-2%.

This is NOT the released hierarchical sBG + Gamma reference. Choices are justified
by chronological validation in scripts/backtest.py and experiments/.
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
from scipy.special import betaln, expit

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
RETENTION_MODEL = 'bdw_pooled'  # 'bdw_pooled' | 'sbg' | 'bdw' | 'logit_hazard'
PRIOR_LOGC_SD = 0.5         # BdW: log Weibull shape c ~ N(0, sd); c = 1 is sBG
PRIOR_LOGIT_SD = 2.0        # logistic hazard: ridge on slopes
CASH_METHOD = 'recalibrate' # 'age' | 'recency' | 'cohort_trend' | 'recalibrate'
RECENCY_HALF_LIFE = 6.0     # months, for CASH_METHOD == 'recency'
RECAL_MONTHS = 3            # recent calendar months, for CASH_METHOD == 'recalibrate'
# Shrinkage (cell -> channel -> overall), fixed before scoring.
POOL_CHANNEL_SD = 0.5       # bdw_pooled: channel deviations from overall parameters
POOL_CELL_SD = 0.3          # bdw_pooled: cell deviations from channel parameters
RECAL_SHRINK = 300          # member-months credibility constant for recalibration (0 = off)
DRIFT_MONTHS = 12           # >0: project pooled cash drift estimated over this many months
DRIFT_CAP = 0.02            # max |monthly drift| projected


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


def _renewal_data(cell_panel: pd.DataFrame):
    p = cell_panel.sort_values(['cohort','age'])
    prev = p.groupby('cohort').n_retained.shift(1)
    d = p.assign(prev=prev).dropna(subset=['prev'])
    d = d[d.prev>0]
    return (d.age.to_numpy(float), d.prev.to_numpy(float), d.n_retained.to_numpy(float), _cohort_z(d.cohort))


def renewal_prob(params: np.ndarray, t: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Probability of renewing at age t (t >= 1) given active at t-1."""
    if RETENTION_MODEL == 'logit_hazard':
        c0,c1,c2 = params
        return 1.0-expit(c0+c1*np.log(t)+c2*z)
    a,b,lphi = params[:3]
    phi = np.exp(lphi); mu = expit(a+b*z)
    if RETENTION_MODEL in ('bdw','bdw_pooled'):
        al, be, c = mu*phi, (1-mu)*phi, np.exp(params[3])
        return np.exp(betaln(al,be+t**c)-betaln(al,be+(t-1.0)**c))
    return 1.0-mu*phi/(phi+t-1.0)


def fit_retention(cell_panel: pd.DataFrame) -> np.ndarray:
    """Penalized MLE of the selected retention model on observed renewals.
    sbg: (a, b, log phi). bdw: Beta-discrete-Weibull, adds log c, where c != 1
    lets each member's churn propensity rise (c > 1) or fall (c < 1) with
    tenure. logit_hazard: churn = logistic(c0 + c1 log t + c2 z_k), with no
    Beta mixture."""
    if RETENTION_MODEL == 'sbg':
        return fit_sbg(cell_panel)
    t,s_prev,s,z = _renewal_data(cell_panel)
    if RETENTION_MODEL == 'bdw':
        starts = [[a,0.0,lp,0.0] for a,lp in ((-1.5,np.log(4.0)),(-2.5,0.0),(-0.5,np.log(10.0)))]
        bounds = [(-10,5),(-5,5),(-5,8),(-2,2)]
        def pen(th):
            return (0.5*(th[1]/PRIOR_TREND_SD)**2 + 0.5*((th[2]-PRIOR_LOGPHI[0])/PRIOR_LOGPHI[1])**2
                    + 0.5*(th[3]/PRIOR_LOGC_SD)**2)
    else:
        starts = [[-2.0,0.0,0.0],[-1.0,-0.5,0.0],[-3.0,-1.0,0.0]]
        bounds = [(-10,5),(-5,5),(-5,5)]
        def pen(th):
            return 0.5*((th[1]/PRIOR_LOGIT_SD)**2+(th[2]/PRIOR_TREND_SD)**2)
    def nll(th):
        q = np.clip(renewal_prob(th,t,z),1e-12,1-1e-12)
        return -(s*np.log(q)+(s_prev-s)*np.log1p(-q)).sum()+pen(th)
    best = None
    for st in starts:
        r = minimize(nll,np.array(st,float),method='L-BFGS-B',bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def fit_bdw_pooled(panel: pd.DataFrame) -> dict:
    """Beta-discrete-Weibull fitted jointly across the channel x need cells with
    hierarchical shrinkage: theta_cell = theta_all + gamma_channel + delta_cell,
    gamma ~ N(0, POOL_CHANNEL_SD^2), delta ~ N(0, POOL_CELL_SD^2) elementwise.
    theta = (a, b, log phi, log c), with the overall priors used by 'bdw'."""
    cells = sorted({tuple(x) for x in panel[CELL].drop_duplicates().to_numpy()})
    chans = sorted({c[0] for c in cells})
    data = {c:_renewal_data(panel[(panel.channel==c[0])&(panel.primary_need==c[1])]) for c in cells}
    nC, nH = len(cells), len(chans)
    def unpack(x):
        base = x[:4]; gam = x[4:4+4*nH].reshape(nH,4); dlt = x[4+4*nH:].reshape(nC,4)
        return base, gam, dlt
    def nll(x):
        base, gam, dlt = unpack(x)
        tot = 0.0
        for i,c in enumerate(cells):
            th = base+gam[chans.index(c[0])]+dlt[i]
            t,sp,s,z = data[c]
            q = np.clip(np.exp(betaln(expit(th[0]+th[1]*z)*np.exp(th[2]),(1-expit(th[0]+th[1]*z))*np.exp(th[2])+t**np.exp(th[3]))
                               -betaln(expit(th[0]+th[1]*z)*np.exp(th[2]),(1-expit(th[0]+th[1]*z))*np.exp(th[2])+(t-1.0)**np.exp(th[3]))),1e-12,1-1e-12)
            tot -= (s*np.log(q)+(sp-s)*np.log1p(-q)).sum()
        pen = (0.5*(base[1]/PRIOR_TREND_SD)**2 + 0.5*((base[2]-PRIOR_LOGPHI[0])/PRIOR_LOGPHI[1])**2
               + 0.5*(base[3]/PRIOR_LOGC_SD)**2 + 0.5*np.sum((gam/POOL_CHANNEL_SD)**2)
               + 0.5*np.sum((dlt/POOL_CELL_SD)**2))
        return tot+pen
    best = None
    for a0,lp0 in ((-1.5,np.log(4.0)),(-2.5,0.0),(-0.5,np.log(10.0))):
        x0 = np.zeros(4+4*nH+4*nC); x0[0], x0[2] = a0, lp0
        r = minimize(nll,x0,method='L-BFGS-B')
        if best is None or r.fun < best.fun:
            best = r
    base, gam, dlt = unpack(best.x)
    return {c: base+gam[chans.index(c[0])]+dlt[i] for i,c in enumerate(cells)}


def recalibration_ratios(panel: pd.DataFrame, origin: pd.Period) -> dict:
    """Observed / profile-implied contribution over the last RECAL_MONTHS months.
    With RECAL_SHRINK > 0 the cell ratio is credibility-weighted toward its
    channel ratio, and the channel ratio toward the overall ratio, using
    retained member-months in the window: Z = n / (n + RECAL_SHRINK)."""
    parts = []
    for (ch,nd),cell in panel.groupby(CELL):
        p = cell[(cell.age>=1)&(cell.n_retained>0)].copy()
        prof = cash_profile(cell,np.sort(p.age.unique()))
        p['implied'] = p.n_retained*prof.loc[p.age,'contribution_pm'].to_numpy()
        back = origin.ordinal-(p.cohort.apply(lambda c:c.ordinal)+p.age)
        w = p[back < RECAL_MONTHS]
        parts.append((ch,nd,w.contribution.sum(),w.implied.sum(),w.n_retained.sum()))
    df = pd.DataFrame(parts,columns=['channel','primary_need','obs','imp','n'])
    ratio = lambda o,i: float(np.clip(o/i,0.5,2.0)) if i>0 else 1.0
    if RECAL_SHRINK <= 0:
        return {(r.channel,r.primary_need):ratio(r.obs,r.imp) for r in df.itertuples()}
    r_all = ratio(df.obs.sum(),df.imp.sum()); out = {}
    for ch,g in df.groupby('channel'):
        z = g.n.sum()/(g.n.sum()+RECAL_SHRINK); r_ch = z*ratio(g.obs.sum(),g.imp.sum())+(1-z)*r_all
        for r in g.itertuples():
            z2 = r.n/(r.n+RECAL_SHRINK); out[(ch,r.primary_need)] = z2*ratio(r.obs,r.imp)+(1-z2)*r_ch
    return out


def pooled_drift(panel: pd.DataFrame, origin: pd.Period) -> float:
    """Monthly growth of observed vs profile-implied contribution, pooled over all
    cells (no per-cell parameter), from a weighted log-linear fit over the last
    DRIFT_MONTHS calendar months; capped at +/- DRIFT_CAP."""
    if DRIFT_MONTHS <= 0:
        return 0.0
    rows = []
    for _,cell in panel.groupby(CELL):
        p = cell[(cell.age>=1)&(cell.n_retained>0)].copy()
        prof = cash_profile(cell,np.sort(p.age.unique()))
        p['implied'] = p.n_retained*prof.loc[p.age,'contribution_pm'].to_numpy()
        p['back'] = origin.ordinal-(p.cohort.apply(lambda c:c.ordinal)+p.age)
        rows.append(p[p.back < DRIFT_MONTHS][['back','contribution','implied']])
    m = pd.concat(rows).groupby('back')[['contribution','implied']].sum()
    m = m[(m.implied>0)&(m.contribution>0)]
    if len(m) < 3:
        return 0.0
    x = -m.index.to_numpy(float); y = np.log(m.contribution/m.implied).to_numpy(); w = m.implied.to_numpy()
    xb = np.average(x,weights=w); g = np.sum(w*(x-xb)*(y-np.average(y,weights=w)))/np.sum(w*(x-xb)**2)
    return float(np.clip(np.expm1(g),-DRIFT_CAP,DRIFT_CAP))


def cash_profile(cell_panel: pd.DataFrame, ages: np.ndarray, origin: pd.Period | None = None) -> pd.DataFrame:
    """Net revenue and contribution per retained member by age, pooling
    neighbouring ages; beyond observed ages use the late-age average.
    With CASH_METHOD == 'recency', observations are weighted by
    0.5 ** (months before origin / RECENCY_HALF_LIFE)."""
    p = cell_panel
    if CASH_METHOD == 'recency' and origin is not None:
        months_back = origin.ordinal-(p.cohort.apply(lambda c:c.ordinal)+p.age)
        w = 0.5**(months_back/RECENCY_HALF_LIFE)
        p = p.assign(n_retained=p.n_retained*w,net=p.net*w,contribution=p.contribution*w)
    by_age = p.groupby('age')[['n_retained','net','contribution']].sum()
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


def cash_multipliers(cell_panel: pd.DataFrame, cohorts: pd.Series, origin: pd.Period) -> np.ndarray:
    """Per-row multiplier on the age profile (1.0 for the plain 'age' method).
    'cohort_trend': exp(g * z_k), g from a retained-weighted regression of the
    log ratio (observed / age-profile cash per member) on the cohort index.
    'recalibrate': ratio of observed to profile-implied cash over the last
    RECAL_MONTHS calendar months of the cell."""
    if CASH_METHOD not in ('cohort_trend','recalibrate'):
        return np.ones(len(cohorts))
    p = cell_panel[(cell_panel.age>=1)&(cell_panel.n_retained>0)].copy()
    prof = cash_profile(cell_panel,np.sort(p.age.unique()))
    p['implied'] = p.n_retained*prof.loc[p.age,'contribution_pm'].to_numpy()
    if CASH_METHOD == 'recalibrate':
        months_back = origin.ordinal-(p.cohort.apply(lambda c:c.ordinal)+p.age)
        r = p[months_back < RECAL_MONTHS]
        ratio = r.contribution.sum()/r.implied.sum() if r.implied.sum()>0 else 1.0
        return np.full(len(cohorts),float(np.clip(ratio,0.5,2.0)))
    p = p[(p.implied>0)&(p.contribution>0)]
    z = _cohort_z(p.cohort); y = np.log(p.contribution/p.implied); w = p.n_retained.to_numpy(float)
    zbar = np.average(z,weights=w)
    g = np.sum(w*(z-zbar)*(y-np.average(y,weights=w)))/np.sum(w*(z-zbar)**2)
    return np.exp(np.clip(g,-1,1)*(_cohort_z(cohorts)-zbar))


def predict(tables: dict[str,pd.DataFrame], prediction_grid: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Use only the supplied observed tables; return one forecast per grid row.
    Deterministic: `seed` is accepted for the common interface but unused."""
    grid = prediction_grid.copy(deep=True)
    origin = pd.Period(grid.calendar_month.min(),freq='M') - 1
    panel = build_panel(tables,origin)
    pooled = fit_bdw_pooled(panel) if RETENTION_MODEL == 'bdw_pooled' else None
    shrunk_recal = CASH_METHOD == 'recalibrate' and (RECAL_SHRINK > 0 or DRIFT_MONTHS > 0)
    ratios = recalibration_ratios(panel,origin) if shrunk_recal else None
    drift = pooled_drift(panel,origin)
    results = []
    for (channel,need),cell in panel.groupby(CELL):
        params = pooled[(channel,need)] if pooled is not None else fit_retention(cell)
        rows = grid[(grid.channel==channel)&(grid.primary_need==need)].copy()
        if rows.empty:
            continue
        prof = cash_profile(cell,np.sort(rows.age_month.unique()),origin)
        rows['cohort'] = pd.PeriodIndex(rows.cohort_month,freq='M')
        if ratios is not None:
            ahead = np.array([pd.Period(m,freq='M').ordinal-origin.ordinal for m in rows.calendar_month],float)
            mult = ratios[(channel,need)]*(1.0+drift)**ahead
        else:
            mult = cash_multipliers(cell,rows.cohort,origin)
        rows['z'] = _cohort_z(rows.cohort)
        rows['origin_age'] = [origin.ordinal-c.ordinal for c in rows.cohort]
        surv = []
        for r in rows.itertuples(index=False):
            ts = np.arange(r.origin_age+1,int(r.age_month)+1,dtype=float)
            q = renewal_prob(params,ts,np.full(len(ts),r.z))
            surv.append(float(r.origin_retained)*float(np.prod(q)))
        rows['predicted_retained'] = surv
        net_pm = prof.loc[rows.age_month,'net_pm'].to_numpy()*mult
        con_pm = np.minimum(net_pm,prof.loc[rows.age_month,'contribution_pm'].to_numpy()*mult)
        rows['predicted_net_revenue'] = rows.predicted_retained*net_pm
        rows['predicted_contribution'] = rows.predicted_retained*con_pm
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
    receipt = {'model':'Hierarchically shrunk Beta-discrete-Weibull retention + recalibrated cash with pooled drift',
               'rows':len(predictions),'seed':args.seed,'elapsed_seconds':time.monotonic()-started,
               'source_sha256':sha(__file__),'uv_lock_sha256':sha(Path(__file__).with_name('uv.lock')),
               'contract_sha256':sha(contract_path),'prediction_grid_sha256':sha(args.prediction_grid),
               'input_sha256':{name:sha(args.data_dir/(name+'.csv')) for name in tables},
               'prediction_sha256':sha(args.output),'future_outcomes_read':False}
    args.output.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'validated','rows':len(predictions),'output':str(args.output)}))


if __name__ == '__main__':
    main()
