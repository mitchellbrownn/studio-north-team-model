"""Retention round 2: compare retention models on the challenge backtest harness.

Experiment only. It does not change main.py's selected configuration or the
released sBG + Gamma reference, and it uses observed history only. At each cutoff
every input is truncated to what was observable then; no 2026 outcomes exist here.

Scores per model and cutoff:
  * contribution WAPE (cohort x calendar month, cells summed), with the selected
    cash method (age profile recalibrated to the last 3 months), as in backtest.py;
  * retained-member WAPE (cell x month);
  * retention-only contribution WAPE: predicted retained x the historically
    realized contribution per surviving member in that cell-month, which removes
    revenue-model error. Cell-months with zero realized survivors use 0 per member.

Usage:
  uv run --frozen --with matplotlib==3.10.3 python experiments/retention_round2.py --data-dir PATH
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, softmax

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
import main as M            # noqa: E402
import backtest as B        # noqa: E402

SEED = 20261116
CELLS = [('partner','build_skill'),('partner','save_time'),('search','build_skill'),('search','save_time')]
# Pre-specified shrinkage constants (set before scoring):
K_CRED_MEMBERS = 100.0   # chain-ladder credibility: Z = at-risk / (at-risk + K)
K_CRED_DECISIONS = 200.0 # recent-churn credibility: Z = decisions / (decisions + K)
LCG_DELTA_SD = 0.5       # latent-class: cell mixing logits ~ N(overall logits, sd^2)


# ---------------------------------------------------------------- helpers
def survivors_table(panel):
    """dict cell -> DataFrame(index=cohort Period, columns=age, values=n_retained)."""
    out = {}
    for cell, g in panel.groupby(['channel','primary_need']):
        out[cell] = g.pivot(index='cohort', columns='age', values='n_retained')
    return out


def origin_age(cohort, origin):
    return origin.ordinal - cohort.ordinal


def project(grid, cell_fn):
    """cell_fn(cell, cohort, t0, ages) -> survival ratios S(age)/S(t0) for future ages."""
    g = grid.copy()
    g['cohort'] = pd.PeriodIndex(g.cohort_month, freq='M')
    vals = np.zeros(len(g))
    for (ch, nd, co), idx in g.groupby(['channel','primary_need','cohort']).groups.items():
        rows = g.loc[idx].sort_values('age_month')
        origin = pd.Period(rows.calendar_month.min(), freq='M') - 1
        t0 = origin_age(co, origin)
        ratio = np.asarray(cell_fn((ch, nd), co, t0, rows.age_month.to_numpy(int)), float)
        ratio = np.minimum.accumulate(np.clip(ratio, 0.0, 1.0))
        vals[g.index.get_indexer(rows.index)] = rows.origin_retained.to_numpy(float) * ratio
    g['predicted_retained'] = vals
    return g


def apply_cash(panel, g, origin):
    """Selected cash method from main.py: age profile recalibrated to last 3 months."""
    M.CASH_METHOD, M.RECAL_MONTHS = 'recalibrate', 3
    out = []
    for (ch, nd), rows in g.groupby(['channel','primary_need']):
        cell = panel[(panel.channel==ch)&(panel.primary_need==nd)]
        prof = M.cash_profile(cell, np.sort(rows.age_month.unique()), origin)
        mult = M.cash_multipliers(cell, rows.cohort, origin)
        net = prof.loc[rows.age_month, 'net_pm'].to_numpy()*mult
        con = np.minimum(net, prof.loc[rows.age_month, 'contribution_pm'].to_numpy()*mult)
        rows = rows.copy()
        rows['predicted_net_revenue'] = rows.predicted_retained*net
        rows['predicted_contribution'] = rows.predicted_retained*con
        out.append(rows)
    res = pd.concat(out)
    zero = res.predicted_retained <= 0
    res.loc[zero, ['predicted_retained','predicted_net_revenue','predicted_contribution']] = 0.0
    return res[M.KEYS+M.EXPOSURES+M.PREDICTIONS]


# ---------------------------------------------------------------- Model A: chain-ladder
def chain_ladder(panel, origin, tail):
    st = survivors_table(panel)
    def sums(keys):
        num, den = {}, {}
        for cell in keys:
            T = st[cell]
            for co, row in T.iterrows():
                oa = origin_age(co, origin)
                for a in range(0, oa):
                    s0, s1 = row.get(a), row.get(a+1)
                    if pd.notna(s0) and pd.notna(s1) and s0 > 0:
                        num[a] = num.get(a, 0.0)+s1; den[a] = den.get(a, 0.0)+s0
        return num, den
    n_all, d_all = sums(CELLS)
    a_max = max(d_all)
    f_all = {a: n_all[a]/d_all[a] for a in d_all}
    f_ch, f_cell = {}, {}
    for ch in ('partner','search'):
        n, d = sums([c for c in CELLS if c[0]==ch])
        for a in f_all:
            raw = n[a]/d[a] if d.get(a, 0) > 0 else f_all[a]
            z = d.get(a, 0)/(d.get(a, 0)+K_CRED_MEMBERS)
            f_ch[(ch, a)] = z*raw+(1-z)*f_all[a]
    for cell in CELLS:
        n, d = sums([cell])
        for a in f_all:
            raw = n[a]/d[a] if d.get(a, 0) > 0 else f_ch[(cell[0], a)]
            z = d.get(a, 0)/(d.get(a, 0)+K_CRED_MEMBERS)
            f_cell[(cell, a)] = min(1.0, z*raw+(1-z)*f_ch[(cell[0], a)])
    # Tail: channel-level sBG (cells pooled within channel -> shrinkage), or flat.
    sbg = {}
    if tail == 'sbg':
        for ch in ('partner','search'):
            pooled = (panel[panel.channel==ch].groupby(['cohort','age'], as_index=False)
                      [['n_retained']].sum())
            sbg[ch] = M.fit_sbg(pooled)
    def factor(cell, a, cohort):
        if a <= a_max:
            return f_cell[(cell, a)]
        if tail == 'flat':
            return float(np.mean([f_cell[(cell, x)] for x in range(a_max-2, a_max+1)]))
        aa, bb, lp = sbg[cell[0]]
        phi = np.exp(lp); mu = expit(aa+bb*M._cohort_z(pd.Series([cohort]))[0])
        t = a+1
        return 1.0 - mu*phi/(phi+t-1.0)
    def cell_fn(cell, cohort, t0, ages):
        out, s = [], 1.0
        a = t0
        for target in ages:
            while a < target:
                s *= factor(cell, a, cohort); a += 1
            out.append(s)
        return out
    return cell_fn, {'oldest_observed_transition_age': int(a_max)}


# ---------------------------------------------------------------- Model B: latent-class geometric
def renewal_records(panel):
    rec = []
    for i, cell in enumerate(CELLS):
        c = panel[(panel.channel==cell[0])&(panel.primary_need==cell[1])].sort_values(['cohort','age'])
        prev = c.groupby('cohort').n_retained.shift(1)
        d = c.assign(prev=prev).dropna(subset=['prev']); d = d[d.prev > 0]
        rec.append(pd.DataFrame({'cell': i, 't': d.age.to_numpy(float),
                                 'sp': d.prev.to_numpy(float), 's': d.n_retained.to_numpy(float)}))
    return pd.concat(rec, ignore_index=True)


def fit_lcg(panel, K):
    r = renewal_records(panel)
    ci, t, sp, s = r.cell.to_numpy(), r.t.to_numpy(), r.sp.to_numpy(), r.s.to_numpy()
    nC = len(CELLS)
    def unpack(x):
        theta = expit(x[:K])
        base = np.concatenate([[0.0], x[K:2*K-1]])
        delta = x[2*K-1:].reshape(nC, K-1)
        logits = base[None, :]+np.concatenate([np.zeros((nC, 1)), delta], axis=1)
        return theta, softmax(logits, axis=1), delta
    def nll(x):
        theta, pi, delta = unpack(x)
        S = lambda tt: (pi[ci]*(1-theta)[None, :]**tt[:, None]).sum(1)
        q = np.clip(S(t)/S(t-1), 1e-12, 1-1e-12)
        return -(s*np.log(q)+(sp-s)*np.log1p(-q)).sum()+0.5*np.sum((delta/LCG_DELTA_SD)**2)
    rng = np.random.default_rng(SEED)
    fixed = {2: [[0.03, 0.30], [0.08, 0.50], [0.02, 0.15]],
             3: [[0.02, 0.10, 0.40], [0.04, 0.20, 0.60], [0.01, 0.07, 0.30]]}[K]
    starts = [np.log(np.array(th)/(1-np.array(th))) for th in fixed]
    starts += [st+rng.normal(0, 0.3, K) for st in starts]
    best = None
    for th0 in starts:
        x0 = np.concatenate([th0, np.zeros(K-1), np.zeros(nC*(K-1))])
        res = minimize(nll, x0, method='L-BFGS-B')
        if best is None or res.fun < best.fun:
            best = res
    theta, pi, _ = unpack(best.x)
    order = np.argsort(theta)
    return theta[order], pi[:, order]


def lcg_model(panel, origin, K):
    theta, pi = fit_lcg(panel, K)
    def S(cell, tt):
        i = CELLS.index(cell)
        return (pi[i][None, :]*(1-theta)[None, :]**np.asarray(tt, float)[:, None]).sum(1)
    def cell_fn(cell, cohort, t0, ages):
        return S(cell, ages)/S(cell, [t0])[0]
    info = {'segment_churn': theta.round(4).tolist(),
            'cell_weights': {f'{c[0]}/{c[1]}': pi[i].round(3).tolist() for i, c in enumerate(CELLS)},
            'long_run_churn_ages_24_35': {f'{c[0]}/{c[1]}': float(np.mean(
                1-S(c, np.arange(24, 36))/S(c, np.arange(23, 35)))) for c in CELLS}}
    return cell_fn, info, S


# ---------------------------------------------------------------- Model C: recent churn flat
def recent_flat(panel, origin, months):
    lo = origin.ordinal-months+1
    def churn(keys):
        dec = ch = 0.0
        for cell in keys:
            c = panel[(panel.channel==cell[0])&(panel.primary_need==cell[1])].sort_values(['cohort','age'])
            prev = c.groupby('cohort').n_retained.shift(1)
            d = c.assign(prev=prev).dropna(subset=['prev'])
            cal = d.cohort.apply(lambda p: p.ordinal)+d.age
            d = d[(cal >= lo)&(cal <= origin.ordinal)&(d.prev > 0)]
            dec += d.prev.sum(); ch += (d.prev-d.n_retained).sum()
        return ch, dec
    ca, da = churn(CELLS); c_all = ca/da
    rate = {}
    for ch in ('partner','search'):
        c, d = churn([x for x in CELLS if x[0]==ch]); z = d/(d+K_CRED_DECISIONS)
        c_ch = z*(c/d if d else c_all)+(1-z)*c_all
        for cell in [x for x in CELLS if x[0]==ch]:
            c2, d2 = churn([cell]); z2 = d2/(d2+K_CRED_DECISIONS)
            rate[cell] = z2*(c2/d2 if d2 else c_ch)+(1-z2)*c_ch
    def cell_fn(cell, cohort, t0, ages):
        return (1-rate[cell])**(np.asarray(ages)-t0)
    return cell_fn, {'flat_monthly_churn': {f'{c[0]}/{c[1]}': round(float(rate[c]), 4) for c in CELLS}}


# ---------------------------------------------------------------- existing models via main.py
def main_model(name):
    def run(tables, grid):
        M.RETENTION_MODEL, M.CASH_METHOD, M.RECAL_MONTHS = name, 'recalibrate', 3
        try:
            return M.validate_output(M.predict(tables, grid.copy(deep=True), seed=SEED), grid)
        finally:
            M.RETENTION_MODEL = 'sbg'
    return run


# ---------------------------------------------------------------- scoring
def scores(pred, actual):
    keys = ['cohort_month','channel','primary_need','calendar_month']
    m = pred.merge(actual, on=keys, validate='one_to_one')
    pm = np.where(m.n_retained > 0, m.contribution/m.n_retained.where(m.n_retained > 0, 1), 0.0)
    m['ret_only_contribution'] = m.predicted_retained*pm
    cm = m.groupby(['cohort_month','calendar_month'])[['predicted_contribution','contribution','ret_only_contribution']].sum()
    den = cm.contribution.abs().sum()
    err = (cm.predicted_contribution-cm.contribution).abs()
    return ({'contribution_wape': float(err.sum()/den),
             'retained_member_wape': float((m.predicted_retained-m.n_retained).abs().sum()/m.n_retained.sum()),
             'retained_signed_bias': float((m.predicted_retained.sum()-m.n_retained.sum())/m.n_retained.sum()),
             'retention_only_contribution_wape': float((cm.ret_only_contribution-cm.contribution).abs().sum()/den),
             'signed_bias': float((cm.predicted_contribution.sum()-cm.contribution.sum())/den)},
            err.groupby(level=0).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=ROOT/'experiments'/'retention_round2')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    contract = json.loads((ROOT/'submission-contract.json').read_text())
    tables = M.load_tables(args.data_dir, contract)
    full = M.build_panel(tables, pd.Period('2025-12', freq='M'))
    full['cohort_month'] = full.cohort.apply(lambda c: c.to_timestamp().strftime('%Y-%m-%d'))
    full['calendar_month'] = full.apply(lambda r: (r.cohort+int(r.age)).to_timestamp().strftime('%Y-%m-%d'), axis=1)
    actual = full[['cohort_month','channel','primary_need','calendar_month','n_retained','contribution']]

    def new_model(builder):
        def run(tables_t, grid, panel, origin):
            fn, info = builder(panel, origin)[:2]
            g = project(grid, fn)
            return M.validate_output(apply_cash(panel, g, origin), grid), info
        return run
    MODELS = {
        'sBG (cell, cohort trend)': ('main', main_model('sbg')),
        'BdW': ('main', main_model('bdw')),
        'Logistic hazard': ('main', main_model('logit_hazard')),
        'A: chain-ladder, sBG tail': ('new', new_model(lambda p, o: chain_ladder(p, o, 'sbg'))),
        'A: chain-ladder, flat tail': ('new', new_model(lambda p, o: chain_ladder(p, o, 'flat'))),
        'B: latent-class geometric K=2': ('new', new_model(lambda p, o: lcg_model(p, o, 2))),
        'B: latent-class geometric K=3': ('new', new_model(lambda p, o: lcg_model(p, o, 3))),
        'C: recent churn flat, 3m': ('new', new_model(lambda p, o: recent_flat(p, o, 3))),
        'C: recent churn flat, 6m': ('new', new_model(lambda p, o: recent_flat(p, o, 6))),
    }
    results = {name: {'cutoffs': {}, 'cohort_error': {}, 'info': {}, 'runtime_s': []} for name in MODELS}
    for o, h in B.ORIGINS:
        origin = pd.Period(o, freq='M')
        tt = B.truncate(tables, origin)
        panel = M.build_panel(tt, origin)
        grid = B.make_grid(panel, origin, h)
        for name, (kind, run) in MODELS.items():
            t0 = time.perf_counter()
            if kind == 'main':
                pred, info = run(tt, grid), {}
            else:
                pred, info = run(tt, grid, panel, origin)
            results[name]['runtime_s'].append(time.perf_counter()-t0)
            sc, by_cohort = scores(pred, actual)
            results[name]['cutoffs'][o] = sc
            results[name]['cohort_error'][o] = by_cohort.to_dict()
            if info:
                results[name]['info'][o] = info
        print('done', o, flush=True)

    # Full-data fits (released observations through Dec 2025) for curves and reporting.
    origin = pd.Period('2025-12', freq='M')
    curves = {}
    jan24 = pd.Period('2024-01', freq='M')
    ages = np.arange(23, 36)
    for name, (kind, run) in MODELS.items():
        t0 = time.perf_counter()
        if kind == 'main':
            mdl = {'sBG (cell, cohort trend)':'sbg','BdW':'bdw','Logistic hazard':'logit_hazard'}[name]
            M.RETENTION_MODEL = mdl
            fns = {}
            for cell in CELLS:
                cp = full[(full.channel==cell[0])&(full.primary_need==cell[1])]
                params = M.fit_retention(cp)
                z = M._cohort_z(pd.Series([jan24]))[0]
                q = M.renewal_prob(params, ages[1:].astype(float), np.full(len(ages)-1, z))
                fns[cell] = np.concatenate([[1.0], np.cumprod(q)])
            M.RETENTION_MODEL = 'sbg'
        else:
            builder = {'A: chain-ladder, sBG tail': lambda p, o: chain_ladder(p, o, 'sbg'),
                       'A: chain-ladder, flat tail': lambda p, o: chain_ladder(p, o, 'flat'),
                       'B: latent-class geometric K=2': lambda p, o: lcg_model(p, o, 2),
                       'B: latent-class geometric K=3': lambda p, o: lcg_model(p, o, 3),
                       'C: recent churn flat, 3m': lambda p, o: recent_flat(p, o, 3),
                       'C: recent churn flat, 6m': lambda p, o: recent_flat(p, o, 6)}[name]
            fn, info = builder(full, origin)[:2]
            results[name]['info']['2025-12 (full released data)'] = info
            fns = {cell: np.asarray(fn(cell, jan24, 23, ages), float) for cell in CELLS}
        results[name]['runtime_full_fit_s'] = time.perf_counter()-t0
        curves[name] = {f'{c[0]}/{c[1]}': fns[c].tolist() for c in CELLS}
    obs = {}
    for cell in CELLS:
        r = full[(full.channel==cell[0])&(full.primary_need==cell[1])&(full.cohort==jan24)].sort_values('age')
        obs[f'{cell[0]}/{cell[1]}'] = {'age': r.age.tolist(), 'share': (r.n_retained/r.n_acquired).tolist(),
                                       'n_acquired': int(r.n_acquired.iloc[0])}
    (args.out/'results.json').write_text(json.dumps({'results': results, 'curves_from_age23': curves,
                                                     'ages': ages.tolist(), 'observed_jan2024': obs,
                                                     'settings': {'K_CRED_MEMBERS': K_CRED_MEMBERS,
                                                                  'K_CRED_DECISIONS': K_CRED_DECISIONS,
                                                                  'LCG_DELTA_SD': LCG_DELTA_SD, 'seed': SEED}},
                                                    indent=1, default=str)+'\n')
    print('wrote', args.out/'results.json')


if __name__ == '__main__':
    main()
