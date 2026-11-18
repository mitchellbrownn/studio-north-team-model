# Studio North team forecast: cell-level sBG with cohort trend

Forecasts expected retained members, net revenue and contribution for each existing
acquisition cohort × channel × stated need, for every calendar month of
**January–December 2026** (1,152 grid rows, 24 cohorts, 96 cells). This is
remaining value over the next twelve calendar months, not the lab report's LTV over
each cohort's first twelve receipts. Acquisition cost is not subtracted again.

This model is **not** the released hierarchical sBG + Gamma reference. It is a
lighter, deterministic alternative chosen by chronological validation (below).

## Run

```text
uv sync --frozen
uv run --frozen python main.py --data-dir /path/to/released/data --prediction-grid prediction_grid.csv --output outputs/predictions.csv --seed 20261116
```

`main.py` exposes `predict(tables, prediction_grid, *, seed)` with the contract's
interface. It reads only the eight observed tables and the public grid, fits from
scratch on each run, uses no network, GPU or cached artifacts, and writes a
receipt with input, code, lock and output hashes.

## Model

**Retention: shifted-Beta-Geometric (Fader & Hardie, 2007) per channel × need cell.**
Studio North is contractual and churn is absorbing. Within a cell, each member has a
persistent monthly churn propensity drawn from Beta(α, β). Written as mean μ and
concentration φ (α = μφ, β = (1 − μ)φ), the churn hazard at renewal t ≥ 1 is
μφ / (φ + t − 1). It falls with age because fragile members leave first (survivor
selection), even though no individual's propensity changes.

- μ varies by acquisition cohort through a linear trend:
  μ_k = logistic(a + b·z_k), with z_k = (cohort_index − 11.5)/11.5 (January 2024 = 0).
- φ is shared within the cell. That gives 3 parameters per cell, 12 in total.
- Fit: penalized maximum likelihood on observed renewal decisions only. Survivors at
  age t ~ Binomial(survivors at t − 1, 1 − hazard_t), each decision counted once.
  There are weak priors: b ~ N(0, 1) and log φ ~ N(log 4, 2). L-BFGS-B uses three
  fixed starting points; the method is deterministic, so the seed is unused.
- Forecast: starting from each cell's observed survivors at December 2025
  (`origin_retained`), multiply by (1 − hazard) for each future age. The sBG hazard
  sequence applies directly to survivors, so no re-fitting at the origin is needed.

**Cash: age-profiled amounts per retained member.** For each cell, observed net
revenue and contribution per retained member are computed by customer age, pooling
the neighbouring ages (±1) to reduce noise. Forecast receipts = expected retained ×
per-member amount at that age. Net revenue includes discounts and refunds;
contribution also deducts recorded variable costs. Contribution is capped at net
revenue and floored at zero, and cells with zero expected members get zero receipts.

**Extrapolation beyond observed ages.** The oldest cohort is forecast at ages
24–35, but the largest observed age is 23. Retention extrapolates through the sBG
hazard, which keeps declining slowly. That is the model's main structural
assumption, and it is checked only up to age 23. Cash per member beyond age 23
reuses the cell's average over its last six observed ages, which assumes
late-tenure spending stays flat.

## Historical validation

`scripts/backtest.py` reproduces the scoring rule on observed history only. For
each origin, it truncates every input to what was available then (transactions
before the cutoff, renewals already due), rebuilds the grid from that truncated
history, forecasts the following months, and scores them against the months later
observed. The score is the challenge's primary metric: contribution WAPE after
summing cells to cohort × calendar month. No 2026 outcomes exist in this
repository or were used.

| Origin | Horizon | Score points | **sBG contribution WAPE** | Example constant-hazard WAPE | sBG retention cell WAPE | sBG signed bias |
|---|---:|---:|---:|---:|---:|---:|
| Dec 2024 | 12 months | 144 | **7.5%** | 37.9% | 11.7% | −4.5% |
| Mar 2025 | 9 months | 135 | **7.1%** | 32.2% | 7.5% | −6.1% |
| Jun 2025 | 6 months | 108 | **8.1%** | 27.0% | 5.9% | −7.6% |

Full results are in `validation/backtest.json`. The baseline is the scaffold's
pooled constant-hazard example, kept unchanged in
`scripts/example_constant_hazard.py`.

**Known limitations.**
- **Under-forecast:** the sBG model is consistently a little low on contribution
  (−4% to −8%). Cash per member has risen over calendar time, and an age-only
  profile doesn't capture that drift.
- **Narrow validation range:** the backtests never require ages above 23, and the
  12-month horizon only resembles the final task for cohorts at least a year old.
- **Point forecasts only:** these are point expectations, with no uncertainty
  intervals.
- **No accuracy guarantee:** format validity and backtest accuracy do not establish
  private-holdout accuracy.

## Final settings and reproducibility evidence

- **Settings:** trend scale 11.5; trend prior sd 1.0; log-φ prior N(log 4, 2);
  cash pooling window ±1 age; 6 late ages used for extrapolation; seed 20261116
  (accepted, unused).
- **Environment:** Python 3.12.16, managed by uv; pandas, numpy and scipy, locked
  in `uv.lock`.
- **Runtime:** model time about 1.3 s, and about 6 s per run including startup, on
  one Windows laptop CPU. That is far inside the 60-minute, 4-core, 16 GiB budget.
- **Reproducibility:** two runs from a clean `uv sync --frozen` environment produced
  byte-identical predictions (SHA-256
  `cbe63c06fa2edf966ee166f6a643a3b5f2118c8fa32473f1f5d795f5abbef790`). Both
  passed the lab's public `validate-submission` check (1,152 rows, no future
  outcomes read). Input data manifest SHA-256 `557d88a8…f1e43`; prediction grid
  SHA-256 `c3d4d1e0…e75a2`.

The authoritative interface, limits and scoring rule are in `submission-contract.json`.
