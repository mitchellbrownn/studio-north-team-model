# Studio North team forecast: shrunk Beta-discrete-Weibull retention with drift-adjusted cash

Forecasts expected retained members, net revenue and contribution for each existing
acquisition cohort × channel × stated need, for every calendar month of
**January–December 2026** (1,152 grid rows, 24 cohorts, 96 cells). This is
remaining value over the next twelve calendar months, not the lab report's LTV over
each cohort's first twelve receipts. Acquisition cost is not subtracted again.

## Run

```text
uv sync --frozen
uv run --frozen python main.py --data-dir /path/to/released/data --prediction-grid prediction_grid.csv --output outputs/predictions.csv --seed 20261116
```

`main.py` exposes `predict(tables, prediction_grid, *, seed)` exactly as specified
in `submission-contract.json` and the published challenge specification. It reads
only the eight released observed tables and the public grid, fits from scratch on
each run, uses no network, GPU or cached artifacts, and writes a receipt with
input, code, lock and output hashes. The model is deterministic, so the seed is
accepted but unused.

## How this differs from the reference sBG + Gamma model

| | Released reference | This submission |
|---|---|---|
| Retention model | Shifted-Beta-Geometric (sBG): constant individual churn propensity | **Beta-discrete-Weibull (BdW):** individual propensity may rise or fall with tenure |
| Heterogeneity across cells | Hierarchical Bayesian, with cohort random effects | Joint fit across cells with **cell → channel → overall shrinkage**, plus a linear cohort trend |
| Estimation | NumPyro NUTS posterior sampling | Penalized maximum likelihood, deterministic, ≈4 s |
| Revenue | Gamma model of gross revenue per member | **Empirical cash per member by age**, recalibrated to recent levels, plus a **pooled drift** projected forward |

The released reference is unchanged and is not used by this code.

## Model

**Retention: Beta-discrete-Weibull** (Fader, Hardie, Liu, Davin & Steenburgh, 2018,
"How to Project Customer Retention" Revisited: The Role of Duration Dependence,
*Journal of Interactive Marketing*).
Studio North is contractual and churn is absorbing.
- **Propensities:** each member's churn propensity Θ is drawn from Beta(α, β), with
  mean μ and concentration φ (α = μφ, β = (1 − μ)φ).
- **Survival:** a member survives t renewals with probability (1 − Θ)^(t^c), so the
  cohort survival curve is S(t) = B(α, β + t^c) / B(α, β), and the chance of
  renewing at age t is S(t)/S(t − 1).
- **The shape c:** c = 1 is exactly sBG. With c < 1, an individual's churn
  propensity falls with tenure; with c > 1, it rises. So the model separates two
  reasons aggregate churn falls with age: tenure effects within each member, and
  survivor selection across members.

Parameters and fitting:
- **Cohort trend:** μ_k = logistic(a + b·z_k), with z_k = (cohort_index − 11.5)/11.5
  (January 2024 = 0). Each cell has θ = (a, b, log φ, log c).
- **Shrinkage (no free per-cell parameter):** θ_cell = θ_overall + γ_channel + δ_cell,
  with γ ~ N(0, 0.5²) and δ ~ N(0, 0.3²) elementwise. Overall priors:
  b ~ N(0, 1), log φ ~ N(log 4, 2), log c ~ N(0, 0.5).
- **Fit:** all four cells are fitted jointly by penalized maximum likelihood on
  observed renewal decisions. Survivors at age t ~ Binomial(survivors at t − 1,
  renewal probability); each decision is counted once. L-BFGS-B uses three fixed
  starting points.
- **Fitted shapes:** on all released data, c is 0.89–0.93 for the search cells and
  1.07–1.12 for the partner cells. Across the backtest cutoffs and the full data, c
ranged from 0.77 to 1.12, so
  the shape is only weakly identified with 24 months of history.
- **Forecast:** start from each cell's observed survivors at December 2025
  (`origin_retained`) and multiply the renewal probabilities for each future age.

**Cash per retained member:**
1. **Age profile.** For each cell, observed net revenue and contribution per
   retained member by customer age, pooling ages ±1. Ages beyond those observed use
   the cell's average over its last six observed ages.
2. **Recalibration.** The profile is scaled by observed ÷ profile-implied
   contribution over the cell's last three observed months. That ratio is
   credibility-weighted toward the channel ratio, and the channel toward the
   overall ratio, with Z = member-months ÷ (member-months + 300). It is bounded to
   [0.5, 2.0].
3. **Pooled drift.** Contribution per member has been rising relative to the age
   profile. A weighted log-linear fit over the last 12 months, **pooled across all
   cells**, gives a monthly growth rate g, capped at ±2%. A forecast h months ahead
   is multiplied by (1 + g)^h. On all released data g = 0.72% per month; at the
   backtest cutoffs it was 0.96%, 0.79% and 0.81%.

Contribution is capped at net revenue and floored at zero. Cells with zero expected
members get zero receipts.

**Extrapolation beyond observed ages.** The oldest cohort is forecast at ages
24–35, but the largest observed age is 23. Retention there relies on the BdW
hazard. Cash relies on the late-age profile plus drift, which assumes spending per
member keeps growing at its recent pooled rate through December 2026 (about +8% by
year end).

## Historical validation

`scripts/backtest.py` reproduces the challenge's primary score on observed history
only. For each origin, it truncates every input to what was available then
(transactions before the cutoff, renewals already due), rebuilds the grid from that
truncated history, forecasts the following months, and scores them against the
months later observed. The score is contribution WAPE after summing cells to
cohort × calendar month. No 2026 outcomes exist in this repository or were used.

**Selected model:**

| Origin | Horizon | Score points | **Contribution WAPE** | Signed bias |
|---|---:|---:|---:|---:|
| Dec 2024 | 12 months | 144 | **6.50%** | +2.6% |
| Mar 2025 | 9 months | 135 | **4.98%** | −1.0% |
| Jun 2025 | 6 months | 108 | **4.26%** | −1.6% |
| **Mean** | | | **5.24%** | **0.0%** |

For comparison, on the same origins the scaffold's constant-hazard example scores
37.9 / 32.2 / 27.0%. The sBG version of this pipeline scores 6.91 / 5.81 / 5.50%.

### How the model was chosen (all candidates reported)

All settings were fixed before each round was scored.
- **Isolating retention:** "retention-only WAPE" scores each model's retained
  members multiplied by the contribution per surviving member that was actually
  realized, which isolates retention error from revenue error.
- **Shrinkage:** credibility constants were 100 at-risk members for chain-ladder
  factors and 200 decisions for recent churn rates.
- **Scripts and outputs:** `experiments/`, with figures in
  `experiments/retention_round2/`.

**Round 1: cash specification** (with sBG retention). A diagnostic showed retention
was roughly unbiased, but cash per member was under-forecast by 6.5–10% in every
cell.

| Cash specification | Dec 2024 | Mar 2025 | Jun 2025 | Mean bias |
|---|---:|---:|---:|---:|
| Age profile only | 7.5% | 7.1% | 8.1% | −6.0% |
| Recency-weighted, half-life 3 / 6 / 12 months | 7.1 / 7.3 / 7.4% | 6.7 / 6.9 / 7.0% | 7.0 / 7.4 / 7.7% | −5.1 / −5.5 / −5.8% |
| Cohort trend in cash | 7.8% | 6.5% | 6.5% | −3.7% |
| Recalibrate to last 3 months | 6.9% | 5.8% | 5.5% | −3.3% |
| Recalibrate, last 1 / 6 / 12 months | 6.8 / 7.1 / 7.4% | 5.7 / 6.2 / 7.0% | 4.9 / 6.2 / 7.4% | −2.4 / −4.3 / −5.7% |

**Round 2: retention model** (with 3-month recalibrated cash).

| Retention model | Mean contribution WAPE | Retained-member WAPE | **Retention-only WAPE** |
|---|---:|---:|---:|
| sBG, per cell | 6.07% | 8.40% | 4.22% |
| **BdW, per cell** | 6.64% | 8.09% | **3.84%** |
| Logistic discrete-time hazard | 7.33% | 8.51% | 3.98% |
| Chain-ladder, credibility-weighted, sBG tail | 9.16% | 8.23% | 4.77% |
| Chain-ladder, flat tail | 11.49% | 9.43% | 7.07% |
| Latent-class geometric, K = 2 | 9.14% | 8.42% | 4.62% |
| Latent-class geometric, K = 3 | 8.55% | 7.85% | 4.23% |
| Recent churn held flat, 3 / 6 months | 17.12 / 18.27% | 14.47 / 15.34% | 13.55 / 14.73% |

BdW had the lowest retention-only error. sBG's lower total error came partly from
**offsetting errors**: it over-forecast retention (+6.3% at Dec 2024) while the cash
model under-forecast. At the Dec 2024 cutoff the oldest observed transition was age
10, so chain-ladder there depends heavily on its tail assumption.

**Round 3: shrinkage and revenue, with BdW.**

| Variant | Dec 2024 | Mar 2025 | Jun 2025 | Mean | Bias |
|---|---:|---:|---:|---:|---:|
| BdW per cell + recalibrate 3m | 7.70% | 6.62% | 5.61% | 6.64% | −4.2% |
| BdW shrunk + recalibrate 3m | 7.70% | 6.59% | 5.52% | 6.60% | −4.1% |
| … + shrunk recalibration | 7.70% | 6.59% | 5.52% | 6.60% | −4.1% |
| **… + pooled drift, 12 months (selected)** | **6.50%** | **4.98%** | **4.26%** | **5.24%** | **0.0%** |
| BdW shrunk + recalibrate 1m shrunk | 7.53% | 6.48% | 4.87% | 6.29% | −3.1% |
| … + pooled drift, 12 months | 6.69% | 4.94% | 4.11% | 5.25% | +1.0% |

Shrinking BdW across cells barely changed accuracy, so it buys robustness at no
cost. Shrinking the recalibration ratios moved them by at most 0.4%, because the
cells' ratios were already close (1.02–1.07). The drift term removed the remaining
bias and improved every origin.

**Known limitations.**
- **Selection risk:** model choices were made on the same three backtest origins,
  so 5.24% probably overstates out-of-sample accuracy somewhat.
- **The drift assumption:** the drift is the main new assumption. If spending per
  member stops growing, 2026 contribution will be over-forecast; the ±2% cap limits
  this.
- **Narrow validation range:** the backtests never require ages above 23, and the
  12-month horizon only resembles the final task for cohorts at least a year old.
- **Point forecasts only:** these are point expectations, with no predictive
  intervals.
- **No accuracy guarantee:** format validity and backtest accuracy do not establish
  private-holdout accuracy.

## Final settings and reproducibility evidence

- **Settings:**
  - retention `bdw_pooled`; trend scale 11.5; priors b ~ N(0, 1),
    log φ ~ N(log 4, 2), log c ~ N(0, 0.5); shrinkage sd: channel 0.5, cell 0.3;
  - cash pooling ±1 age, 6 late ages; recalibration over 3 months with credibility
    constant 300, bounded [0.5, 2.0]; drift over 12 months, capped ±2%;
  - seed 20261116 (accepted, unused).
- **Environment:** Python 3.12.16, managed by uv; pandas, numpy and scipy, locked
  in `uv.lock`.
- **Runtime:** model time about 4 s, and about 8 s per run including startup, on
  one Windows laptop CPU. That is far inside the 60-minute, 4-core, 16 GiB budget.
- **Reproducibility:** two runs from a clean `uv sync --frozen` environment produced
  byte-identical predictions (SHA-256
  `67f58e665ddca1201a4964a6c480938dc57082bf937d1e4426905f277a42c9f7`). Both
  passed the lab's public `validate-submission` check (1,152 rows, no future
  outcomes read). Input data manifest SHA-256 `557d88a8…f1e43`; prediction grid
  SHA-256 `c3d4d1e0…e75a2`.
- **Experiment scripts:** they need matplotlib only for figures:
  `uv run --frozen --with matplotlib==3.10.3 python experiments/plot_round2.py`.

The authoritative interface, limits and scoring rule are in `submission-contract.json`.
