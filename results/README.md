# results/

Frozen inputs and exported outputs of the original study. The notebooks and tests read only from this folder, so every published number can be checked without raw data.

## Frozen inputs (fixed before the test periods were opened)

| file | contents | used by |
|---|---|---|
| `frozen_model.json` | The forecast model used in both test periods: feature list, standardization means and standard deviations, ridge coefficients, λ, the OFI benchmark, the taker threshold, and the parameters of the execution study. | `scripts/run_day.py`, `scripts/fit_model.py`, notebook 1 |

## Training and validation (Oct 1 – Nov 9, 2025)

| file | contents | used by |
|---|---|---|
| `validation.json` | Model selection: validation MSE for each λ on the two forward folds, pooled and daily validation scores, and the shuffled-label diagnostic. | notebook 1, `scripts/fit_model.py` |
| `training_check.json` | Output of `scripts/fit_model.py` rerun on the 40 training days from raw data, compared field by field with `frozen_model.json` (`all_identical: true`). | notebook 1 |
| `imbalance_bins.csv` | Mean 5-second return by decile of best-level imbalance and of OFI on the training days (`mean_x`, `mean_y_bp`, standard error, count). | notebook 1, `scripts/hero_figure.py` |
| `sample_hour.npz` | One training hour (Oct 15, 14:00–15:00 UTC) on the one-second grid: mid price, the eight features, target and eligibility. | notebook 1 |

## Test results

| file | contents | used by |
|---|---|---|
| `sealed_test_days.csv` | First test, Nov 10–29, 2025, one row per day (20 rows). `D1`, `D2`: daily forecast gain over the zero and OFI forecasts (MSE difference, bp²). `lat_*` / `lot2_*`: taker P&L before fees (`pi`), traded notional (`n`) and round trips (`trips`) for each latency and order-size scenario. | notebooks 1–2, tests |
| `execution_days.csv` | Second test, Dec 1, 2025 – Jan 14, 2026, one row per day (45 rows). `D4`, `D5`, `D7`: daily mean execution saving of signal timing against TWAP, random timing and the shifted-forecast placebo (bp). `Q6`: daily forecast gain over the zero forecast. `eligible`, `unresolved_parents`: parent orders run and left incomplete at the deadline. | notebooks 1, 3, tests |
| `summary.json` | Every number quoted in the README and notebooks: point estimates, bootstrap and t-intervals, classifications under the pre-set rules, and descriptive checks. `tests/test_model_stats.py` recomputes the intervals from the two CSV files above and checks them against this file. | all notebooks, tests, `scripts/hero_figure.py` |

## Reproduction checks

| file | contents | used by |
|---|---|---|
| `reference_days.json` | For each of the 65 test days, the SHA-256 of the one-second grid, the book store and the fill logs, plus the daily results. `scripts/run_day.py` compares a fresh run against these values. | `scripts/run_day.py` |
| `sample/2025-11-10.json`, `sample/2025-12-01.json` | Output of `scripts/run_day.py` on one day of each test period, rebuilt from raw data with this repository. Both match `reference_days.json` byte for byte. | documentation |
| `provenance.json` | SHA-256 of the original-study files each export was copied from, under their names in the development repository. | documentation |

## Naming

`D1`–`D7` are daily quantities and `Q1`–`Q7` their means over a test period. Returns and costs are in basis points (1 bp = 0.01%), MSE in bp².

| ID | quantity | test period |
|---|---|---|
| Q1 | forecast gain over the zero forecast (MSE difference) | first (Nov 10–29) |
| Q2 | forecast gain over the one-variable OFI model ² | first |
| Q3 | break-even fee of the taker policy | first |
| Q4 | execution saving against TWAP | second (Dec 1 – Jan 14) |
| Q5 | execution saving against random timing | second |
| Q6 | forecast gain over the zero forecast | second |
| Q7 | execution saving against the shifted-forecast placebo | second |

² The OFI benchmark uses raw OFI linearly. OFI's relation to returns is S-shaped and fat-tailed, so this comparison is conservative.
