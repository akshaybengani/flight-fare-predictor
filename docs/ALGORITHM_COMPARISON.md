# Algorithm comparison

Ten algorithms on **exactly** the data, features, splits, target (log price) and metric used for the current model (`src/train.py`).
Accuracy = 100 − MAPE. Machine-readable results: `models/algorithms.json` (seed spread: `models/algorithms_noise.json`, Random Forest study: `models/rf_study.json`). Re-run with `python src/compare_models.py`.

Scenarios (same as `models/metrics.json`):

| | Scenario | Test rows |
|---|---|---|
| A | Same period: random 20% of 2023 rows held out (other days of the same flights are in training) | 89,044 |
| E | Recalibrated: train on 2022 + 10% of 2023 flight-days, test on the other 2023 flight-days | 400,049 |
| C | Unseen period: train on 2022 only, test on all 2023 (no period / exact-time features) | 445,222 |

## Results (18-feature baseline set)

Accuracy (100 − MAPE, %). **Bold** = best in column.

| Algorithm | A same-period | E recalibrated | C unseen-period |
|---|---|---|---|
| HistGradientBoosting (current model) | 93.04 | 87.00 | 66.17 |
| LightGBM | 93.19 | 87.22 | 60.87 |
| XGBoost | **93.42** | 86.88 | **67.74** |
| CatBoost | 91.47 | 86.25 | 67.39 |
| Random Forest | 92.01 | 86.52 | 43.93 |
| Extra Trees | 92.64 | 87.12 | 57.33 |
| Ridge (linear) | 81.46 | 72.43 | 66.27 |
| Median lookup (naive) | 75.42 | 67.39 | 61.75 |
| Neural net (MLP) | 86.80 | 80.46 | 47.29 |
| Blend (equal average of 3, see below) | 93.41 | **87.60** | 67.13 |

Within-10% of the true fare (%) and MAE (₹) for the main contenders:

| Algorithm | A within-10% | A MAE | E within-10% | E MAE | C within-10% | C MAE |
|---|---|---|---|---|---|---|
| HGB (current) | 77.6 | 1,500 | 55.1 | 2,717 | 19.9 | 5,889 |
| LightGBM | 78.4 | 1,465 | 55.8 | 2,704 | 19.2 | 6,848 |
| XGBoost | 79.4 | 1,395 | 54.5 | 2,759 | 21.3 | 5,798 |
| Random Forest | 72.7 | 1,820 | 54.8 | 2,831 | 19.0 | 8,553 |
| Blend | 79.1 | 1,413 | 57.2 | 2,638 | 21.2 | 5,547 |

RMSE and R² for every algorithm are in `algorithms.json`.

The current model reproduces: 93.04 / 87.00 / 66.17 (identical to `metrics.json`).

**Blend**: equal-weight average (in log price) of the three algorithms with the best *validation* accuracy on the tuning split (not the test fold): A = XGBoost + LightGBM + HGB; E = Extra Trees + LightGBM + XGBoost; C = Extra Trees + Random Forest + HGB.

## Who wins, and is it more than noise?

| Scenario | Winner | Margin over current model |
|---|---|---|
| A | XGBoost 93.42 (blend 93.41, LightGBM 93.19) | +0.38 pts |
| E | Blend 87.60 (LightGBM 87.22 best single) | +0.60 pts (LightGBM +0.22) |
| C | XGBoost 67.74 (CatBoost 67.39, blend 67.13) | +1.57 pts |

**Noise.** I re-fitted the top three by mean accuracy (XGBoost, HGB, CatBoost) with two more model seeds (same splits, same tuned parameters):

| | A | E | C |
|---|---|---|---|
| HGB, 3 seeds: mean ± sd | 93.05 ± 0.01 | 86.97 ± 0.03 | 66.07 ± 0.09 |
| CatBoost, 3 seeds | 91.48 ± 0.01 | 86.26 ± 0.03 | 64.89 ± 2.17 (range 63.6–67.4) |
| XGBoost, 3 seeds | identical every seed (0.00) | identical | identical |

- XGBoost as configured has no random element (full column use, no row sampling), so its seed spread is exactly zero. That does **not** mean it is noise-free: the real uncertainty (which rows land in train/test, tuning luck) is not captured by a model seed. HGB's ±0.01–0.09 is the best guide to seed-level noise.
- On A and E the gaps between the top boosters (0.2–0.6 pts) are larger than seed noise (≤0.1) but small in absolute terms and from a **single** split and a single tuning draw; treat them as "roughly equal, XGBoost/LightGBM very slightly ahead", not as a decisive win.
- **C is not reliable for ranking.** CatBoost swings by 3.8 pts between seeds, LightGBM scores 60.9 while XGBoost scores 67.7 with near-identical settings, and a plain linear Ridge (66.27) matches the current model. Differences on C of a couple of points are not evidence of a better algorithm.

## Speed and size (scenario A; training seconds are for the final fit and exclude tuning)

| Algorithm | Train s | Predict ms / 1,000 rows | Model MB |
|---|---|---|---|
| HGB (current) | 399 | 37 | 106 |
| LightGBM | 163 | 157 | 164 |
| XGBoost | 288 | 32 | 134 |
| CatBoost | 557 | 2.7 | 50 |
| Random Forest | 37 | 10 | 920 |
| Extra Trees | 23 | 18 | 2,057 |
| Ridge | 0.4 | 0.3 | 0.01 |
| Median lookup | 0.1 | 0.3 | 0.02 |
| MLP | 11 | 0.8 | 0.4 |
| Blend (3 models) | sum of members | sum | sum |

Timing was measured on a shared, heavily loaded MacBook (4 threads), so treat seconds as relative. Training seconds for scenarios E/C in the checkpoints include periods when the laptop slept and are not meaningful; the A figures are clean. Prediction time excludes feature encoding.

## Random Forest (requested by the portfolio manager)

- **Accuracy vs the current model and boosters:** A 92.01 (current 93.04, XGBoost 93.42) → about 1 pt behind; E 86.52 (current 87.00) → 0.5 pt behind; **C 43.93 (current 66.17) → badly worse**. Forests cannot extrapolate: when 2023 prices sit at a different level than 2022, a forest can only return values it saw in training, while boosters (and even the linear model) can adjust through the route-airline price-level feature. Extra Trees fares better but is still far behind (57.3).
- **Train time:** about 37 s for 120 trees on scenario A, far faster than any booster (HGB 399 s). Prediction is fast too (10 ms / 1,000 rows).
- **Size is the practical problem:** 920 MB for the tuned forest (120 trees, `min_samples_leaf=3`), versus 106 MB for the current model. It is too big for a git repo or a small server, and slow to load. Trade-offs measured on scenario A (accuracy %, MB):

| Variant | Accuracy | Within-10% | Train s | MB |
|---|---|---|---|---|
| Tuned (120 trees, leaf 3) | 92.01 | 72.7 | 35 | 920 |
| 60 trees | 91.98 | 72.6 | 18 | 460 |
| **30 trees** | **91.91** | 72.4 | 9 | **230** |
| `min_samples_leaf=5` | 91.69 | 71.4 | 31 | 563 |
| `min_samples_leaf=10` | 91.07 | 69.0 | 30 | 287 |
| `min_samples_leaf=20` | 90.27 | 66.1 | 28 | 145 |
| `max_samples=0.5` | 91.49 | 70.6 | 21 | 588 |
| `max_samples=0.25` | 90.79 | 68.0 | 12 | 340 |
| 30 trees + leaf 10 + `max_samples=0.5` | 90.39 | 66.6 | 6 | 45 |

  Fewer trees is by far the cheapest way to shrink it: 30 trees loses only 0.1 pt for a 4× smaller file (230 MB). `min_samples_leaf` and `max_samples` cost much more accuracy per MB saved. Even at 30 trees the model is twice the size of the current one and still behind on A and badly behind on C.
- **Feature importances** (scenario A forest, 120 trees). Impurity importance: `te_flight` 0.67, `te_route_airline` 0.13, `class` 0.07, `duration_h` 0.05, `days_left` 0.03, everything else ≤ 0.01. Held-out permutation importance (accuracy points lost when shuffled, 30k test rows): `te_flight` 80.0, `te_route_airline` 15.7, `days_left` 9.4, `duration_h` 8.4, `distance_km` 4.3, `destination` 2.9, `class` 2.6, `route_supply` 2.4, `stops` 2.0, `arr_hour` 1.6; `period` is 0 (constant within scenario A). In plain terms, the forest, like the boosters, mostly leans on "what does this specific flight usually cost" and "what does this route-airline usually cost", then on how early you book and how long the flight is. Full list in `models/rf_study.json`.
- **Verdict:** a reasonable, quick-to-train second opinion within the same period, but not a replacement: it is bigger, slightly less accurate on A/E, and unusable for the unseen-period case.

## Tuning protocol (equal effort)

- Per scenario, per algorithm: **6 random-search trials** (same RNG seed) on ≤150,000 sampled *training* rows, scored on a fixed validation split cut from that scenario's training rows only (A random 10%; E held-out 2023 flight-days; C the latest 10% of 2022 by date). The test fold was never used for tuning or for choosing the blend members.
- Boosters were tuned at learning rate 0.1 (≤800 rounds) and refit at 0.05 (≤3,500 rounds, CatBoost 0.06 / ≤3,000) with early stopping on a random 10% of the training rows, as HGB does internally. Forests tuned with 40 trees, refit with 120. MLP: 10 epochs tuning, ≤25 final. Ridge: 6 values of alpha.
- **HGB was not re-tuned**: it uses the existing hand-tuned configuration (the incumbent already had several tuning rounds), so the comparison is slightly kind to the competitors' search budgets only in that HGB got none now. Median lookup has no parameters.
- Categorical features: HGB, LightGBM, XGBoost and CatBoost use their native categorical handling; forests use integer codes; Ridge/MLP use one-hot plus scaled numbers.

## Recommendation for the client

1. **Boosted trees are the right family.** Every gradient-boosting library (HGB, LightGBM, XGBoost) lands within about half a point of the others on same-period and recalibrated data. Moving off the current model for a few tenths of a point buys little; XGBoost or LightGBM are fine choices if the team prefers their tooling, the current model is perfectly defensible.
2. **If maximum accuracy is wanted**, the three-model blend is the best on recalibrated data (87.6%, +0.6 over now) at the cost of three models to maintain and roughly 3× the prediction time.
3. **Simple baselines are far worse** on same-period data (naive lookup 75%, linear 81%, neural net 87% vs 93%), which shows the learned model is adding real value. The neural net did not beat trees on this tabular data.
4. **For prices in a period the model has never seen** (scenario C, about 66–68%), no algorithm fixes it; even a linear model scores the same. The remedy is fresh data (the recalibration scenario shows 10% of new data lifts accuracy to ~87%), not a different algorithm.
5. Random Forest: see its section: not recommended.

## Caveats

- All of it is one snapshot of Indian domestic fares, one train/test split per scenario, and a small tuning budget (6 trials). Differences between boosting libraries of a few tenths of a point should not be over-read.
- Scenario C is limited by data (two scrape periods), not by algorithm, and its algorithm ranking is unstable (see CatBoost's 3.8 pt seed range).
- Tuning validation splits are random within the training period, so they under-represent the shift seen in C; tuned parameters may favour A-style fit.
- Timings come from a shared, loaded laptop and are only comparable with each other.
