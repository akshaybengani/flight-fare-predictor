# Price-over-time model on the US Airfare Time Machine

Plain-English report. Numbers come from `models/ts_metrics.json` (built by `src/ts_train.py`; features by `src/ts_features.py`). Accuracy = 100 − MAPE, the project's metric, on the price scale (USD).

## 1. The question

The India model scores 93% when train and test come from the same scrape day and 66% on an unseen period, because the data never shows the same flight priced on many days. This dataset does. So: **what can a model really learn when it sees the same flight priced on many days, and does it beat just repeating yesterday's price?**

Short answer: **barely.** Repeating the last seen price is already 93% accurate. The best model adds about 0.25 percentage points on the next-day question in the honest time split (0.3 to 0.8 points on the other splits), and does better on large price jumps (RMSE 16% lower). The one thing it clearly learns is which way a price goes *when it does move* (69% vs a 54% coin-flip baseline). Calendar, fuel and weather features add little to nothing.

## 2. Data

- `data/us-airfare-time-machine/flight_prices.parquet`: 38,584,829 quotes, 50 US routes, 17 airline labels, search dates 2026-03-12 to 2026-09-30 (203 days), departure dates 2026-03-19 to 2027-03-29, `days_before_departure` 7 to 180. Licence listed by Kaggle as "other": check before commercial use.
- **Entity "flight"** = airline + route + departure date + departure time: 988,708 flights. About 1% of (flight, search day) pairs have more than one quote (different fares or stops); I keep the cheapest. That leaves 38,461,556 flight-day rows.
- Partial search days (2 Apr, 3 Apr, 6 May, 11-26 Jun) are handled as low coverage, not zero: a `day_coverage` feature (quotes that day ÷ rolling median) is given to the model, and history features are built from "the last days this flight was actually seen" (`last_gap` says how stale that is).
- **Sample used for training/evaluation: 10% of flights (98,585 flights, 3,829,218 quotes), sampled by flight, not by quote**, so history features stay intact. History and route-market features were computed on the *full* 38.5M rows first. A larger-sample run (30%) was not done (see limitations).

## 3. Features (everything computable on the search date)

| family | columns | notes |
|---|---|---|
| basics | days before departure, search weekday, departure weekday, departure hour, departure month, airline, route, stops, duration, distance | distance from `routes.csv` |
| history (same flight, earlier search days only) | last observed price, gap since it, price 1/3/7 days earlier, min/mean/std over last 3 and 7 observations, running min/max, days since last change, number of changes, number of observations, plus ratios `log(last / x)` | "price k days earlier" = latest observation at or before search date − k |
| route-market (earlier days only) | route median over previous 7 search days, airline's relative price level and rank on the route over previous 7 days, ratio last price / route median, competitors (airlines, flights) on that route and departure date that day, day coverage | competitor counts are the market listing visible that day, not their prices |
| calendar | for both search date and departure date: US federal holiday flag, days to next / since last holiday, holiday-weekend flag, days to Easter, spring-break / summer / winter / Thanksgiving windows | `holidays` package (MIT). The school-break windows are hand-set national rules of thumb, **not a dataset** |
| fuel | US Gulf Coast jet fuel and Brent, level plus 7 and 30-day change at the search date, last quote carried over weekends | FRED CSV (EIA data), no key |
| weather | per airport (origin and destination) tmax, tmin, precipitation, snowfall, wind on the departure date, as a **climatology** (same calendar day ±3 days averaged over 2016-2025) | Open-Meteo archive (ERA5); known at booking time, so no leakage |
| weather ORACLE (experiment only) | actual weather on the departure date | **not known at booking time**; covers 67% of rows (departures up to 2026-10-28); never in the main model |

The target is log price, learned as a residual against an anchor known at search time (log last observed price of the same flight; else log route 7-day median; else a constant). Predictions are `exp(anchor + model)`. Loss: L1 for LightGBM; XGBoost and HistGB needed other losses (section 8).

**Sources and licences.** python-holidays https://github.com/vacanza/holidays (MIT). FRED https://fred.stlouisfed.org/graph/fredgraph.csv?id=DJFUELUSGULF and `id=DCOILBRENTEU` (EIA public data; credit the source; no key used). Open-Meteo https://archive-api.open-meteo.com/v1/archive: data CC BY 4.0, but **the free API is for non-commercial use only** (https://open-meteo.com/en/terms). This work is a research POC, so the free tier is fine; a product would need a paid plan. Cached joins: `data/external_us/` (gitignored); `SOURCES.json` there repeats this.

## 4. Splits

- **T (time, the honest one):** train on search dates ≤ 2026-08-15, validate 08-16 to 08-31 (early stopping and tuning only), test 2026-09-01 to 09-30. The test flights also appear in training on earlier days: this is genuine forecasting forward in time. 2.93M train / 587k test rows.
- **F (flight):** 20% of flights held out entirely (all their search days). Their history features are intact, but the model has never seen those flights.
- **R (route):** 5 whole routes held out: LAX-JFK, LAX-ORD, JFK-SFO, EWR-LAX, DFW-ORD. Their reverse directions were also removed from training (stricter than asked). `route_id` is dropped in R.

Naive baselines on the same rows: **(a)** last observed price carried forward; **(b)** median price of (route, days-before-departure) from the training rows. (a) only exists for flights with a previous observation (98.3% of the T test rows), so the main comparison uses those rows; "all rows" results are in the JSON. (b) cannot be computed honestly for unseen routes (R).

## 5. Results

### Headline table (accuracy %, rows with history; within 10%, MAE USD, RMSE USD in the JSON)

| split | LightGBM | XGBoost | HistGB | Random Forest | **naive (a) last price** | naive (b) route × days-out median |
|---|---|---|---|---|---|---|
| **T time** | **93.28** | 92.94 | 91.95 | 91.88 | **93.03** | 66.30 |
| F flight | 94.87 | **94.93** | 94.01 | 94.12 | 94.08 | 67.40 |
| R route | **94.50** | 94.19 | 93.51 | 93.60 | 94.18 | n/a |

T-split detail for the chosen model (LightGBM, picked on **validation** accuracy, not test): accuracy 93.28, within 10% 81.65, MAE $12.62, RMSE $38.84. Naive (a): 93.03, 81.64, $12.94, $46.15. Naive (b): 66.30, 22.6, $70.60, $118.92. On all rows including the 1.7% first-time flights, the model gets 92.97 (naive (a) cannot score those).

**Reading it.** The model only counts as learning something if it beats the baselines. It beats (b) by 27 points (that is the old "static" view of the problem: a booking-window curve alone cannot do this). It beats (a) by a small margin: +0.25 points in time, +0.8 on unseen flights, +0.3 on unseen routes. XGBoost, HistGB and Random Forest do not beat (a) in time. These are single runs of one seed; I did not measure seed-to-seed spread, so treat gaps under about 0.3 points as unproven. The gain over (a) is larger in RMSE (−16%) than in the average error, which means it helps mostly on the big jumps.

### 5b. Random Forest (added at the portfolio manager's request)

scikit-learn `RandomForestRegressor`, same features, splits and log-price-residual target as the boosters (residual clipped to ±0.25 and squared-error loss, as for HistGB; NaN features filled with −1). Settings: 60 trees, `max_samples` 0.3, `min_samples_leaf` 5, `max_features` 0.33, 2 cores. Tuned on the validation window only (2 configs; the second, 0.5 / leaf 3, was no better and slower).

| split | RF accuracy (rows with history) | naive (a) last price | best booster | within 10% / MAE / RMSE (RF) | train time | model size |
|---|---|---|---|---|---|---|
| T time | 91.88 | 93.03 | LightGBM 93.28 | 81.49 / $15.40 / $41.69 | 637 s | 968 MB |
| F flight | 94.12 | 94.08 | XGBoost 94.93 | 85.67 / $11.97 / $40.86 | 595 s | 921 MB |
| R route | 93.60 | 94.18 | LightGBM 94.50 | 83.71 / $17.39 / $51.03 | 552 s | 893 MB |

- **Versus the naive baselines:** worse than last-price-carried-forward in time (−1.15 points) and on unseen routes (−0.6), level on unseen flights (+0.04, within noise). Better than route × days-out median by 25 points in time. By days before departure (T): 7-14 92.3, 15-30 90.2, 31-60 91.5, 61-120 91.8, 121-180 91.7, each below naive (a).
- **Versus the boosters:** 1.4 points behind LightGBM in time, 0.8 behind the best booster on F, 0.9 on R; similar to HistGB. It is also 1,000× larger on disk than the LightGBM model (about 0.6 MB). Random forests cannot take small corrections off an anchor as precisely as boosting does, and the clipped squared-error loss pulls predictions toward the mean move rather than "no change".
- **Direction (T, moved quotes):** sign right 66.6% (majority 53.75%), change MAE $15.40 vs "no change" $12.94, i.e. worse than the no-change guess.
- **Ablation (T, accuracy on all rows):** basics 65.7, + history 91.8, all 91.6. Same story: history is the model; the other families do not help.
- **Feature importance.** Permutation (T test window, 150k rows, drop in accuracy points): history 64.9, basics 0.55, route-market 0.44, calendar 0.05, fuel 0.03, weather 0.02 (shuffling `last_price` alone costs 66.1). Impurity (MDI) importance, which spreads over many continuous columns and favours them: lr_mean7 (last price vs 7-obs mean) 0.058, days before departure 0.041, airline relative price 0.035, last price vs route median 0.031, duration 0.028. Both agree with the boosters: recent price history and the route's price level matter; calendar, fuel and weather are near zero.
- **Not verified:** one seed, 60 trees, 2 tuning configs; a larger forest or different loss might close part of the gap but is unlikely to beat boosting here.

### By days before departure (T split, accuracy %)

| bucket | model | naive (a) last price | naive (b) route median |
|---|---|---|---|
| 7-14 | 93.23 | 93.34 | 76.52 |
| 15-30 | 91.13 | 91.26 | 65.88 |
| 31-60 | 92.50 | 92.71 | 62.76 |
| 61-120 | 93.16 | 93.15 | 64.68 |
| 121-180 | 93.53 | 93.55 | 68.60 |

(The model rows are all rows; naive (a) rows are rows with history, so the two are not exactly the same rows. The per-bucket gap shows no advantage in any bucket.) Fares 15-30 days out are the hardest; 121-180 days out the easiest.

### By route (T split)

Hardest: BOS-DCA 82.7, DCA-BOS 82.9, MSP-ORD 87.1, ORD-MSP 87.1. Easiest: SFO-EWR 95.8, ORD-DCA 95.8, LAX-ATL 96.0, ATL-LAX 96.2. All 50 routes are in the JSON. Short, cheap, low-fare shuttles are hardest in percentage terms.

### "Will it go up or down?" (the direction question)

Direction of the next observed price versus the last observed price (T split test rows with a previous observation: 577,183). **64.7% of the time the price has not changed**, so direction only exists for the 35% that moved.

| method | correct sign, rows that changed |
|---|---|
| always guess the majority sign (up) | 53.75% |
| route × days-out median reversion | 57.42% |
| regression model (LightGBM) sign of predicted − last | 59.96% (XGBoost 70.15%, HistGB 66.87%) |
| **dedicated up/down classifier** (LightGBM binary, trained on changed rows only, AUC 0.77) | **69.18%** |

So given that a fare moves, the features do tell you which way (a lot of it is a bounce-back after an unusual move: the top features are the last price relative to its recent mean and 3-observation minimum). The regression models' signs are weaker because their predicted moves are tiny. The model does **not** tell you whether a price will move: 65% do not. Mean absolute error of the predicted change: model $12.62 vs the "no change" guess $12.94 (T split, LightGBM); the regression gains are small in dollars.

Versus a week earlier (flights with a price about 7 days earlier): the model gets the sign right 87% of the time, but a "keep the last move going" persistence rule gets 90%, so there is nothing learned beyond persistence on this horizon.

### Algorithms

| algo | tuning (validation window only, 3 configs each) | notes |
|---|---|---|
| LightGBM | num_leaves 127, lambda_l2 10, L1 loss | best; chosen by validation accuracy 94.03 |
| XGBoost | pseudo-Huber (slope 0.05), depth 8, lambda 10 | 4 rounds before early stopping in time, 800 (the cap) on F and R |
| HistGB | squared error on residual clipped to ±0.25, 63 leaves | cap of 800 rounds reached; weaker |
| Random Forest | max_features 0.33, min_samples_leaf 5 (2 configs tried on 500k rows) | see section 5b |

L1 on XGBoost/HistGB collapsed to "no change" (every leaf median is 0 because 65% of next prices are unchanged), so their losses were changed; that is an equal tuning budget but not an identical objective, and it probably understates those two.

## 6. Ablation ladder (T split, LightGBM, fixed 124 rounds, accuracy % on all test rows)

Run with a fixed number of rounds because with early stopping the first ladder (kept in the JSON as `..._early_stopping`) was noisy: some steps stopped after 1-15 rounds, so the step-by-step gains were not trustworthy.

| step (cumulative) | accuracy | change |
|---|---|---|
| basics only | 72.14 | |
| + history | 92.87 | **+20.7** |
| + route-market | 92.78 | −0.1 |
| + calendar / holidays | 92.93 | +0.16 |
| + fuel | 92.89 | −0.04 |
| + weather (climatology) = all | 92.97 | +0.08 |

Drop one family from the full model: without history 91.80 (−1.2; the route 7-day median then acts as the anchor), without route-market 92.94 (+0.0), without calendar 92.89 (−0.08), without fuel 92.75 (−0.22), without weather 92.89 (−0.08). Adding the actual-weather oracle gives 92.91 (no gain, even with the future weather).

Without history, the extra families do help: basics + calendar 76.9 and basics + weather 76.0 vs basics 71.7 (early-stopping ladder), because dates and places let the model partly guess the flight; basics + route-market 92.2 for the same reason (a route median is a level anchor). Fuel alone is slightly *worse* than basics (70.0 vs 71.7): it behaves like a time stamp and does not transfer forward. In 2026 this data has a large fuel move (between the March and September search dates, jet about $2.7 to $4.7/gal and Brent about $69 to $138), which is real but unique to this year.

**Honest conclusion on families:** history is the model. Everything else is within ±0.2 points, a size I cannot separate from run-to-run noise with one seed. Fuel has the largest effect when dropped (−0.22) but that is the most likely to be a time proxy rather than a price driver; weather and holidays add nothing measurable on top of a price history.

### Permutation importance (T test window, 250k rows, accuracy drop in points when shuffled)

Families: history 66.99, route-market 0.63, basics 0.56, calendar 0.012, fuel 0.010, weather 0.000. Features: last_price 66.97, route_med7 0.59, route_id 0.33, run_max 0.15, std3 0.15, distance 0.08. Shuffling `last_price` breaks the anchor too, hence the huge value: it shows the model works as "last price + small correction", not that nothing else matters.

## 7. What this means for the FareSight story

- Seeing the same flight on many days turns an unsolvable "what will this flight cost, never seen before" problem (66% in India across time) into a solved-looking one (93%), but almost all of it is **persistence**: tomorrow's price is yesterday's price 65% of the time.
- A model adds a small, real improvement over persistence and some skill on direction after a move. It cannot say whether a move is coming.
- Holiday, fuel and weather features are not worth the effort on top of a price history; they may matter more for first-time flights and long horizons (not demonstrated here).

## 8. Limitations and what is unverified

- US routes (50, mostly the busiest), not India; one year of 2026 data; no seat availability or fare class; scraped quotes, with partial-coverage days and no knowledge of what a booking would actually cost.
- Single seed, single 10% flight sample, one test month (September). No confidence intervals. A 30% sample scaling check was not run.
- XGBoost/HistGB losses differ from LightGBM's (section 5); HistGB used its own internal early stopping and hit the 800-round cap on F/R, so those two may be under-trained.
- School-break windows are rules of thumb. Holiday dates for 2027 come from the package.
- Open-Meteo actual weather covers departures only up to 2026-10-28 (67% of rows); the ORACLE column is for diagnosis only.
- The saved `models/fare_model_ts.joblib` (LightGBM, 124 rounds, trained on search dates ≤ 2026-08-15) needs the history features built by `src/ts_features.py`; it is a research artefact, not wired into the app.
- The R split removes reverse routes too; `route_id` is dropped there. Baseline (b) is undefined for unseen routes.

## 9. To port this to India

You need the same flight observed on many search days, which is the missing ingredient: a scheduled (daily) collection of fares per airline, route, departure date and time, with the capture timestamp, for at least a few months and ideally a full year, plus ideally seat/fare class. Then reuse `src/ts_features.py` almost unchanged: swap the holiday calendar for the existing India calendar and festival windows in `src/enrich.py`, ATF price (Indian) instead of the US Gulf Coast series, and a few India airports in the weather fetch. Expect the same shape of result: persistence does most of the work, the model adds a little, and direction after a move is the one learnable signal. The existing two single-day India scrapes cannot produce this.

## 10. Files

`src/ts_features.py`, `src/ts_train.py`, `models/ts_metrics.json`, `models/fare_model_ts.joblib` (demo model, see 'Horizon accuracy for the demo'), `models/fare_model_ts_nextday_v1.joblib`, `models/ts_horizon_metrics.json`, `src/us_horizon.py`, `src/us_service.py`, `models/ts_work/*.json` (checkpoints), `data/external_us/` (cached joins, DuckDB file, `features.parquet`, logs; gitignored).

## 11. Horizon accuracy for the demo

The demo app serves US listings with a "predicted price" some days ahead and a price curve for the next 14 days. The next-day model above cannot do that, so `src/us_horizon.py` trains **one LightGBM model per horizon (1, 2, 3, 5, 7, 10, 14 days ahead)**; other horizons are interpolated between them (in log price ratio). It is a direct model, not a recursive roll-forward: each model sees only information known at the end of the as-of day (the flight's own quotes up to that day, route-market over the 7 days to that day, fuel at that day) plus things known in advance about the target day (weekday, holidays, days before departure) and climatological weather. Target: log(price on day s+h / price on day s); L1 loss; the number of rounds is chosen on the validation window (target dates 16-31 Aug). Test = target dates 1-30 September 2026, 10% flight sample (about 450k to 565k rows per horizon), the flight must be quoted on both days. Fuel and route-market features are held at their as-of values for every horizon.

| days ahead | model | last price carried forward | route × days-out median | gain vs last price | within 10% (model / last price) | verdict |
|---|---|---|---|---|---|---|
| 1 | 93.19 | 93.13 | 66.30 | +0.06 | 82.0 / 82.0 | no better |
| 2 | 89.23 | 89.15 | 66.28 | +0.08 | 72.1 / 72.1 | no better |
| 3 | 86.51 | 86.42 | 66.25 | +0.09 | 64.9 / 64.8 | no better |
| 5 | 84.93 | 84.65 | 66.17 | +0.28 | 60.5 / 59.9 | better (small) |
| 7 | 84.80 | 83.94 | 66.10 | +0.86 | 52.3 / 57.8 | better on average error, worse on "within 10%" |
| 10 | 82.85 | 79.86 | 66.09 | +2.98 | 47.7 / 49.9 | better on average error, worse on "within 10%" |
| 14 | 81.66 | 77.97 | 66.19 | +3.69 | 43.9 / 45.0 | better on average error, worse on "within 10%" |

Accuracy = 100 − MAPE. Verdict rule: at least +0.1 point is "better", within ±0.1 is "no better". **Reading it:** up to 3 days ahead the model is no better than "the price stays the same" (at 1 to 3 days the early-stopped models are essentially the last price: they stop after 1 round). From about a week ahead it does beat carrying the price forward on average error (+0.9 to +3.7 points), because at that range prices really move (only 12.5% of prices are unchanged 14 days later, vs 65% a day later) and the model captures the typical rise toward departure and some bounce-back. But it loses on "within ±10%" at 7+ days: it cuts the big misses and gives up some of the small ones. Single seed; differences under about 0.3 points are not proven. These numbers are for the evaluation (train to 15 Aug) models; the served models are refit on all target dates up to 30 Sep with the same settings.

**Do "may drop" flags mean anything?** Among September test rows where the model predicted a fall of at least 10%: at 7 days ahead 49.7% really fell (base rate 30.2%), 38.0% fell by 10% or more (base 18.5%); at 14 days ahead 62.3% fell (base 34.6%) and 49.9% fell by 10% or more (base 22.3%), with a mean actual change of −11.7%. So at the long horizon a flag carries real information, roughly doubling the chance of a 10% fall, but it is wrong half the time. The flag in the demo uses the lowest predicted price at or after the chosen horizon within the 14-day curve, so it is driven by the 7 to 14-day models, not by the 1-day one. In the 2026-09-30 snapshot about 5.6% of flights are flagged at the default 10% threshold.

Limitations specific to the demo: the snapshot is as of 2026-09-30 (the data ends there), the curve cannot extend below 7 days before departure (the data never shows closer quotes), the model files were trained on 10% of flights, and the same US caveats as above apply.
