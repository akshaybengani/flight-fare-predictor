# Switchable prediction engines

Eight algorithms behind one interface so the app can switch between them. **Default engine: XGBoost.**
Code: `src/engines.py` (loader), `src/build_engines.py` (training / retraining), `src/engines_selftest.py` (checks).
Files: `models/engines/<key>.joblib` + `models/engines/manifest.json`.

## Interface

```python
from engines import list_engines, load_engine
eng = load_engine("xgboost")          # cached; at most 2 engines stay in memory; thread-safe; lazy
log_price = eng.predict(feats)        # feats = ENC.transform(add_basic(X)): exactly the frame the app builds
price = np.exp(log_price)             # numpy array of predicted LOG price, so np.exp(MODEL.predict(feats)) -> np.exp(eng.predict(feats))
list_engines()                        # manifest entries + "available" / "unavailable_reason" for this Python
```

- Every engine uses the **same encoder**, the deployed `ENC` in `models/fare_model.joblib` (training asserts a freshly fitted copy is identical). Each engine does its own frame conversion inside (native categoricals for XGBoost/CatBoost, NaN fill for forests, one-hot + scaling for Ridge/MLP).
- Errors: `engines.EngineUnavailable` with a plain message if the file is missing, a library is not installed, the pickle cannot be read (version mismatch), or the key is unknown.
- The `hgb` engine wraps the deployed `models/fare_model.joblib` as is (not retrained, no separate file).
- Manifest per engine: key, label, family, description, file, size_mb, load_seconds, predict_ms_per_1k, memory_mb_peak, accuracy / within10 / mae for scenarios A, E, C (from the benchmark, `models/algorithms.json`), `requires`, versions built with, params, sanity checks, notes, `default` (true only for xgboost), `scores` (full accuracy / within10 / mae / rmse / r2 per scenario), `demo_safe`, `december_check`.

## Sizes, speed, memory (this MacBook, 4 threads; predict time excludes feature encoding)

| Engine | File MB | Load s | Predict ms / 1k rows | Peak process MB | Benchmark accuracy A / E / C | Needs |
|---|---|---|---|---|---|---|
| **XGBoost (default)** | 135.6 | 0.15 | 30.4 | 935 | 93.42 / 86.88 / 67.74 | xgboost 3.4.1 |
| HistGradientBoosting (current model) | 109.2 | 0.22 | 21.6 | 317 | 93.04 / 87.00 / 66.17 | sklearn only |
| LightGBM | 163.5 | 0.32 | 105.3 | 640 | 93.19 / 87.22 / 60.87 | lightgbm 4.7.0 |
| CatBoost | 50.6 | 0.05 | 2.8 | 190 | 91.47 / 86.25 / 67.39 | catboost 1.2.10 |
| Random Forest (capped) | 113.2 | 0.09 | 1.4 | 407 | 91.47 (capped) / see below | sklearn only |
| Extra Trees (capped) | 111.9 | 0.11 | 1.4 | 427 | 90.86 (capped) / see below | sklearn only |
| Ridge (linear) | 0.005 | 0.01 | 0.3 | 198 | 81.46 / 72.43 / 66.27 | sklearn only |
| Neural net (MLP) | 0.4 | 0.01 | 0.6 | 201 | 86.80 / 80.46 / 47.29 | sklearn only |

Benchmark scores come from `models/algorithms.json` (full-size models trained on each scenario's train split), not from these deployed files, which are trained on all rows. Peak memory = max resident size of a fresh process that only loads the engine and predicts a small frame.

## Which engines load without extra libraries?

The server today has numpy, pandas, pyarrow, scikit-learn 1.9.1 and joblib. Verified earlier by running the self-test in the project `.venv` before lightgbm and catboost were installed:

- **Load with no extra libraries (safe default set):** `hgb`, `random_forest`, `extra_trees`, `ridge`, `mlp`.
- **Need an extra package:** `xgboost` (3.4.1), `lightgbm` (4.7.0), `catboost` (1.2.10). All three are installed in the project `.venv` now, so all eight engines load there. Where one is missing they report `available: false` with the reason, and `load_engine` raises `EngineUnavailable`.
- **Dates outside the training months.** The data only covers **January to April** (2022 and 2023), and Ridge/MLP use month and weekday as plain numbers, so a December search used to be extrapolated (MLP ~₹40k for Delhi to Mumbai). The Ridge and MLP engine wrappers now clip every plain-number input (month, weekday, days_left, period, hours, distance, price-level encodings, ...) to the range seen in training before predicting, so a December search behaves like the nearest trained month (April), as the tree engines effectively do. The clip ranges are stored inside the engine files. This does not change the benchmark scores (held-out rows are inside the trained months): on the 20,000-row agreement check the numbers are identical before and after (Ridge 78.8% / log difference 0.0902; MLP 85.5% / 0.0772).
- **December sanity search** (Economy, 2026-12-10, 21 days ahead; mean fare in ₹; Delhi to Mumbai has 207 flights, Ahmedabad to Mumbai 31):

| Engine | Delhi to Mumbai | vs median engine | Ahmedabad to Mumbai | vs median engine |
|---|---|---|---|---|
| HGB | 7,753 | -0.2% | 5,941 | +0.7% |
| XGBoost | 7,648 | -1.6% | 5,614 | -4.8% |
| LightGBM | 7,830 | +0.8% | 5,859 | -0.7% |
| CatBoost | 8,059 | +3.7% | 6,869 | +16.4% |
| Random Forest (capped) | 7,789 | +0.2% | 5,521 | -6.4% |
| Extra Trees (capped) | 8,044 | +3.5% | 6,435 | +9.1% |
| Ridge | 7,021 | -9.7% | 6,263 | +6.2% |
| MLP | 6,451 | -17.0% | 5,744 | -2.6% |

  Median engine: ₹7,771 and ₹5,900. Every engine is within 25% of it on both routes, so every manifest entry has `"demo_safe": true`. The MLP is the furthest out (-17% on Delhi to Mumbai); both linear engines are still blunt instruments (benchmark accuracy 81% / 87%) and are not recommended as the demo engine. The rule is applied automatically by `python src/build_engines.py --december-check`: any engine more than 25% from the median engine gets `"demo_safe": false` and a CAUTION note in the manifest.

### Library versions to LOAD

Built with Python 3.12.3, numpy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1, joblib 1.6.0, xgboost 3.4.1, lightgbm 4.7.0, catboost 1.2.10. Loaded and tested in the project `.venv` (numpy 2.5.4, pandas 3.1.0, pyarrow 26.0.1, scikit-learn 1.9.1, joblib 1.6.0, xgboost 3.4.1, lightgbm 4.7.0, catboost 1.2.10). Server `requirements`: `scikit-learn==1.9.1 xgboost==3.4.1 lightgbm==4.7.0 catboost==1.2.10` plus numpy, pandas, pyarrow, joblib (any recent versions; the ones above are what was tested). **scikit-learn must match 1.9.1** (pickles are tied to it); the boosting libraries should match the versions above or later patch releases of the same major version.

### Server install notes

- **Linux:** `pip install xgboost==3.4.1` (a wheel; no compiler needed). The wheels need the OpenMP runtime `libgomp1`: `apt-get install -y libgomp1` on Debian/Ubuntu images (most images already have it). LightGBM and (optionally) CatBoost use the same.
- **macOS:** the xgboost and lightgbm wheels link `@rpath/libomp.dylib`. Easiest: `brew install libomp`. Without Homebrew, `engines.py` makes a copy of the libomp that scikit-learn ships (`sklearn/.dylibs`) with its install name rewritten to `@rpath/libomp.dylib` (needs Xcode command-line tools: `install_name_tool`, `codesign`), caches it in `~/.cache/flight-poc/libomp_shim/` and loads it before xgboost is imported, inside the running process (no environment variables or re-exec needed, works under Flask/gunicorn). A plain preload of scikit-learn's copy does NOT work because its install name is different. Tested by stripping Homebrew's libomp path from a copy of the xgboost library.

## Retrain

```bash
python src/build_engines.py                       # all engines not yet built (about 40 min on 4 cores; run in the background)
python src/build_engines.py --only xgboost --force # one engine
python src/build_engines.py --assemble-only       # rebuild manifest.json from models/engines/_meta
python src/engines_selftest.py                    # load every available engine, predict, check
```

Training protocol: all rows of `data/merged.parquet` (class != First), the encoder identical to the deployed one, the best scenario-A parameters from `models/algorithms.json`, boosters early-stopping on a random 10% of the rows (cap 3,500 rounds; the XGBoost model hit the cap). Needs xgboost, lightgbm, catboost installed. The benchmark code in `src/compare_models.py` supplies the fit functions. After changing the deployed `models/fare_model.joblib`/encoder, rebuild every engine.

## Checks recorded in the manifest

- **Benchmark agreement:** each engine's predictions on 20,000 rows are compared with the benchmark's saved predictions for the same rows (mean |log difference| <= 0.08 and correlation >= 0.99). All engines pass except Ridge (0.090: a linear model changes more when retrained on 2x the rows); XGBoost: 0.039 / 0.997. Caveat: the 20,000 rows are held-in for the deployed models, so their accuracy there (XGBoost 95.7%) is optimistic by design.
- **Sample search:** each engine prices Delhi to Mumbai, Economy, 2026-12-10, 21 days ahead (207 flights) with no NaN or inf (see the Ridge/MLP warning above).
- **Self-test:** load, cache identity, shapes, finiteness, plausible fares, business above economy on four routes, bounded cache, helpful unknown-engine error: all eight engines pass in the project `.venv`.

## Explaining a price: `explain()` and `reference_values()`

```python
from engines import explain, reference_values
out = explain("xgboost", feats_one_row)     # feats_one_row = one-row DataFrame, same format as predict(); other engines work too
rv  = reference_values()                    # per-feature facts for building editable input fields
```

`explain(engine_key, feats)` returns `{"method": "shap" | "sensitivity", "base_price", "price", "steps": [{"feature", "label", "value", "effect_inr"[, "value_display"]}, ...]}`. Steps are ordered by absolute effect, at most 12 plus one `"other"` step (`"All N other features"`), and `base_price + sum(effect_inr) == price` to within 0.01 rupee (rounding crumbs go to the smallest step). `value` is the category name for categorical features. `price` is the engine's own prediction, `exp(engine.predict(feats))`.

- **XGBoost → `"shap"`**: exact tree contributions (`Booster.predict(pred_contribs=True)`) in log-price space, turned into rupees by a multiplicative walk from `exp(bias)` in order of the size of each contribution. Each effect is "what this feature added on top of the features before it", so effects are sequential, not independent. `base_price` is `exp(bias)` (the model's baseline price, a geometric mean of training fares, about ₹13.9k); the arithmetic mean over the reference sample is also returned as `reference_mean_price` (₹22.0k, pulled up by Business fares). About 35 ms per row (the trees are scored in 4 parallel slices; the first call takes about 0.5 s while it builds them and uses about 135 MB extra memory).
- **All other engines → `"sensitivity"`** (a what-if, not an exact attribution; it ignores interactions): each feature in turn is replaced by the median (numbers) or mode (categories) of a fixed 2,000-row reference sample of training rows (`models/engines/reference.parquet`), the row is re-predicted through the engine's own frame conversion, and the price difference is that feature's effect. The raw effects rarely add up to `price - base_price` (`base_price` = the engine's mean predicted price over the reference sample), so the gap is spread over the features in proportion to the size of their effect. Between 4 and 150 ms per row (HGB slowest).
- Features that are **derived** from the training data (`te_flight`, `te_route_airline`: the usual price level of this flight / airline-route, stored as the natural log of a price; `route_supply`: number of flights on the route) are explained like any other. In the demo, remember that changing the airline or flight does not by itself change `te_flight`; the app recomputes them with the encoder when it builds the frame, so call `explain` on the rebuilt frame. For the same reason `te_flight` is usually the biggest step: it is "this flight's usual fare level".
- Example, Delhi to Mumbai, Economy, 2026-12-10 (21 days ahead), one AirAsia flight, XGBoost: base ₹13,904, price ₹5,024. Steps: usual price of this flight (about ₹6,673) −₹7,178; usual price of AirAsia on this route (about ₹4,950) −₹1,386; month (December, treated as April) −₹652; arrival hour 19:35 +₹425; airline AirAsia −₹365; data period 2023 +₹289; departure hour 04:55 +₹229; journey time 14.7 h +₹208; weekday Thursday −₹146; destination Mumbai −₹120; cabin Economy −₹49; departure bucket Before 6 AM −₹36; 6 other features −₹98.

`reference_values()` returns `{"order_note", "target", "reference_sample", "features": [...]}` with the 18 features **in the order the model receives them** (`ENC.features`). Each has `name`, `order`, `label` (plain English), `description` (one sentence; read from `FEATURE_DOCS` in `src/app.py` at build time, read-only), `type` (`"number"` or `"category"`), `derived`, and for categories `categories` (names; the frame holds the position in this list as the code) and `typical` (most common); for numbers `min`, `max`, `typical` (median) and `integer`. The derived features are `te_flight`, `te_route_airline` and `route_supply` (typical price levels and route supply computed from the training data, not something a traveller types); the `te_*` ones also carry `typical_price_inr`, `min_price_inr`, `max_price_inr`. Built with `python src/build_engines.py --build-reference`; the self-test checks the order against `ENC.features`.

The self-test also checks, for every available engine and 20 random rows: steps add up to the price (1 rupee tolerance, actual error below 1e-10), no NaN, ordering, speed (XGBoost under 100 ms, others under 500 ms), and that booking 30 days ahead is not more expensive than 3 days ahead (about 0.8x for every engine).

## Showing the calculation tree by tree: `trace()` (XGBoost only)

```python
from engines import trace
out = trace("xgboost", feats_one_row, first_n=5)     # same one-row frame as explain(); about 30 ms
trace("hgb", feats_one_row)                          # -> {"available": False, "reason": "...only available for the XGBoost engine..."}
```

**Formula:** predicted price = `exp(start_log + sum of every tree's leaf value)`. The model starts from its base score (`start.log` 9.54, `start.price_inr` about ₹13,905) and each of the 3,500 trees adds a leaf value to that log price (the learning rate 0.05 is already inside the leaf values). HistGradientBoosting is not supported (its trees are in `model._predictors` with a different node layout; not done).

Returned fields: `available`, `engine`, `n_trees`, `learning_rate`, `price`, `start {log, price_inr, formula, note}`; `cumulative` (price after 1, 2, 3, 5, 10, 25, 50, 100, 250, 500, 1000, 2000 and all trees; computed with `iteration_range=(0, k)`; the last value must equal the engine's normal prediction and the call asserts that to within ₹1); `trees`; `questions_used`; `summary_sentence`.

- **`trees`**: the first `first_n` trees (role `"first"`), the 3 trees whose leaf for this row is largest in absolute value (`"biggest #1..3"`), and the 3 biggest after the first 5 (`"biggest after the first 5 #1..3"`). Boosting front-loads the large corrections, so the overall biggest three are usually among the first five; the "after" ones add variety. Per tree: `tree_number` (1-based), `roles`, `steps`, `leaf_value` (log), `price_before`, `price_after`, `effect_inr` (what adding this tree does to the running price when it is added in its turn). Each step: `feature`, plain `label`, `kind` (`number`/`category`), `question` in words, `threshold` + `threshold_display` or the category set (`categories`, names), the row's own value (`row_value`, `row_value_display` for the price-level features), `answer` (`yes`/`no`/`missing`) and `next_node`. XGBoost's convention: the **yes** branch is taken when value < threshold (compared as float32, as xgboost does); for a category split, **yes** when the category is in the listed set; a missing value follows the tree's default branch.
- **`questions_used`**: over all 3,500 trees, per feature `times_asked`, `share_of_questions_pct` (how often it is used as a question) and `gain_share_pct` (xgboost `total_gain`, normalised to 100%), in plain labels, plus `total_questions` (1,785,000 here). It is computed once from `models/engines/xgboost_trace.joblib` (29 MB, git-ignored like the engines; rebuild with `python src/build_engines.py --build-trace-cache`, about 15 s and 1 GB of memory). If the file is missing, the first call computes the same thing in memory (about 15 s).
- Speed: about 30 ms per row after the first call.
- Verified by the self-test for 20 random rows: the cumulative end equals the prediction, every printed question re-evaluates by hand to the printed answer, each printed leaf equals what that single tree adds to the log price, no NaN, shares sum to 100%, under 300 ms.

Example, Delhi to Mumbai, Economy, 2026-12-10 (21 days ahead), the same AirAsia flight as the `explain()` example:

> The model starts at ₹13,905. 3,500 small decision trees each add or subtract a little. After 100 trees the price is ₹5,787, after 1,000 it is ₹5,093, and after all 3,500 it settles at ₹5,024.

Cumulative: 1 tree ₹13,465; 3 trees ₹12,624; 10 trees ₹10,423; 25 trees ₹8,021; 50 trees ₹6,483; 100 trees ₹5,787; 500 trees ₹5,227; 1,000 trees ₹5,093; 2,000 trees ₹4,998; 3,500 trees ₹5,024.

Tree 1 path (leaf value −0.0322 in log price, price ₹13,905 → ₹13,465, effect −₹440): usual price of this flight below 9.88 (about ₹19,529)? row 8.81 (about ₹6,673): yes; month of travel below 3? row 12: no; departure hour below 23.17? row 4.92: yes; usual price of this flight below 9.06 (about ₹8,624)? yes; journey time below 2.17 h? row 14.67: no; distance below 1,254 km? row 1,147: yes; usual price of this flight below 8.70 (about ₹5,974)? no; is "From" one of [Bangalore, Chennai]? row Delhi: no; usual price of this flight below 8.96 (about ₹7,782)? yes; is "To" one of [Bangalore, Chennai, Delhi, Hyderabad, Mumbai]? row Mumbai: yes; distance below 879 km? row 1,147: no; leaf.

Most used question: "usual price level of this flight" is 14.5% of all questions and 77.5% of the total gain; then month of travel (0.7% of questions, 5.8% of gain), usual price of this airline on this route (8.8%, 5.2%), journey time (10.7%, 4.0%), days before departure (20.5%, 1.8%).
