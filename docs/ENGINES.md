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
