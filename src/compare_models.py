"""Benchmark several regression algorithms on EXACTLY the data, features, splits, target (log price) and
metric of src/train.py (scenarios A / E / C). Writes only new files; never touches metrics.json etc.

    python src/compare_models.py                       # everything, all scenarios (long: run in background)
    python src/compare_models.py --only lightgbm,xgboost --scenarios A,C
    python src/compare_models.py --assemble-only       # rebuild models/algorithms.json from checkpoints
    python src/compare_models.py --noise               # re-fit the top 3 with seeds 1,2 (no re-tuning) -> models/algorithms_noise.json
    python src/compare_models.py --feature-set enriched --top3   # second pass on the enriched features
    python src/compare_models.py --smoke               # tiny, fast sanity run into a scratch dir

Protocol
  * Splits / encoder / NUMS / TE come from train.py + features.py (leak-safe out-of-fold target encoding fitted on train rows only).
  * Tuning: per scenario, a fixed validation split carved ONLY from that scenario's training rows
    (A random 10%, E held-out 2023 flight-days, C latest 10% of 2022 by journey date), on <=150k sampled training rows,
    N_TRIALS random-search trials per algorithm (same RNG seed, same budget). The test fold is never used for tuning or selection.
  * Final fit: full scenario training set with the chosen params. Boosters early-stop on a random 10% of the training rows
    (what HistGradientBoosting does internally), so every booster sees the same data budget.
  * Thread budget: half of the CPU cores.
Checkpoints: models/algo_runs/<feature_set>/<algo>_<scenario>[_seedN].json (+ preds/*.npy used by the blend).
"""
import os
import sys


def _bootstrap():
    """Cap threads to half the cores; on macOS borrow the libomp that scikit-learn ships so lightgbm/xgboost can load."""
    if os.environ.get("_CMP_BOOT"):
        return
    env = dict(os.environ, _CMP_BOOT="1")
    nthr = str(max(1, (os.cpu_count() or 2) // 2))
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        env[k] = nthr
    if sys.platform == "darwin":
        import importlib.util
        spec = importlib.util.find_spec("sklearn")
        if spec and spec.submodule_search_locations:
            d = os.path.join(list(spec.submodule_search_locations)[0], ".dylibs")
            if os.path.isdir(d):
                env["DYLD_FALLBACK_LIBRARY_PATH"] = d + ":" + env.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


if __name__ == "__main__":
    _bootstrap()

import argparse
import json
import tempfile
import time
import warnings
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CATS, Encoder, add_basic
from train import GBM, NUMS, TE

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
NT = max(1, (os.cpu_count() or 2) // 2)
N_TRIALS = 6
TUNE_N = 150_000
SCEN = {"A": "A_same_period_random_20pct", "E": "E_2022_plus_10pct_of_2023", "C": "C_train_2022_test_2023"}
SCEN_LABEL = {"A": "Same period: random 20% of 2023 rows held out (other days of the same flights are in training)",
              "E": "Recalibrated: train on 2022 + 10% of 2023 flight-days, test on the other 2023 flight-days",
              "C": "Unseen period: train on 2022 only, test on all of 2023 (no period/exact-time features, route-airline encoding only)"}
ORDER = ["hgb", "lightgbm", "xgboost", "catboost", "random_forest", "extra_trees", "ridge", "median_lookup", "mlp", "blend"]
NAMES = {"hgb": "HistGradientBoosting (current model)", "lightgbm": "LightGBM", "xgboost": "XGBoost", "catboost": "CatBoost",
         "random_forest": "Random Forest", "extra_trees": "Extra Trees", "ridge": "Ridge regression (linear)",
         "median_lookup": "Route-airline-class median lookup (naive)", "mlp": "Neural net (MLP)", "blend": "Blend (average of top 3)"}
FAMILY = {"hgb": "boosting", "lightgbm": "boosting", "xgboost": "boosting", "catboost": "boosting", "random_forest": "bagging",
          "extra_trees": "bagging", "ridge": "linear", "median_lookup": "naive", "mlp": "neural", "blend": "ensemble"}


class D:
    """encoded features X, raw frame df (for the naive lookup), log-price target, price"""

    def __init__(self, X, df):
        self.X, self.df, self.price = X, df, df.price.values.astype(float)
        self.y = np.log(self.price)


# ----------------------------------------------------------------------------------------------- data / scenarios
def load_df(feature_set, smoke):
    if feature_set == "enriched":
        df = add_basic(pd.read_parquet(ROOT / "data/enriched.parquet"))
    else:
        df = add_basic(pd.read_parquet(ROOT / "data/merged.parquet"))
    df = df[df["class"] != "First"].reset_index(drop=True)
    if smoke:
        df = df.sample(frac=0.04, random_state=0).reset_index(drop=True)
    return df


def build_scenario(sc, df, extra):
    """Same construction as train.py. Returns train, test, nums, te, (tune_train, tune_val)."""
    d22, d23 = df[df.period == 2022], df[df.period == 2023]
    if sc == "A":
        te_idx = d23.sample(frac=0.2, random_state=0).index
        tr, te_df, nums, te = d23.drop(te_idx), d23.loc[te_idx], NUMS, TE
        rng = np.random.RandomState(1)
        v = rng.rand(len(tr)) < 0.10
        tt, tv = tr[~v], tr[v]
    elif sc == "E":
        rng = np.random.RandomState(0)
        keys = d23.flight_code + d23.journey_date.astype(str)
        pick = set(rng.choice(keys.unique(), size=int(keys.nunique() * 0.1), replace=False))
        mk = keys.isin(pick)
        tr, te_df, nums, te = pd.concat([d22, d23[mk]]), d23[~mk], NUMS, TE
        k23 = np.array(sorted(pick))
        hold = set(np.random.RandomState(1).choice(k23, size=max(1, int(len(k23) * 0.3)), replace=False))
        in_hold = (tr.flight_code + tr.journey_date.astype(str)).isin(hold) & (tr.period == 2023)
        tt, tv = tr[~in_hold], tr[in_hold]
    else:
        tr, te_df = d22, d23
        nums, te = [n for n in NUMS if n not in ("period", "dep_hour", "arr_hour")], ["te_route_airline"]
        cut = tr.journey_date.quantile(0.9)
        tt, tv = tr[tr.journey_date < cut], tr[tr.journey_date >= cut]
    if len(tt) > TUNE_N:
        tt = tt.sample(TUNE_N, random_state=0)
    return tr, te_df, nums + extra, te, (tt, tv)


def enriched_extra(df, tr, use_enriched):
    if not use_enriched:
        return []
    from enrich import groups, nonconstant
    g = groups(df.columns)
    new = [c for k in ("calendar", "weather_lag", "fuel_fx", "airport_static") for c in g[k]]
    return nonconstant(tr, new)


# ----------------------------------------------------------------------------------------------- metrics
def metrics(pred, y):
    err = pred - y
    ape = np.abs(err) / y
    return {"accuracy": round(float(100 - ape.mean() * 100), 3), "within10": round(float((ape <= 0.10).mean() * 100), 2),
            "mae": round(float(np.abs(err).mean()), 1), "rmse": round(float(np.sqrt((err ** 2).mean())), 1),
            "r2": round(float(1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()), 4)}


# ----------------------------------------------------------------------------------------------- model fitters
# each fitter: (tr: D, params, seed, tuning: bool, cx) -> (predict_log(D) -> ndarray, picklable object, extra info dict)
def split_es(tr, seed=0):
    i_a, i_v = train_test_split(np.arange(len(tr.y)), test_size=0.1, random_state=0)
    return tr.X.iloc[i_a], tr.X.iloc[i_v], tr.y[i_a], tr.y[i_v]


def fit_hgb(tr, p, seed, tuning, cx):
    g = dict(GBM, random_state=seed)
    if tuning:
        g.update(learning_rate=0.1, max_iter=800)
    m = HistGradientBoostingRegressor(categorical_features=cx["cat_idx"], **g).fit(tr.X, tr.y)
    return (lambda d: m.predict(d.X)), m, {"n_rounds": int(m.n_iter_)}


def fit_lgbm(tr, p, seed, tuning, cx):
    import lightgbm as lgb
    Xa, Xv, ya, yv = split_es(tr)
    m = lgb.LGBMRegressor(n_estimators=800 if tuning else 3500, learning_rate=0.1 if tuning else 0.05, n_jobs=NT,
                          random_state=seed, verbose=-1, **p)
    m.fit(Xa, ya, eval_set=[(Xv, yv)], categorical_feature=CATS, callbacks=[lgb.early_stopping(40, verbose=False)])
    return (lambda d: m.predict(d.X)), m, {"n_rounds": int(m.best_iteration_)}


def xgb_frame(X, enc):
    X = X.copy()
    for c in CATS:
        X[c] = pd.Categorical.from_codes(X[c].values.astype(int), categories=list(range(len(enc.cats[c]))))
    return X


def fit_xgb(tr, p, seed, tuning, cx):
    import xgboost as xgb
    Xa, Xv, ya, yv = split_es(tr)
    enc = cx["enc"]
    m = xgb.XGBRegressor(tree_method="hist", grow_policy="lossguide", max_depth=0, enable_categorical=True,
                         n_estimators=800 if tuning else 3500, learning_rate=0.1 if tuning else 0.05,
                         early_stopping_rounds=40, n_jobs=NT, random_state=seed, **p)
    m.fit(xgb_frame(Xa, enc), ya, eval_set=[(xgb_frame(Xv, enc), yv)], verbose=False)
    return (lambda d: m.predict(xgb_frame(d.X, enc))), m, {"n_rounds": int(m.best_iteration) + 1}


def cat_frame(X):
    X = X.copy()
    for c in CATS:
        X[c] = X[c].astype(int)
    return X


def fit_cat(tr, p, seed, tuning, cx):
    from catboost import CatBoostRegressor
    Xa, Xv, ya, yv = split_es(tr)
    m = CatBoostRegressor(iterations=500 if tuning else 3000, learning_rate=0.12 if tuning else 0.06, thread_count=NT,
                          random_seed=seed, verbose=0, early_stopping_rounds=50, cat_features=CATS, **p)
    m.fit(cat_frame(Xa), ya, eval_set=(cat_frame(Xv), yv))
    return (lambda d: m.predict(cat_frame(d.X))), m, {"n_rounds": int(m.get_best_iteration()) + 1}


def fit_forest(cls):
    def f(tr, p, seed, tuning, cx):
        m = cls(n_estimators=40 if tuning else 120, n_jobs=NT, random_state=seed, **p).fit(tr.X.fillna(-1), tr.y)
        return (lambda d: m.predict(d.X.fillna(-1))), m, {"n_trees": m.n_estimators}
    return f


def linear_pre(cx):
    cat_cols = list(CATS)
    num_cols = [c for c in cx["features"] if c not in CATS]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num_cols)])


def fit_ridge(tr, p, seed, tuning, cx):
    m = Pipeline([("pre", linear_pre(cx)), ("m", Ridge(**p))]).fit(tr.X, tr.y)
    return (lambda d: m.predict(d.X)), m, {}


def fit_mlp(tr, p, seed, tuning, cx):
    p = dict(p)
    p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
    m = Pipeline([("pre", linear_pre(cx)), ("m", MLPRegressor(
        batch_size=2048, early_stopping=True, n_iter_no_change=4, validation_fraction=0.1,
        max_iter=10 if tuning else 25, random_state=seed, **p))]).fit(tr.X, tr.y)
    return (lambda d: m.predict(d.X)), m, {"n_epochs": int(m[-1].n_iter_)}


class MedianLookup:
    KEYS = [["source", "destination", "airline", "class"], ["source", "destination", "class"], ["class"]]

    def fit(self, df):
        y = np.log(df.price)
        self.tabs = [y.groupby([df[c] for c in k]).median() for k in self.KEYS]
        return self

    def predict(self, df):
        out = np.full(len(df), np.nan)
        for k, t in zip(self.KEYS, self.tabs):
            idx = pd.MultiIndex.from_frame(df[k]) if len(k) > 1 else pd.Index(df[k[0]])
            v = t.reindex(idx).values
            out = np.where(np.isnan(out), v, out)
        return np.where(np.isnan(out), np.nanmedian(self.tabs[-1].values), out)


def fit_median(tr, p, seed, tuning, cx):
    m = MedianLookup().fit(tr.df)
    return (lambda d: m.predict(d.df)), m, {}


# search spaces: dict of grids, N_TRIALS random draws with a fixed RNG seed (identical budget for every tunable algorithm)
SPACES = {
    "lightgbm": dict(num_leaves=[63, 127, 255, 511], min_child_samples=[5, 10, 20, 50], reg_lambda=[0.0, 1.0, 5.0, 10.0],
                     colsample_bytree=[0.6, 0.8, 1.0]),
    "xgboost": dict(max_leaves=[63, 127, 255, 511], min_child_weight=[1, 3, 5, 10], reg_lambda=[1.0, 5.0, 10.0],
                    colsample_bytree=[0.6, 0.8, 1.0]),
    "catboost": dict(depth=[6, 8, 10], l2_leaf_reg=[1.0, 3.0, 10.0, 30.0], rsm=[0.7, 1.0]),
    "random_forest": dict(max_features=[0.4, 0.6, 0.8, 1.0], min_samples_leaf=[2, 3, 5, 8]),
    "extra_trees": dict(max_features=[0.6, 0.8, 1.0], min_samples_leaf=[2, 3, 5, 8]),
    "ridge": dict(alpha=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]),
    "mlp": dict(hidden_layer_sizes=[[64], [128, 64], [128, 64, 32]], alpha=[1e-5, 1e-4, 1e-3],
                learning_rate_init=[3e-3, 1e-2]),
}
FIT = {"hgb": fit_hgb, "lightgbm": fit_lgbm, "xgboost": fit_xgb, "catboost": fit_cat,
       "random_forest": fit_forest(RandomForestRegressor), "extra_trees": fit_forest(ExtraTreesRegressor),
       "ridge": fit_ridge, "median_lookup": fit_median, "mlp": fit_mlp}
FIXED = {"hgb": "fixed: the current train.py config (3,500 rounds, lr 0.05, 255 leaves, min leaf 5, L2 1.0); 1 validation run only",
         "median_lookup": "none (rule based: median of log price by route+airline+class, fallback route+class, then class)"}


def sample_trials(name, n):
    sp = SPACES[name]
    rng = np.random.RandomState(123)
    out, seen = [], set()
    for _ in range(200):
        t = {k: v[rng.randint(len(v))] for k, v in sp.items()}
        key = json.dumps(t, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(t)
        if len(out) == n:
            break
    return out


# ----------------------------------------------------------------------------------------------- run
def model_mb(obj):
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        path = f.name
    try:
        joblib.dump(obj, path)
        return os.path.getsize(path) / 1e6
    finally:
        os.unlink(path)


def predict_ms(fn, te):
    n = min(len(te.y), 50_000)
    sub = D(te.X.iloc[:n], te.df.iloc[:n])
    best = 1e9
    for _ in range(2):
        t = time.perf_counter()
        fn(sub)
        best = min(best, time.perf_counter() - t)
    return best * 1e6 / n  # ms per 1,000 rows (feature encoding excluded)


def run_algo(name, sc, data, outdir, seed, force, log):
    ck = outdir / f"{name}_{sc}{'' if seed == 0 else f'_seed{seed}'}.json"
    if ck.exists() and not force:
        log(f"[{sc}] {name} seed{seed}: checkpoint exists, skipping")
        return json.loads(ck.read_text())
    res = {"algorithm": name, "scenario": sc, "seed": seed}
    try:
        tune_info = {"desc": FIXED.get(name)}
        if seed == 0:
            # ---- tuning on the scenario's fixed validation split
            trials, best = [], None
            grid = sample_trials(name, N_TRIALS) if name in SPACES else [{}]
            if name == "median_lookup":
                trials = []
                params = {}
            else:
                for i, p in enumerate(grid):
                    t = time.time()
                    fn, _, info = FIT[name](data["tt"], p, 0, True, data["cx_tune"])
                    v = metrics(np.exp(fn(data["tv"])), data["tv"].price)["accuracy"]
                    trials.append({"params": p, "val_accuracy": v, "seconds": round(time.time() - t, 1)})
                    log(f"[{sc}] {name} tune {i + 1}/{len(grid)} val_acc={v} {p} ({time.time() - t:.0f}s)")
                    if best is None or v > best[0]:
                        best = (v, p)
                params = best[1]
                tune_info.update(n_trials=len(grid), best_val_accuracy=best[0], trials=trials)
                if name in SPACES:
                    tune_info["desc"] = (f"random search, {len(grid)} trials on <={TUNE_N:,} sampled training rows, validated on a fixed "
                                         f"{ {'A': 'random 10% hold-out', 'E': 'hold-out of 2023 flight-days', 'C': 'latest-10%-of-2022 hold-out'}[sc]}; "
                                         "boosters tuned at lr 0.1 then refit at lr 0.05 / nets 10 -> 25 epochs / forests 40 -> 120 trees")
        else:
            params = json.loads((outdir / f"{name}_{sc}.json").read_text())["params"]
            tune_info = json.loads((outdir / f"{name}_{sc}.json").read_text())["tuning"]
        # ---- final fit on the full scenario training set
        t = time.time()
        fn, obj, info = FIT[name](data["tr"], params, seed, False, data["cx"])
        train_s = time.time() - t
        pred_log = fn(data["te"])
        sc_ = metrics(np.exp(pred_log), data["te"].price)
        res.update(params=params, tuning=tune_info, scores=sc_, train_seconds=round(train_s, 1),
                   predict_ms_per_1k=round(predict_ms(fn, data["te"]), 2), fit_info=info, n_test=len(data["te"].y))
        if seed == 0:
            res["model_mb"] = round(model_mb(obj), 2)
            if name != "median_lookup":
                np.save(outdir / "preds" / f"{name}_{sc}.npy", pred_log.astype(np.float32))
        log(f"[{sc}] {name} seed{seed}: {sc_} train={train_s:.0f}s")
    except Exception as e:  # keep going; record why
        res.update(error=f"{type(e).__name__}: {e}")
        log(f"[{sc}] {name} FAILED: {res['error']}")
    ck.write_text(json.dumps(res, indent=2))
    return res


def run_blend(sc, data, outdir, force, log):
    ck = outdir / f"blend_{sc}.json"
    if ck.exists() and not force:
        return
    cands = []
    for n in ORDER:
        f = outdir / f"{n}_{sc}.json"
        p = outdir / "preds" / f"{n}_{sc}.npy"
        if n in ("blend",) or not (f.exists() and p.exists()):
            continue
        r = json.loads(f.read_text())
        v = (r.get("tuning") or {}).get("best_val_accuracy")
        if v is not None and "error" not in r and r["fit_info"] is not None:
            cands.append((v, n))
    cands.sort(reverse=True)
    top = [n for _, n in cands[:3]]
    if len(top) < 2:
        log(f"[{sc}] blend: not enough members")
        return
    preds = np.mean([np.load(outdir / "preds" / f"{n}_{sc}.npy").astype(float) for n in top], axis=0)
    mem = [json.loads((outdir / f"{n}_{sc}.json").read_text()) for n in top]
    sc_ = metrics(np.exp(preds), data["te"].price)
    res = {"algorithm": "blend", "scenario": sc, "seed": 0, "params": {"members": top, "weights": "equal, averaged in log-price"},
           "tuning": {"desc": "no fitting: equal-weight average of the 3 algorithms with the best VALIDATION accuracy (picked on the tuning split, not the test fold)"},
           "scores": sc_, "train_seconds": round(sum(m["train_seconds"] for m in mem), 1),
           "predict_ms_per_1k": round(sum(m["predict_ms_per_1k"] for m in mem), 2),
           "model_mb": round(sum(m["model_mb"] for m in mem), 2), "fit_info": {}, "n_test": len(data["te"].y)}
    log(f"[{sc}] blend of {top}: {sc_}")
    ck.write_text(json.dumps(res, indent=2))


def prepare(sc, df, feature_set):
    tr, te_df, nums, te, (tt, tv) = build_scenario(sc, df, [])
    if feature_set == "enriched":
        nums = nums + enriched_extra(df, tr, True)
    enc = Encoder(nums, te=te, supply=True)
    Xtr = enc.fit_transform(tr)
    cx = {"cat_idx": enc.cat_idx, "enc": enc, "features": enc.features}
    enc2 = Encoder(nums, te=te, supply=True)
    Xtt = enc2.fit_transform(tt)
    cx_t = {"cat_idx": enc2.cat_idx, "enc": enc2, "features": enc2.features}
    return {"tr": D(Xtr, tr), "te": D(enc.transform(te_df), te_df), "tt": D(Xtt, tt), "tv": D(enc2.transform(tv), tv),
            "cx": cx, "cx_tune": cx_t, "n_features": len(enc.features)}


# ----------------------------------------------------------------------------------------------- assemble
def read_all(outdir):
    out = {}
    for f in outdir.glob("*.json"):
        r = json.loads(f.read_text())
        if r.get("seed", 0) == 0:
            out[(r["algorithm"], r["scenario"])] = r
    return out


def assemble(outdir, feature_set, dest, n_tests, only=None):
    rs = read_all(outdir)
    results, win = [], {}
    for name in ORDER:
        if only and name not in only and name != "blend":
            continue
        per = {sc: rs.get((name, sc)) for sc in "ACE"}
        if not any(per.values()):
            continue
        scores = {sc: (r["scores"] if r and "error" not in r else None) for sc, r in per.items()}
        ref = next((r for sc in "AEC" if (r := per[sc]) and "error" not in r), None)
        notes = ['requested_by: portfolio manager'] if name == 'random_forest' else []
        for sc, r in per.items():
            if r is None:
                notes.append(f"{sc}: not run")
            elif "error" in r:
                notes.append(f"{sc}: FAILED ({r['error']})")
        if ref:
            notes.append("train seconds " + ", ".join(f"{sc}={per[sc]['train_seconds']:.0f}" for sc in "ACE" if per[sc] and "error" not in per[sc])
                         + " (final fit, excludes tuning)")
            iters = {sc: per[sc]["fit_info"] for sc in "ACE" if per[sc] and "error" not in per[sc] and per[sc].get("fit_info")}
            if iters:
                notes.append("fit info " + json.dumps(iters))
        base = rs.get((name, "A")) if rs.get((name, "A")) and "error" not in rs[(name, "A")] else ref
        results.append({
            "algorithm": NAMES[name], "family": FAMILY[name],
            "params": {sc: per[sc]["params"] for sc in "AEC" if per[sc] and "error" not in per[sc]},
            "tuning": (ref["tuning"]["desc"] if ref else None),
            "scores": {sc: scores[sc] for sc in "AEC"},
            "train_seconds": base["train_seconds"] if base else None,
            "predict_ms_per_1k": base["predict_ms_per_1k"] if base else None,
            "model_mb": base.get("model_mb") if base else None,
            "notes": "; ".join(notes) + ("; train_seconds / predict_ms_per_1k / model_mb are for scenario A (predict time excludes feature encoding)" if base else "")})
    for sc in "AEC":
        ok = [(r["scores"][sc]["accuracy"], r["algorithm"]) for r in results if r["scores"][sc]]
        win[sc] = max(ok)[1] if ok else None
    out = {"generated": date.today().isoformat(), "feature_set": feature_set,
           "scenarios": {sc: {"label": SCEN_LABEL[sc], "n_test": n_tests.get(sc)} for sc in "AEC"},
           "results": results, "winner": win}
    dest.write_text(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--scenarios", default="A,E,C")
    ap.add_argument("--feature-set", default="baseline18", choices=["baseline18", "enriched"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--assemble-only", action="store_true")
    ap.add_argument("--noise", action="store_true", help="refit top 3 with seeds 1,2 (reuses tuned params)")
    ap.add_argument("--top3", action="store_true", help="only the 3 best algorithms of the baseline18 run (for the enriched pass)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    smoke_tag = "_smoke" if a.smoke else ""
    base = ROOT / "models/algo_runs" / (a.feature_set + smoke_tag)
    (base / "preds").mkdir(parents=True, exist_ok=True)
    dest = ROOT / "models" / (("algorithms.json" if a.feature_set == "baseline18" else "algorithms_enriched.json"))
    if a.smoke:
        dest = base / "algorithms_smoke.json"
    logf = open(base / "run.log", "a")

    def log(s):
        line = f"{time.strftime('%H:%M:%S')} {s}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    scs = [s for s in a.scenarios.split(",") if s]
    only = [s for s in a.only.split(",") if s] or None
    n_tests = {}
    for sc in "AEC":
        f = base / f"hgb_{sc}.json"
        if f.exists():
            n_tests[sc] = json.loads(f.read_text()).get("n_test")

    names = only or ORDER
    seeds = [0]
    if a.top3 or a.noise:
        ref = ROOT / "models/algo_runs" / ("baseline18" + smoke_tag)
        rs = read_all(ref)
        score = {}
        for n in ORDER:
            if n == "blend":
                continue
            accs = [rs[(n, sc)]["scores"]["accuracy"] for sc in "AEC" if (n, sc) in rs and "error" not in rs[(n, sc)]]
            if len(accs) == 3:
                score[n] = float(np.mean(accs))
        top = sorted(score, key=score.get, reverse=True)[:3]
        log(f"top 3 by mean accuracy across scenarios (baseline18): {top}")
        names = top
        if a.noise:
            seeds = [1, 2]
            base = ref  # noise checkpoints live next to the seed-0 ones (they reuse their tuned params)

    if not a.assemble_only:
        df = load_df(a.feature_set if not a.noise else "baseline18", a.smoke)
        for sc in scs:
            t0 = time.time()
            data = prepare(sc, df, a.feature_set if not a.noise else "baseline18")
            n_tests[sc] = len(data["te"].y)
            log(f"[{sc}] prepared: train {len(data['tr'].y):,} rows x {data['n_features']} features, test {n_tests[sc]:,}, "
                f"tune-train {len(data['tt'].y):,}, val {len(data['tv'].y):,} ({time.time() - t0:.0f}s)")
            for name in names:
                if name == "blend":
                    run_blend(sc, data, base, a.force, log)
                    continue
                for seed in seeds:
                    run_algo(name, sc, data, base, seed, a.force, log)
            if names == ORDER or (only is None and not (a.top3 or a.noise)):
                run_blend(sc, data, base, a.force, log)
            # incremental assemble so partial results are always on disk
            if not a.noise:
                assemble(base, a.feature_set, dest, n_tests, only=(names if a.top3 else None))

    if a.noise:
        out = {"note": "model seed 1 and 2 re-fits of the top 3 algorithms (same splits, same tuned params); seed 0 is the headline run",
               "seeds": [0, 1, 2], "algorithms": {}}
        for name in names:
            out["algorithms"][NAMES[name]] = {}
            for sc in "AEC":
                vals = []
                for s in (0, 1, 2):
                    f = base / f"{name}_{sc}{'' if s == 0 else f'_seed{s}'}.json"
                    if f.exists():
                        r = json.loads(f.read_text())
                        if "error" not in r:
                            vals.append(r["scores"]["accuracy"])
                if vals:
                    out["algorithms"][NAMES[name]][sc] = {"accuracy_by_seed": vals, "mean": round(float(np.mean(vals)), 3),
                                                          "std": round(float(np.std(vals, ddof=1)), 3) if len(vals) > 1 else None,
                                                          "range": round(max(vals) - min(vals), 3)}
        (ROOT / "models" / ("algorithms_noise.json" if not a.smoke else "algorithms_noise_smoke.json")).write_text(json.dumps(out, indent=2))
    elif a.assemble_only:
        assemble(base, a.feature_set, dest, n_tests)
    log("DONE")


if __name__ == "__main__":
    main()
