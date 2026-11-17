"""Train the FINAL deployable engines on all training rows and write models/engines/*.joblib + manifest.json.

    python src/build_engines.py                      # everything not yet built (long; run in the background)
    python src/build_engines.py --only xgboost,ridge # some engines
    python src/build_engines.py --force              # rebuild even if the file exists
    python src/build_engines.py --assemble-only      # rebuild manifest.json from the per-engine meta files

Protocol = src/train.py --final-only: all rows of data/merged.parquet (class != First), the SAME encoder configuration
as the deployed models/fare_model.joblib (its fitted ENC is used at inference for every engine; a freshly fitted copy is
asserted identical before training), best scenario-A parameters from models/algorithms.json, fit functions imported from
compare_models.py. HistGradientBoosting is NOT retrained: the deployed model is wrapped as is.
Random Forest / Extra Trees: the benchmark versions were 920 MB / 2 GB, so the deployable versions are capped at
<= CAP_MB by choosing, among a few smaller configurations, the most accurate one on scenario A (retrained on the A train split).
"""
import sys

if __name__ == "__main__":
    import compare_models as _cm
    _cm._bootstrap()  # thread cap (half the cores) + libomp for lightgbm/xgboost on macOS

import argparse
import json
import platform
import resource
import subprocess
import time
import warnings
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

import compare_models as cm
import engines
from compare_models import D, FIT, NT, NAMES, FAMILY, metrics, model_mb
from features import CATS, Encoder, add_basic
from train import NUMS, TE

warnings.filterwarnings("ignore")
ROOT = cm.ROOT
OUT = ROOT / "models" / "engines"
META = OUT / "_meta"
CAP_MB = 150.0
KEYS = ["hgb", "xgboost", "lightgbm", "catboost", "random_forest", "extra_trees", "ridge", "mlp"]
DESC = {
    "hgb": "Builds thousands of small decision trees one after another, each fixing the previous ones' mistakes (scikit-learn gradient boosting). This is the current model.",
    "xgboost": "Another gradient-boosting tree ensemble: many small trees added in sequence, each correcting the errors left so far. Typically the most accurate here.",
    "lightgbm": "Gradient boosting that grows trees leaf by leaf; very similar accuracy to the others and quick to train.",
    "catboost": "Gradient boosting with 'symmetric' trees and built-in handling of text-like categories (airline, route); small and fast to run, less accurate on this data.",
    "random_forest": "Averages hundreds of independent decision trees, each trained on a random sample of flights. Simple and robust, but cannot extrapolate beyond price levels it has seen.",
    "extra_trees": "Like a random forest but with more randomness in where each tree splits; fast to train, a little better than a random forest here.",
    "ridge": "A straight-line (linear) formula over the flight details, one weight per feature. A simple baseline that shows how much the non-linear models add.",
    "mlp": "A small neural network (a few layers of weighted sums) over the flight details; here it learns less well than tree models on this table-style data.",
}
FOREST_CANDIDATES = {  # (n_trees, min_samples_leaf, max_samples, bootstrap); sizes are verified by really fitting on all rows
    "random_forest": [(25, 10, None, True), (25, 5, 0.5, True), (50, 10, 0.5, True), (50, 20, None, True), (12, 5, None, True)],
    "extra_trees": [(16, 10, None, False), (30, 20, None, False), (60, 40, None, False), (30, 10, 0.5, True), (50, 20, 0.5, True)],
}
FOREST_CLS = {"random_forest": RandomForestRegressor, "extra_trees": ExtraTreesRegressor}


def log(s):
    print(f"{time.strftime('%H:%M:%S')} {s}", flush=True)


def versions():
    v = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    import sklearn
    v["scikit-learn"], v["joblib"] = sklearn.__version__, joblib.__version__
    for m in ("xgboost", "lightgbm", "catboost"):
        try:
            v[m] = __import__(m).__version__
        except Exception:
            pass
    return v


def bench_params():
    d = json.loads((ROOT / "models/algorithms.json").read_text())
    by = {r["algorithm"]: r for r in d["results"]}
    return {k: by[NAMES[k]] for k in KEYS}


def clip_ranges(X):
    """training min/max of every non-categorical column (Ridge/MLP clip inputs to this so calendar numbers cannot extrapolate)"""
    return {c: [float(X[c].min()), float(X[c].max())] for c in X.columns if c not in CATS}


def fit_final(key, params, D_full, cx):
    _, obj, info = FIT[key](D_full, params, 0, False, cx)
    return obj, info


def sample_search(df, enc):
    """Delhi -> Mumbai, 2026-12-10 (Thursday), Economy: every flight on that route, 21 days before departure"""
    rows = df[(df.source == "Delhi") & (df.destination == "Mumbai") & (df["class"] == "Economy") & (df.period == df.period.max())]
    rows = rows.drop_duplicates("flight_code").copy()
    rows["days_left"], rows["dow"], rows["month"], rows["period"] = 21, 3, 12, int(df.period.max())
    return enc.transform(add_basic(rows))


def measure(key, heldin_df, feats_heldin, bench_dir, df, enc):
    engines.clear_cache()
    t = time.perf_counter()
    eng = engines.load_engine(key)
    load_s = time.perf_counter() - t
    best = 1e9
    for _ in range(2):
        t = time.perf_counter()
        pred = eng.predict(feats_heldin)
        best = min(best, time.perf_counter() - t)
    ms1k = best * 1e6 / len(feats_heldin)
    price = heldin_df.price.values.astype(float)
    acc = lambda p: round(float(100 - (np.abs(np.exp(p) - price) / price).mean() * 100), 2)
    san = {"rows": len(feats_heldin), "finite": bool(np.isfinite(pred).all()), "accuracy_on_heldin_rows": acc(pred),
           "note": "these rows were in the final model's training data (held-in), so accuracy here is optimistic by design"}
    bp = bench_dir / f"{key}_A.npy"
    if bp.exists():
        b = np.load(bp).astype(float)[:len(pred)]
        d = np.abs(pred - b)
        san.update(benchmark_mean_abs_log_diff=round(float(d.mean()), 4), benchmark_p95_abs_log_diff=round(float(np.quantile(d, .95)), 4),
                   benchmark_corr=round(float(np.corrcoef(pred, b)[0, 1]), 4), benchmark_accuracy_same_rows=acc(b),
                   tolerance="mean |log diff| <= 0.08 and corr >= 0.99 (benchmark model saw only the scenario-A train rows, the engine all rows)")
        san["agrees_with_benchmark"] = bool(d.mean() <= 0.08 and san["benchmark_corr"] >= 0.99)
    sp = np.exp(eng.predict(sample_search(df, enc)))
    san["sample_search"] = {"query": "Delhi -> Mumbai, 2026-12-10 (Thu), Economy, 21 days ahead", "flights": int(len(sp)),
                            "finite": bool(np.isfinite(sp).all()), "min_inr": round(float(sp.min())), "mean_inr": round(float(sp.mean())),
                            "max_inr": round(float(sp.max()))}
    # peak memory of a fresh process that only loads this engine and predicts the sample
    tmp = OUT / "_meta" / f"_probe_{key}.parquet"
    feats_heldin.iloc[:2000].to_parquet(tmp)
    code = ("import sys,resource,pandas as pd;sys.path.insert(0,'src');import engines;e=engines.load_engine(%r);"
            "e.predict(pd.read_parquet(%r));r=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;"
            "print(r/1e6 if sys.platform=='darwin' else r/1e3)") % (key, str(tmp))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                         env={**__import__("os").environ, "_CMP_BOOT": "1"})
    tmp.unlink(missing_ok=True)
    try:
        mem = round(float(out.stdout.strip().splitlines()[-1]), 0)
    except Exception:
        mem = None
    return {"load_seconds": round(load_s, 2), "predict_ms_per_1k": round(ms1k, 2), "memory_mb_peak": mem, "sanity": san}


def build(keys, force):
    OUT.mkdir(parents=True, exist_ok=True); META.mkdir(exist_ok=True)
    df = cm.load_df("baseline18", False)
    bundle = joblib.load(ROOT / "models/fare_model.joblib")
    ENC = bundle["enc"]
    enc = Encoder(NUMS, te=TE, supply=True)
    Xfull = enc.fit_transform(df)
    # the freshly fitted encoder must equal the deployed one, otherwise one encoder could not serve all engines
    for n in TE:
        a, b = enc.stats[n], ENC.stats[n]
        assert a.index.equals(b.index) and np.allclose(a.values, b.values), f"encoder stats differ for {n}"
    assert enc.features == ENC.features
    probe = df.sample(500, random_state=0)
    assert np.allclose(enc.transform(probe).values.astype(float), ENC.transform(probe).values.astype(float), equal_nan=True)
    log(f"encoder identical to the deployed ENC; {len(df):,} training rows x {len(enc.features)} features")
    D_full = D(Xfull, df)
    cx = {"cat_idx": enc.cat_idx, "enc": enc, "features": enc.features}
    cat_sizes = {c: len(enc.cats[c]) for c in CATS}
    bp = bench_params()

    # held-in rows = the 20,000 first scenario-A test rows (benchmark predictions were saved in that order)
    _, te_df, _, _, _ = cm.build_scenario("A", df, [])
    heldin = te_df.iloc[:20000]
    feats_heldin = ENC.transform(heldin)
    bench_dir = ROOT / "models/algo_runs/baseline18/preds"
    dataA = None

    for key in keys:
        f_meta = META / f"{key}.json"
        if key != "hgb" and (OUT / f"{key}.joblib").exists() and f_meta.exists() and not force:
            log(f"{key}: exists, skipping"); continue
        t0 = time.time()
        meta = {"key": key, "versions": versions(), "built": date.today().isoformat(), "n_train_rows": len(df)}
        if key == "hgb":
            meta.update(file="models/fare_model.joblib", params=bp[key]["params"]["A"],
                        notes="the deployed model, wrapped as is (not retrained)")
            meta["train_seconds"] = None
        elif key in FOREST_CLS:
            if dataA is None:
                dataA = cm.prepare("A", df, "baseline18")
            ref = {k: v for k, v in bp[key]["params"]["A"].items()}
            cls, rows, best = FOREST_CLS[key], [], None
            for (n, leaf, ms, boot) in FOREST_CANDIDATES[key]:
                p = dict(ref, n_estimators=n, min_samples_leaf=leaf, bootstrap=boot, **({"max_samples": ms} if ms else {}))
                t = time.time()
                m = cls(n_jobs=NT, random_state=0, **p).fit(Xfull.fillna(-1), D_full.y)
                ts = time.time() - t
                mb = model_mb(m)
                row = {"params": p, "full_data_model_mb": round(mb, 1), "train_seconds": round(ts, 1)}
                log(f"{key} candidate {p}: {mb:.0f} MB on all rows")
                if mb <= CAP_MB:
                    mA = cls(n_jobs=NT, random_state=0, **p).fit(dataA["tr"].X.fillna(-1), dataA["tr"].y)
                    row["scenario_A_scores"] = metrics(np.exp(mA.predict(dataA["te"].X.fillna(-1))), dataA["te"].price)
                    row["scenario_A_model_mb"] = round(model_mb(mA), 1)
                    del mA
                    log(f"{key} candidate scenario A: {row['scenario_A_scores']}")
                    if best is None or row["scenario_A_scores"]["accuracy"] > best[0]["scenario_A_scores"]["accuracy"]:
                        best = (row, m)
                rows.append(row)
                if best is None or best[1] is not m:
                    del m
            if best is None:
                raise SystemExit(f"{key}: no candidate fits under {CAP_MB} MB")
            obj, chosen = best[1], best[0]
            full = bp[key]
            meta.update(params=chosen["params"], train_seconds=chosen["train_seconds"],
                        capped={"cap_mb": CAP_MB, "chosen": chosen, "candidates": rows,
                                "accuracy_measured_on": "scenario A: capped config retrained on the A train split (356,178 rows), scored on the A test split (89,044 rows)"},
                        full_size={"params": full["params"]["A"], "model_mb": full["model_mb"], "scores": full["scores"]},
                        notes=f"size-capped for deployment (<= {CAP_MB:.0f} MB); full-size benchmark model was {full['model_mb']:.0f} MB")
            joblib.dump({"format": 1, "kind": key, "model": obj, "features": enc.features, "cat_sizes": cat_sizes, "versions": meta["versions"]},
                        OUT / f"{key}.joblib")
            meta["file"] = f"models/engines/{key}.joblib"
        else:
            params = bp[key]["params"]["A"]
            obj, info = fit_final(key, params, D_full, cx)
            meta.update(params=params, train_seconds=round(time.time() - t0, 1), fit_info=info)
            joblib.dump({"format": 1, "kind": key, "model": obj, "features": enc.features, "cat_sizes": cat_sizes, "versions": meta["versions"],
                         **({"clip": clip_ranges(Xfull)} if key in ("ridge", "mlp") else {})}, OUT / f"{key}.joblib")
            meta["file"] = f"models/engines/{key}.joblib"
        log(f"{key}: fitted/saved ({time.time() - t0:.0f}s)")
        f_meta.write_text(json.dumps(meta, indent=2)); assemble()  # manifest must list the engine before it can be loaded
        meta.update(measure(key, heldin, feats_heldin, bench_dir, df, ENC))
        log(f"{key}: {json.dumps({k: meta[k] for k in ('load_seconds', 'predict_ms_per_1k', 'memory_mb_peak')})} sanity={json.dumps(meta['sanity'])}")
        f_meta.write_text(json.dumps(meta, indent=2))
        obj = None
        assemble()


REQUIRES = {"xgboost": ["xgboost"], "lightgbm": ["lightgbm"], "catboost": ["catboost"]}


def assemble():
    bp = bench_params()
    dec_f = META / "december_check.json"
    december = json.loads(dec_f.read_text()) if dec_f.exists() else {}
    engines_out = []
    for key in KEYS:
        f = META / f"{key}.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        path = ROOT / m["file"]
        r = bp[key]
        sc = r["scores"]
        entry = {
            "key": key, "label": NAMES[key].replace(" (current model)", "") + (" (current)" if key == "hgb" else ""),
            "family": FAMILY[key], "default": key == "xgboost", "description": DESC[key], "file": m["file"],
            "size_mb": round(path.stat().st_size / 1e6, 1), "load_seconds": m.get("load_seconds"),
            "predict_ms_per_1k": m.get("predict_ms_per_1k"), "memory_mb_peak": m.get("memory_mb_peak"),
            "accuracy": {s: sc[s]["accuracy"] for s in "AEC"}, "within10": {s: sc[s]["within10"] for s in "AEC"},
            "mae": {s: sc[s]["mae"] for s in "AEC"},
            "scores_note": "A/E/C scores are the benchmark runs (models/algorithms.json, full-size model trained on each scenario's train split), not this deployed file",
            "requires": ["numpy", "pandas", "scikit-learn", "joblib"] + REQUIRES.get(key, []),
            "versions_built_with": m["versions"], "params": m["params"], "sanity": m.get("sanity"), "notes": m.get("notes", ""),
        }
        entry["scores"] = {sc_: sc[sc_] for sc_ in "AEC"}  # full {accuracy, within10, mae, rmse, r2} per scenario (benchmark)
        if m.get("clip_applied"):
            entry["input_clipping"] = ("month, weekday, days_left and every other plain-number input are clipped to the range seen in training "
                                       "before predicting (a December search is treated like the nearest trained month, April)")
        dec = december.get(key)
        if dec:
            entry["december_check"] = dec["means"]
            entry["december_vs_median_engine_pct"] = dec["vs_median_pct"]
            entry["demo_safe"] = dec["demo_safe"]
            if not dec["demo_safe"]:
                entry["notes"] = (entry["notes"] + "; " if entry["notes"] else "") + (
                    "CAUTION: on the December sanity searches this engine's mean fare differs from the median engine by more than 25% "
                    f"({dec['vs_median_pct']}); use only for dates inside the training months (Jan-Apr)")
        else:
            entry["demo_safe"] = None
        if "capped" in m:
            c = m["capped"]["chosen"]
            entry["capped"] = {"cap_mb": m["capped"]["cap_mb"], "params": c["params"], "scenario_A_scores": c["scenario_A_scores"],
                               "scenario_A_model_mb": c["scenario_A_model_mb"], "measured_on": m["capped"]["accuracy_measured_on"]}
            entry["full_size"] = {"model_mb": m["full_size"]["model_mb"], "params": m["full_size"]["params"],
                                  "accuracy": {s: m["full_size"]["scores"][s]["accuracy"] for s in "AEC"},
                                  "within10": {s: m["full_size"]["scores"][s]["within10"] for s in "AEC"},
                                  "mae": {s: m["full_size"]["scores"][s]["mae"] for s in "AEC"}}
            entry["accuracy_A_capped"] = c["scenario_A_scores"]["accuracy"]
        engines_out.append(entry)
    (OUT / "manifest.json").write_text(json.dumps({
        "generated": date.today().isoformat(), "feature_set": "baseline18", "encoder": "the deployed ENC in models/fare_model.joblib (one encoder serves every engine)",
        "default_engine": "xgboost", "interface": "load_engine(key).predict(feats) -> predicted log price (numpy)",
        "engines": engines_out}, indent=2))


def december_check():
    """Mean fare of every engine for two December searches (outside the Jan-Apr training months); flags engines >25% from the median engine."""
    df = cm.load_df("baseline18", False)
    ENC = joblib.load(ROOT / "models/fare_model.joblib")["enc"]
    routes = [("Delhi", "Mumbai"), ("Ahmedabad", "Mumbai")]
    frames = {}
    for src_, dst in routes:
        rows = df[(df.source == src_) & (df.destination == dst) & (df["class"] == "Economy") & (df.period == df.period.max())].drop_duplicates("flight_code").copy()
        rows["days_left"], rows["dow"], rows["month"], rows["period"] = 21, 3, 12, int(df.period.max())  # 2026-12-10 is a Thursday
        frames[f"{src_} -> {dst}"] = (ENC.transform(add_basic(rows)), len(rows))
    means = {}
    for key in KEYS:
        if not (ROOT / json.loads((META / f"{key}.json").read_text())["file"]).exists():
            continue
        engines.clear_cache()
        eng = engines.load_engine(key)
        means[key] = {r: round(float(np.exp(eng.predict(f)).mean())) for r, (f, n) in frames.items()}
    out = {}
    med = {r: float(np.median([m[r] for m in means.values()])) for r in frames}
    for key, m in means.items():
        dev = {r: round((m[r] / med[r] - 1) * 100, 1) for r in m}
        out[key] = {"means": m, "vs_median_pct": dev, "demo_safe": bool(all(abs(v) <= 25 for v in dev.values()))}
    out["_median_engine_mean_inr"] = med
    out["_query"] = "Economy, 2026-12-10 (Thu), 21 days ahead; flights per route: " + ", ".join(f"{r}={n}" for r, (f, n) in frames.items())
    (META / "december_check.json").write_text(json.dumps(out, indent=2))
    return out


def patch_clip():
    """Add training-range clipping to the Ridge/MLP engine files WITHOUT retraining, re-run their agreement check, redo the December check."""
    df = cm.load_df("baseline18", False)
    ENC = joblib.load(ROOT / "models/fare_model.joblib")["enc"]
    enc = Encoder(NUMS, te=TE, supply=True)
    clip = clip_ranges(enc.fit_transform(df))
    _, te_df, _, _, _ = cm.build_scenario("A", df, [])
    heldin = te_df.iloc[:20000]
    feats_heldin = ENC.transform(heldin)
    for key in ("ridge", "mlp"):
        f = OUT / f"{key}.joblib"
        payload = joblib.load(f)
        payload["clip"] = clip
        joblib.dump(payload, f)
        meta = json.loads((META / f"{key}.json").read_text())
        meta.setdefault("before_clip", {k: meta["sanity"].get(k) for k in ("accuracy_on_heldin_rows", "benchmark_mean_abs_log_diff", "benchmark_corr", "agrees_with_benchmark")})
        meta["clip_applied"] = True
        meta.update(measure(key, heldin, feats_heldin, ROOT / "models/algo_runs/baseline18/preds", df, ENC))
        (META / f"{key}.json").write_text(json.dumps(meta, indent=2))
        log(f"{key}: before {meta['before_clip']} -> after {{acc {meta['sanity']['accuracy_on_heldin_rows']}, diff {meta['sanity'].get('benchmark_mean_abs_log_diff')}, corr {meta['sanity'].get('benchmark_corr')}}}")
    out = december_check()
    log("december check: " + json.dumps({k: v for k, v in out.items()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-clip", action="store_true")
    ap.add_argument("--december-check", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--assemble-only", action="store_true")
    a = ap.parse_args()
    keys = [k for k in a.only.split(",") if k] or KEYS
    if a.patch_clip:
        patch_clip(); assemble(); log("DONE"); return
    if a.december_check:
        december_check(); assemble(); log("DONE"); return
    if not a.assemble_only:
        build(keys, a.force)
    assemble()
    log("DONE")


if __name__ == "__main__":
    main()
