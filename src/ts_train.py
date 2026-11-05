"""Train and honestly evaluate a price-over-time model on the US Airfare Time Machine features.

    OMP_NUM_THREADS=2 python src/ts_train.py <stage> [--pct 10] [--threads 2]
    stages: tune | splits | ladder | importance | scale | merge | all

Target: log price, learned as a RESIDUAL against an anchor known at the search date
(log last observed price of the same flight; else log route 7-day median; else a constant). Predictions are
exp(anchor + model) so metrics are on the price scale. Accuracy = 100 - MAPE (the project metric).
Splits (search dates): T train<=2026-08-15, val 08-16..08-31, test 09-01..09-30; F hold out 20% of flights;
R hold out 5 routes (and their reverse directions are removed from training too).
Checkpoints go to models/ts_work/*.json; `merge` builds models/ts_metrics.json.
"""
import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import os, importlib.util  # noqa: E401
if sys.platform == "darwin" and not os.environ.get("_TS_BOOT"):   # lightgbm/xgboost need the libomp that scikit-learn ships
    _d = Path(importlib.util.find_spec("sklearn").submodule_search_locations[0]) / ".dylibs"
    os.execve(sys.executable, [sys.executable] + sys.argv, dict(os.environ, _TS_BOOT="1", DYLD_FALLBACK_LIBRARY_PATH=f"{_d}:" + os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")))
import lightgbm  # noqa: E402,F401

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data/external_us"
FEATS = EXT / "features.parquet"
import os
WORK = Path(os.environ.get("TS_WORK", ROOT / "models/ts_work"))
WORK.mkdir(parents=True, exist_ok=True)
THREADS = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 2
PCT = float(sys.argv[sys.argv.index("--pct") + 1]) if "--pct" in sys.argv else 10.0
SDN_TRAIN_END, SDN_VAL_END = 226, 242     # 2026-08-15, 2026-08-31 (days since 2026-01-01)
LOGC = float(np.log(180.0))
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


BASICS = ["dbd", "search_dow", "dep_dow", "dep_hour", "dep_month", "airline_id", "route_id", "stops", "dur", "distance_mi"]
HISTORY = ["last_price", "last_gap", "p_1d", "p_3d", "p_7d", "min3", "mean3", "std3", "min7", "mean7", "std7",
           "run_min", "run_max", "days_since_change", "n_changes", "n_obs",
           "lr_p1", "lr_p3", "lr_p7", "lr_min3", "lr_mean7", "lr_runmin", "lr_runmax"]    # lr_* = log(last_price / x): trees cannot form ratios themselves
ROUTE_MKT = ["lr_route", "route_med7", "airline_rel7", "airline_rank7", "n_airlines", "n_flights_day", "day_coverage"]
CALENDAR = [f"{t}_{c}" for t in "sd" for c in ["is_holiday", "days_to_hol", "days_since_hol", "hol_weekend", "days_to_easter",
            "school_summer", "school_spring_break", "school_winter", "thanksgiving_window", "school_break_any", "is_weekend"]]
FUEL = [f"fx_{k}{s}" for k in ("jet", "brent") for s in ("", "_chg7", "_chg30")]
WX = ["tmax", "tmin", "precip", "snow", "wind"]
WEATHER = [f"wxl_{s}_{v}" for s in "od" for v in WX]
WEATHER_ORACLE = [f"wxa_{s}_{v}" for s in "od" for v in WX]
FAMILIES = {"basics": BASICS, "history": HISTORY, "route_market": ROUTE_MKT, "calendar": CALENDAR, "fuel": FUEL, "weather": WEATHER}
CATS = ["airline_id", "route_id"]
ID_COLS = ["fid", "sdn", "ddn", "price"]


# ------------------------------------------------------------------ data
def load(pct=PCT, extra_cols=()):
    import duckdb
    con = duckdb.connect(); con.execute(f"set threads={THREADS}; set memory_limit='6GB'")
    cols = sorted({c for c in ID_COLS + sum(FAMILIES.values(), []) + WEATHER_ORACLE + list(extra_cols) if not c.startswith('lr_')}, key=str)
    lim = int(round(pct * 10))
    df = con.sql(f"select {','.join(cols)} from '{FEATS}' where (fid % 1000) < {lim}").df()
    for c in df.columns:
        if c not in ("fid",) and df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    df["fid"] = df.fid.astype("uint64")
    lp = np.log(df.last_price)
    for nm, col in [("lr_p1", "p_1d"), ("lr_p3", "p_3d"), ("lr_p7", "p_7d"), ("lr_min3", "min3"), ("lr_mean7", "mean7"),
                    ("lr_runmin", "run_min"), ("lr_runmax", "run_max"), ("lr_route", "route_med7")]:
        df[nm] = (lp - np.log(df[col])).astype(np.float32)
    df["fgrp"] = ((df.fid // 1000) % 100).astype(np.int8)       # independent of the sampling hash: 0..99
    log(f"loaded {len(df):,} rows, {df.fid.nunique():,} flights, pct={pct}")
    return df


def route_tables(con_df):
    return con_df


def feature_list(fams, drop=()):
    cols = []
    for f in fams:
        cols += FAMILIES[f] if isinstance(f, str) else f
    return [c for c in dict.fromkeys(cols) if c not in drop]


def anchor_of(df, fams):
    a = np.full(len(df), LOGC, dtype=np.float64)
    if "route_market" in fams:
        m = df.route_med7.values
        a = np.where(np.isnan(m), a, np.log(np.where(np.isnan(m), 1, m)))
    if "history" in fams:
        m = df.last_price.values
        a = np.where(np.isnan(m), a, np.log(np.where(np.isnan(m), 1, m)))
    return a


def make_X(df, cols):
    X = df[cols].copy()
    for c in CATS:
        if c in X:
            X[c] = pd.Categorical(X[c].astype("int16"), categories=range(60 if c == "route_id" else 20))
    return X


# ------------------------------------------------------------------ splits
def split_T(df):
    return df.sdn <= SDN_TRAIN_END, (df.sdn > SDN_TRAIN_END) & (df.sdn <= SDN_VAL_END), df.sdn > SDN_VAL_END


def split_F(df):
    te = (df.fgrp % 5) == 0
    return ~te, None, te


def pick_R(routes_csv=ROOT / "data/us-airfare-time-machine/routes.csv"):
    rt = pd.read_csv(routes_csv)
    rng = np.random.RandomState(11)
    names = list(rt.route)
    chosen = []
    while len(chosen) < 5:
        r = names[rng.randint(len(names))]
        a, b = r.split("-")
        if r not in chosen and f"{b}-{a}" not in chosen:
            chosen.append(r)
    return chosen


def route_ids(names):
    import duckdb
    con = duckdb.connect(str(EXT / "ts.duckdb"), read_only=True)
    rt = con.sql("select route_id, route from rt").df()
    m = dict(zip(rt.route, rt.route_id))
    inv = dict(zip(rt.route_id, rt.route))
    return [m[n] for n in names], m, inv


def split_R(df):
    names = pick_R()
    rev = [f"{r.split('-')[1]}-{r.split('-')[0]}" for r in names]
    ids, m, _ = route_ids(names)
    rids, _, _ = route_ids(rev)
    te = df.route_id.isin(ids)
    tr = ~(te | df.route_id.isin(rids))
    return tr, None, te, names


# ------------------------------------------------------------------ metrics
def acc_block(y, p):
    ape = np.abs(p - y) / y
    return {"rows": int(len(y)), "accuracy_pct": round(100 - float(ape.mean()) * 100, 3),
            "within_10pct": round(float((ape <= 0.10).mean()) * 100, 2),
            "MAE_usd": round(float(np.abs(p - y).mean()), 2), "RMSE_usd": round(float(np.sqrt(((p - y) ** 2).mean())), 2)}


BUCKETS = [(7, 14), (15, 30), (31, 60), (61, 120), (121, 180)]


def by_bucket(df, y, p):
    out = {}
    for lo, hi in BUCKETS:
        m = ((df.dbd >= lo) & (df.dbd <= hi)).values
        if m.sum():
            out[f"{lo}-{hi}"] = acc_block(y[m], p[m])
    return out


def by_route(df, y, p, inv):
    out = {}
    for rid, g in df.groupby("route_id"):
        i = g.index.values if False else np.where(df.route_id.values == rid)[0]
        out[inv[int(rid)]] = acc_block(y[i], p[i])
    return out


def direction(df, y, p, ref_col, rdm=None):
    """hit_rate_persistence: baseline that keeps the last observed move going (sign(last_price - reference))."""
    """Sign of price change vs the reference observation (last observed price, or the one ~7 days earlier)."""
    ref = df[ref_col].values.astype(np.float64)
    ok = ~np.isnan(ref)
    y, p, ref = y[ok], p[ok], ref[ok]
    ach, pch = y - ref, p - ref
    nz = ach != 0
    res = {"rows_with_reference": int(ok.sum()), "share_actually_unchanged": round(float((~nz).mean()) * 100, 2)}
    sa, sp = np.sign(ach[nz]), np.sign(pch[nz])
    res["hit_rate_on_changed_pct"] = round(float((sa == sp).mean()) * 100, 2)
    up = float((sa > 0).mean())
    res["majority_class_pct"] = round(max(up, 1 - up) * 100, 2)
    res["pct_changes_that_are_up"] = round(up * 100, 2)
    big = np.abs(ach) / ref > 0.10
    if big.sum():
        res["hit_rate_changes_over_10pct"] = round(float((np.sign(ach[big]) == np.sign(pch[big])).mean()) * 100, 2)
        res["rows_changes_over_10pct"] = int(big.sum())
    res["MAE_change_usd_model"] = round(float(np.abs(pch - ach).mean()), 2)
    res["MAE_change_usd_no_change_baseline"] = round(float(np.abs(ach).mean()), 2)
    lastp = df["last_price"].values.astype(np.float64)[ok]
    pers = np.sign(lastp - ref)
    gp = nz & (pers != 0)
    if gp.sum():
        res["hit_rate_persistence_baseline_pct"] = round(float((pers[gp] == np.sign(ach[gp])).mean()) * 100, 2)
        res["persistence_baseline_rows"] = int(gp.sum())
        res["hit_rate_model_on_same_rows_pct"] = round(float((np.sign(pch[gp]) == np.sign(ach[gp])).mean()) * 100, 2)
    if rdm is not None:
        rdm = rdm[ok]
        good = nz & ~np.isnan(rdm)
        if good.sum() == 0:
            return res
        res["hit_rate_route_dbd_median_reversion_pct"] = round(float((np.sign(rdm[good] - ref[good]) == np.sign(ach[good])).mean()) * 100, 2)
    return res


def route_dbd_median(train, test):
    """Naive baseline (b): median price of (route, days-before-departure) on TRAIN rows; falls back to route median."""
    g = train.groupby(["route_id", "dbd"]).price.median()
    r = train.groupby("route_id").price.median()
    idx = pd.MultiIndex.from_arrays([test.route_id.values, test.dbd.values])
    v = g.reindex(idx).values
    v = np.where(np.isnan(v), test.route_id.map(r).values, v)
    return v


def full_eval(model_pred, df_te, train, inv, label_extra=None):
    y = df_te.price.values.astype(np.float64)
    p = model_pred
    rdm = route_dbd_median(train, df_te)
    hist = ~df_te.last_price.isna().values
    out = {"model_all_rows": acc_block(y, p), "model_rows_with_history": acc_block(y[hist], p[hist]),
           "naive_last_price_rows_with_history": acc_block(y[hist], df_te.last_price.values[hist].astype(np.float64)),
           "naive_route_dbd_median_all_rows": acc_block(y, rdm) if not np.isnan(rdm).any() else None,
           "naive_route_dbd_median_rows_with_history": acc_block(y[hist], rdm[hist]) if not np.isnan(rdm).any() else None,
           "share_rows_with_history_pct": round(float(hist.mean()) * 100, 2),
           "by_dbd_bucket_model": by_bucket(df_te.reset_index(drop=True), y, p),
           "by_dbd_bucket_naive_last": {k: v for k, v in by_bucket(df_te[hist].reset_index(drop=True), y[hist], df_te.last_price.values[hist].astype(np.float64)).items()},
           "by_dbd_bucket_naive_route_median": by_bucket(df_te.reset_index(drop=True), y, rdm) if not np.isnan(rdm).any() else None,
           "by_route_model": by_route(df_te.reset_index(drop=True), y, p, inv),
           "direction_vs_last_observation": direction(df_te.reset_index(drop=True), y, p, "last_price", rdm),
           "direction_vs_price_7d_earlier": direction(df_te.reset_index(drop=True), y, p, "p_7d", rdm)}
    return out


# ------------------------------------------------------------------ models
def fit_model(algo, params, Xtr, rtr, Xva, rva, rounds=800):
    """rtr/rva: residual targets (log price - anchor). Returns a predict(X)->residual callable."""
    if algo == "lgb":
        import lightgbm as lgb
        p = dict(objective="l1", learning_rate=0.08, num_threads=THREADS, verbose=-1, min_child_samples=50,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, max_bin=127, **params)
        dtr = lgb.Dataset(Xtr, rtr, free_raw_data=True)
        dva = lgb.Dataset(Xva, rva, reference=dtr)
        m = lgb.train(p, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(40, verbose=False)])
        return m, (lambda X: m.predict(X, num_iteration=m.best_iteration)), m.best_iteration
    if algo == "xgb":
        import xgboost as xgb
        p = dict(objective="reg:absoluteerror", eta=0.1, tree_method="hist", nthread=THREADS, max_bin=127,
                 subsample=0.8, colsample_bytree=0.8, min_child_weight=20, enable_categorical=True)
        p.update(params)
        dtr = xgb.DMatrix(Xtr, rtr, enable_categorical=True); dva = xgb.DMatrix(Xva, rva, enable_categorical=True)
        m = xgb.train(p, dtr, rounds, evals=[(dva, "val")], early_stopping_rounds=40, verbose_eval=False)
        bi = m.best_iteration + 1
        return m, (lambda X: m.predict(xgb.DMatrix(X, enable_categorical=True), iteration_range=(0, bi))), bi
    if algo == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        cat = [c in CATS for c in Xtr.columns]
        def num(X):
            X = X.copy()
            for c in CATS:
                if c in X:
                    X[c] = X[c].cat.codes.replace(-1, np.nan).astype(np.float32)
            return X.values.astype(np.float32)
        p = dict(loss="absolute_error", learning_rate=0.1, max_iter=rounds, early_stopping=True, validation_fraction=0.05, n_iter_no_change=25,
                 l2_regularization=1.0, min_samples_leaf=50, categorical_features=cat, random_state=0, max_bins=127)
        params = dict(params); clip = params.pop("clip", None)
        p.update(params)
        m = HistGradientBoostingRegressor(**p).fit(num(Xtr), np.clip(rtr, -clip, clip) if clip else rtr)
        return m, (lambda X: m.predict(num(X))), m.n_iter_
    raise ValueError(algo)


GRID = {"lgb": [dict(num_leaves=63), dict(num_leaves=255), dict(num_leaves=127, lambda_l2=10.0)],
        # xgb / hgb: an L1 loss collapses to 'no change' (65% of next prices are unchanged, so every leaf median is 0): use pseudo-Huber / clipped-residual L2
        "xgb": [dict(objective="reg:pseudohubererror", huber_slope=0.05, max_depth=6), dict(objective="reg:pseudohubererror", huber_slope=0.1, max_depth=8),
                dict(objective="reg:pseudohubererror", huber_slope=0.05, max_depth=8, reg_lambda=10.0)],
        "hgb": [dict(loss="squared_error", clip=0.25, max_leaf_nodes=63), dict(loss="squared_error", clip=0.25, max_leaf_nodes=255),
                dict(loss="squared_error", clip=0.5, max_leaf_nodes=127, l2_regularization=10.0)]}


def residuals(df, fams, cols):
    a = anchor_of(df, fams)
    return a, (np.log(df.price.values.astype(np.float64)) - a)


def prepare(df, fams, drop=()):
    cols = feature_list(fams, drop)
    a, r = residuals(df, fams, cols)
    return make_X(df, cols), a, r, cols


def tune(algo, df, fams, tr_mask, va_mask, n_tune=1_000_000):
    """Pick hyper-parameters on the VALIDATION window only; tune on a flight-sampled subset for speed."""
    tr = df[tr_mask]
    if len(tr) > n_tune:
        keep = tr.fgrp < max(1, int(100 * n_tune / len(tr)))
        tr = tr[keep]
    va = df[va_mask]
    if len(va) > 300_000:
        va = va[va.fgrp < max(1, int(100 * 300_000 / len(va)))]
    Xt, at, rt, cols = prepare(tr, fams); Xv, av, rv, _ = prepare(va, fams)
    best = None
    for params in GRID[algo]:
        t = time.time()
        _, pred, it = fit_model(algo, params, Xt, rt, Xv, rv)
        p = np.exp(pred(Xv) + av)
        a = acc_block(va.price.values.astype(np.float64), p)["accuracy_pct"]
        log(f"  tune {algo} {params}: val acc {a} iters {it} ({time.time() - t:.0f}s)")
        if best is None or a > best[0]:
            best = (a, params)
    return best[1], best[0]


def run_split(algo, params, df, fams, split, inv, drop=()):
    if split == "T":
        trm, vam, tem = split_T(df); extra = {}
    elif split == "F":
        trm, vam, tem = split_F(df); extra = {}
    else:
        trm, vam, tem, names = split_R(df); extra = {"held_out_routes": names, "reverse_routes_also_removed_from_training": True}
        drop = tuple(drop) + ("route_id",)
    tr, te = df[trm], df[tem]
    if vam is None:   # early-stopping validation: 10% of training FLIGHTS (not rows) so rows of one flight stay together
        vm = tr.fgrp.isin(range(5, 15)) if split == "F" else tr.fgrp < 10
        va, tr = tr[vm], tr[~vm]
    else:
        va = df[vam]
    Xt, at, rt, cols = prepare(tr, fams, drop); Xv, av, rv, _ = prepare(va, fams, drop); Xe, ae, re_, _ = prepare(te, fams, drop)
    t = time.time()
    m, pred, it = fit_model(algo, params, Xt, rt, Xv, rv)
    pv = np.exp(pred(Xv) + av); pe = np.exp(pred(Xe) + ae)
    log(f"  fit {algo} {split}: iters {it}, {time.time() - t:.0f}s, train {len(tr):,}, test {len(te):,}")
    res = {"algo": algo, "params": params, "iterations": int(it), "train_rows": int(len(tr)), "val_rows": int(len(va)),
           "test_rows": int(len(te)), "val": acc_block(va.price.values.astype(np.float64), pv), **extra}
    res.update(full_eval(pe, te.reset_index(drop=True), tr, inv))
    return res, (m, pred, cols, te, pe)


def save_json(name, obj):
    (WORK / name).write_text(json.dumps(obj, indent=1))


def load_json(name, default=None):
    p = WORK / name
    return json.loads(p.read_text()) if p.exists() else default


def inv_routes():
    return route_ids([])[2]


# ------------------------------------------------------------------ stages
ALL = ["basics", "history", "route_market", "calendar", "fuel", "weather"]


def stage_splits(df):
    inv = inv_routes()
    best = load_json("best_params.json", {})
    out = load_json("splits.json", {})
    for algo in ("lgb", "xgb", "hgb"):
        trm, vam, _ = split_T(df)
        if algo not in best:
            log("tuning", algo, "on validation window")
            params, va = tune(algo, df, ALL, trm, vam)
            best[algo] = {"params": params, "val_acc": va}
            save_json("best_params.json", best)
        for split in ("T", "F", "R"):
            key = f"{algo}_{split}"
            if key in out:
                continue
            log("running", key)
            res, _ = run_split(algo, best[algo]["params"], df, ALL, split, inv)
            out[key] = res
            save_json("splits.json", out)
            a = res["model_rows_with_history"]["accuracy_pct"]
            log(f"  {key}: acc(all rows) {res['model_all_rows']['accuracy_pct']}, acc(with history) {a}, "
                f"naive last {res['naive_last_price_rows_with_history']['accuracy_pct']}, naive route-dbd {(res['naive_route_dbd_median_all_rows'] or {}).get('accuracy_pct')}")


def stage_ladder(df, algo="lgb"):
    inv = inv_routes()
    best = load_json("best_params.json", {})
    params = best[algo]["params"]
    out = load_json("ladder.json", {})
    steps = [("basics", ["basics"]), ("+history", ["basics", "history"]), ("+route_market", ["basics", "history", "route_market"]),
             ("+calendar", ["basics", "history", "route_market", "calendar"]), ("+fuel", ["basics", "history", "route_market", "calendar", "fuel"]),
             ("+weather", ALL),
             # extras that isolate each family on top of basics (is the family useful without history?)
             ("basics+route_market", ["basics", "route_market"]), ("basics+calendar", ["basics", "calendar"]),
             ("basics+fuel", ["basics", "fuel"]), ("basics+weather", ["basics", "weather"]),
             ("all+weather_ORACLE(actual dep-date weather, not known at booking)", ALL + [WEATHER_ORACLE])]
    for name, fams in steps:
        if name in out:
            continue
        res, _ = run_split(algo, params, df, fams, "T", inv)
        keep = {k: res[k] for k in ("val", "model_all_rows", "model_rows_with_history", "naive_last_price_rows_with_history",
                                    "naive_route_dbd_median_all_rows", "direction_vs_last_observation", "iterations", "train_rows")}
        keep["families"] = [f if isinstance(f, str) else "weather_oracle" for f in fams]
        out[name] = keep
        save_json("ladder.json", out)
        log(f"  ladder {name}: acc(all) {res['model_all_rows']['accuracy_pct']} acc(hist) {res['model_rows_with_history']['accuracy_pct']}")


def stage_importance(df, algo="lgb", n=250_000):
    inv = inv_routes()
    best = load_json("best_params.json", {})
    res, (m, pred, cols, te, pe) = run_split(algo, best[algo]["params"], df, ALL, "T", inv)
    te = te.reset_index(drop=True)
    sub = te.sample(min(n, len(te)), random_state=0).reset_index(drop=True)
    y = sub.price.values.astype(np.float64)

    def score(d):
        X = make_X(d, cols); p = np.exp(pred(X) + anchor_of(d, ALL))
        return acc_block(y, p)["accuracy_pct"]
    base = score(sub)
    rng = np.random.RandomState(0)
    out = {"baseline_accuracy_pct": base, "rows": int(len(sub)), "features": {}, "families": {}}
    for c in cols:
        d = sub.copy(); d[c] = d[c].values[rng.permutation(len(d))]
        out["features"][c] = round(base - score(d), 4)
    for fam, fc in FAMILIES.items():
        drops = []
        for r in range(3):
            d = sub.copy(); perm = rng.permutation(len(d))
            for c in fc:
                d[c] = d[c].values[perm]
            drops.append(base - score(d))
        out["families"][fam] = {"mean_drop_pct_points": round(float(np.mean(drops)), 4), "std": round(float(np.std(drops)), 4)}
    out["note"] = "accuracy drop (percentage points) when the column(s) are shuffled on the test window; whole family shuffled together with one permutation. The anchor is recomputed from the shuffled last_price/route_med7."
    save_json("importance.json", out)
    log("importance done; top:", sorted(out["features"].items(), key=lambda kv: -kv[1])[:8])


def stage_scale(df_big_pct):
    """LightGBM split T at a larger flight sample, to see whether more data helps."""
    inv = inv_routes()
    best = load_json("best_params.json", {})
    df = load(df_big_pct)
    res, (m, pred, cols, te, pe) = run_split("lgb", best["lgb"]["params"], df, ALL, "T", inv)
    res.pop("by_route_model", None)
    out = load_json("scale.json", {})
    out[f"{df_big_pct}pct"] = res
    save_json("scale.json", out)


def stage_direction(df):
    """Dedicated 'will it go up or down' classifier: among rows whose price DID change vs the last observation,
    predict up vs down (LightGBM binary, split T, validation window for early stopping, test = Sept)."""
    import lightgbm as lgb
    cols = feature_list(ALL)
    d = df[df.last_price.notna() & (df.price != df.last_price)]
    trm, vam, tem = split_T(d)
    tr, va, te = d[trm], d[vam], d[tem]
    up = lambda x: (x.price.values > x.last_price.values).astype(int)
    p = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=100, feature_fraction=0.8, bagging_fraction=0.8,
             bagging_freq=1, num_threads=THREADS, verbose=-1, max_bin=127)
    dtr = lgb.Dataset(make_X(tr, cols), up(tr)); dva = lgb.Dataset(make_X(va, cols), up(va), reference=dtr)
    m = lgb.train(p, dtr, 1000, valid_sets=[dva], callbacks=[lgb.early_stopping(40, verbose=False)])
    pr = m.predict(make_X(te, cols), num_iteration=m.best_iteration)
    yt = up(te)
    from sklearn.metrics import roc_auc_score
    hit = float(((pr > 0.5) == yt).mean()) * 100
    maj = max(yt.mean(), 1 - yt.mean()) * 100
    lp7 = te.lr_p7.values; good = ~np.isnan(lp7)
    out = {"test_rows_that_changed": int(len(te)), "hit_rate_pct": round(hit, 2), "majority_class_pct": round(float(maj), 2), "auc": round(float(roc_auc_score(yt, pr)), 4),
           "iterations": int(m.best_iteration), "train_rows": int(len(tr)),
           "hit_rate_mean_reversion_vs_route_pct": round(float(((te.lr_route.values > 0) == (yt == 0)).mean()) * 100, 2) if te.lr_route.notna().all() else
           round(float((((te.lr_route.values > 0) == (yt == 0))[~te.lr_route.isna().values]).mean()) * 100, 2),
           "note": "only rows whose price differs from the last observed price; this answers 'given a move, up or down?'. P(any move) is a separate question: 65% of next observations are unchanged."}
    sh = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False)
    out["top_gain_features"] = {k: round(float(v / sh.sum()), 4) for k, v in sh.head(10).items()}
    save_json("direction.json", out)
    log("direction classifier:", out)


def stage_final(df):
    """Fit the chosen model on the T-split training window (+validation) and save the deliverable joblib."""
    import joblib
    inv = inv_routes()
    best = load_json("best_params.json", {})
    sp = load_json("splits.json", {})
    algo = max(("lgb", "xgb", "hgb"), key=lambda a: sp[f"{a}_T"]["val"]["accuracy_pct"])   # choose on VALIDATION, not test
    log("final algorithm (by validation accuracy):", algo)
    res, (m, pred, cols, te, pe) = run_split(algo, best[algo]["params"], df, ALL, "T", inv)
    joblib.dump({"algo": algo, "model": m, "columns": cols, "families": ALL, "anchor_rule": "log(last_price) else log(route_med7) else log(180); pred = exp(anchor + model)",
                 "categorical": CATS, "route_names": inv, "params": best[algo]["params"], "iterations": res["iterations"],
                 "trained_on": "search dates <= 2026-08-15 (early stopping on 08-16..08-31); flight sample pct=%s" % PCT,
                 "note": "Research artefact on US 2026 data. Needs history features built by src/ts_features.py to run."},
                ROOT / "models/fare_model_ts.joblib", compress=3)
    save_json("final.json", {"algo": algo, "test_accuracy_pct": res["model_all_rows"]["accuracy_pct"]})


def stage_merge():
    sp = load_json("splits.json", {}); lad = load_json("ladder.json", {}); imp = load_json("importance.json", {})
    sc = load_json("scale.json", {}); dr = load_json("direction.json", {}); best = load_json("best_params.json", {}); fin = load_json("final.json", {})
    out = {"generated": time.strftime("%Y-%m-%d %H:%M"), "dataset": "US Airfare Time Machine (38.46M flight-days after collapsing duplicate quotes)",
           "metric": "accuracy = 100 - MAPE on price (USD); within 10%; MAE; RMSE",
           "flight_sample_pct": PCT, "splits": {"T": "train search<=2026-08-15, val 08-16..08-31, test 09-01..09-30",
                                                "F": "20% of flights held out entirely", "R": "5 routes held out (+reverse directions dropped from training)"},
           "best_params_by_validation": best, "final_model": fin, "results": sp, "ablation_ladder_T_lightgbm": lad,
           "permutation_importance_T_test": imp, "direction_classifier_T": dr, "scale_check": sc}
    (ROOT / "models/ts_metrics.json").write_text(json.dumps(out, indent=1))
    log("wrote models/ts_metrics.json")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "merge":
        stage_merge(); sys.exit()
    if stage == "scale":
        stage_scale(float(sys.argv[2])); sys.exit()
    df = load()
    if stage in ("splits", "all"): stage_splits(df)
    if stage in ("ladder", "all"): stage_ladder(df)
    if stage in ("importance", "all"): stage_importance(df)
    if stage in ("direction", "all"): stage_direction(df)
    if stage in ("final", "all"): stage_final(df)
    if stage == "all": stage_merge()
