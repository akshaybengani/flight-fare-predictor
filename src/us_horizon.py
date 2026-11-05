"""Multi-horizon ("what will this flight cost N days from now?") model + demo snapshot for src/us_service.py.

    python src/us_horizon.py base       # as-of base rows for a 10% flight sample  -> data/external_us/asof_sample.parquet
    python src/us_horizon.py train      # T-split evaluation at horizons 1/3/7/14, final model -> models/fare_model_ts.joblib
    python src/us_horizon.py snapshot   # flights seen on the last search day (2026-09-30) -> data/external_us/snapshot.parquet
    python src/us_horizon.py all

Definition. A row is (flight f, as-of search day s, horizon h). Features are everything known at the END of day s
(this flight's quotes up to and including s, route-market over the 7 days up to s, fuel at s) plus things known in advance
about day s+h (weekday, holiday calendar, days before departure) and climatological weather. Target = price of f on search
day s+h (the same flight must be quoted that day). The model learns log(price_{s+h} / price_s), so a prediction of 0 is
"price stays the same". One LightGBM model with `ahead` (= h) as a feature, trained on horizons 1,2,3,5,7,10,14.
Splits are by TARGET date: train <= 2026-08-15, validation 08-16..08-31, test 09-01..09-30 (the honest evaluation).
The final (served) model is refit on all target dates up to 2026-09-30 with the validated number of rounds.
"""
import os
import sys
import json
import time
import warnings
import importlib.util
from pathlib import Path

if sys.platform == "darwin" and not os.environ.get("_USH_BOOT"):    # lightgbm needs the libomp that scikit-learn ships (macOS, training only)
    _d = Path(importlib.util.find_spec("sklearn").submodule_search_locations[0]) / ".dylibs"
    os.execve(sys.executable, [sys.executable] + sys.argv, dict(os.environ, _USH_BOOT="1", DYLD_FALLBACK_LIBRARY_PATH=f"{_d}:" + os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")))

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import us_service as U   # noqa: E402  (feature definition shared with serving; imports numpy/pandas only)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/us-airfare-time-machine"
EXT = ROOT / "data/external_us"
DB = EXT / "ts.duckdb"
SAMPLE = EXT / "asof_sample.parquet"
SNAP = EXT / "snapshot.parquet"
LAST_SDN = 272                                  # 2026-09-30
TRAIN_END, VAL_END = 226, 242                   # 2026-08-15, 2026-08-31
HORIZONS = [1, 2, 3, 5, 7, 10, 14]
EVAL_H = [1, 3, 7, 14]
THREADS = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 4
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


# ------------------------------------------------------------------ SQL base rows
def route_market_asof(con):
    rd = con.sql("select route_id, sdn, median(price) as med, count(*) as nq from q group by all").df()
    ad = con.sql("select route_id, airline_id, sdn, median(price) as amed from q group by all").df()
    days = np.arange(rd.sdn.min(), rd.sdn.max() + 1)
    med = rd.pivot(index="sdn", columns="route_id", values="med").reindex(days)
    r7 = med.rolling(7, min_periods=1).mean().stack().rename("route_med7").reset_index()      # INCLUDES the as-of day
    day_nq = rd.groupby("sdn").nq.sum().reindex(days)
    cov = (day_nq / day_nq.rolling(15, center=True, min_periods=3).median()).rename("day_coverage").reset_index()
    cov.columns = ["sdn", "day_coverage"]
    ad = ad.merge(rd[["route_id", "sdn", "med"]], on=["route_id", "sdn"])
    ad["rel"] = ad.amed / ad.med
    ad["rank_pct"] = ad.groupby(["route_id", "sdn"]).amed.rank(pct=True)
    out = []
    for (rid, aid), g in ad.groupby(["route_id", "airline_id"]):
        g = g.set_index("sdn").reindex(days)
        o = pd.DataFrame({"airline_rel7": g.rel.rolling(7, min_periods=1).mean(), "airline_rank7": g.rank_pct.rolling(7, min_periods=1).mean()})
        o["route_id"], o["airline_id"] = rid, aid
        out.append(o.reset_index().rename(columns={"index": "sdn"}))
    arank = pd.concat(out, ignore_index=True).dropna(subset=["airline_rel7"])
    return r7, arank, cov


def build_base(where, name):
    """Per (flight, as-of day) history features (inclusive of the as-of day) + route-market as-of + static fields."""
    import duckdb
    con = duckdb.connect(str(DB))
    con.execute(f"set threads={THREADS}; set memory_limit='8GB'; set temp_directory='{EXT / 'duck_tmp'}'")
    r7, arank, cov = route_market_asof(con)
    con.register("r7", r7); con.register("arank", arank); con.register("cov", cov)
    con.register("rts_csv", pd.read_csv(RAW / "routes.csv")[["route", "distance_miles"]])
    con.execute(f"""create or replace table ha_{name} as
      with c as (select *, case when price <> lag(price) over (partition by fid order by sdn) then 1 else 0 end as chg from q where {where})
      select fid, sdn, ddn, route_id, airline_id, dbd, dep_min, stops, dur, price,
        last_value(price) over (partition by fid order by sdn range between unbounded preceding and 1 preceding) as p_1d,
        last_value(price) over (partition by fid order by sdn range between unbounded preceding and 3 preceding) as p_3d,
        last_value(price) over (partition by fid order by sdn range between unbounded preceding and 7 preceding) as p_7d,
        min(price) over (partition by fid order by sdn rows between 2 preceding and current row) as min3,
        avg(price) over (partition by fid order by sdn rows between 2 preceding and current row) as mean3,
        stddev_samp(price) over (partition by fid order by sdn rows between 2 preceding and current row) as std3,
        min(price) over (partition by fid order by sdn rows between 6 preceding and current row) as min7,
        avg(price) over (partition by fid order by sdn rows between 6 preceding and current row) as mean7,
        stddev_samp(price) over (partition by fid order by sdn rows between 6 preceding and current row) as std7,
        min(price) over (partition by fid order by sdn rows between unbounded preceding and current row) as run_min,
        max(price) over (partition by fid order by sdn rows between unbounded preceding and current row) as run_max,
        max(case when chg = 1 then sdn end) over (partition by fid order by sdn rows between unbounded preceding and current row) as last_chg_sdn,
        sum(chg) over (partition by fid order by sdn rows between unbounded preceding and current row) as n_changes,
        first_value(sdn) over (partition by fid order by sdn) as first_sdn,
        row_number() over (partition by fid order by sdn) as n_obs
      from c""")
    comp = "select route_id, ddn, sdn, count(distinct airline_id) as n_airlines, count(*) as n_flights_day from q group by all"
    df = con.sql(f"""
      select h.fid, h.sdn, h.ddn, h.route_id, h.airline_id, h.dbd, h.dep_min, h.stops, h.dur, h.price,
        h.p_1d, h.p_3d, h.p_7d, h.min3, h.mean3, h.std3, h.min7, h.mean7, h.std7, h.run_min, h.run_max,
        (h.sdn - coalesce(h.last_chg_sdn, h.first_sdn)) as days_since_change, h.n_changes, h.n_obs,
        r7.route_med7, arank.airline_rel7, arank.airline_rank7, comp.n_airlines, comp.n_flights_day, cov.day_coverage,
        rt.origin, rt.destination, rts_csv.distance_miles as distance_mi
      from ha_{name} h
      join rt on rt.route_id = h.route_id join rts_csv on rts_csv.route = rt.route
      left join r7 on r7.route_id = h.route_id and r7.sdn = h.sdn
      left join arank on arank.route_id = h.route_id and arank.airline_id = h.airline_id and arank.sdn = h.sdn
      left join ({comp}) comp on comp.route_id = h.route_id and comp.ddn = h.ddn and comp.sdn = h.sdn
      left join cov on cov.sdn = h.sdn""").df()
    con.close()
    air = pd.read_csv(RAW / "airports.csv").iata.tolist()
    am = {a: i for i, a in enumerate(air)}
    df["o_idx"] = df.origin.map(am).astype("int16"); df["d_idx"] = df.destination.map(am).astype("int16")
    df["fid"] = df.fid.astype("uint64")
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


# ------------------------------------------------------------------ lookup tables stored with the model
def make_tables():
    cal = pd.read_parquet(EXT / "calendar.parquet")
    dn = (cal.date - U.EPOCH).dt.days.values
    assert (np.diff(dn) == 1).all()
    vals = cal[U.CAL_COLS].values.astype(np.float32)
    calt = {"vals": vals, "dow": cal.date.dt.dayofweek.values.astype(np.int8), "month": cal.date.dt.month.values.astype(np.int8),
            "doy": cal.date.dt.dayofyear.clip(upper=365).values.astype(np.int16)}
    fu = pd.read_parquet(EXT / "fuel.parquet")
    fdn = (fu.date - U.EPOCH).dt.days.values
    assert (np.diff(fdn) == 1).all()
    fuel = fu[["jet", "brent", "jet_chg7", "jet_chg30", "brent_chg7", "brent_chg30"]].values.astype(np.float32)
    air = pd.read_csv(RAW / "airports.csv")
    wc = pd.read_parquet(EXT / "weather_clim.parquet")
    wx = np.zeros((len(air), 365, 5), dtype=np.float32)
    for i, a in enumerate(air.iata):
        g = wc[wc.airport == a].sort_values("doy")
        wx[i] = g[[f"wxl_{v}" for v in U.WX]].values
    return {"cal": calt, "cal_dn0": int(dn[0]), "fuel": fuel, "fuel_dn0": int(fdn[0]), "wx_clim": wx}, air


# ------------------------------------------------------------------ rows for training / evaluation
def horizon_rows(base, h, T, frac, windows, seed):
    """Rows (as-of s, target s+h observed). windows: dict name -> (lo, hi, frac) on the TARGET day."""
    key = base.fid_code.values.astype(np.int64) * 512 + base.sdn.values
    index = pd.Index(key)
    pos = index.get_indexer(key + h)
    tsdn = base.sdn.values + h
    rng = np.random.RandomState(seed + h)
    out = {}
    for name, (lo, hi, fr) in windows.items():
        m = (pos >= 0) & (tsdn >= lo) & (tsdn <= hi)
        if fr < 1:
            m &= rng.random_sample(len(base)) < fr
        sel = np.nonzero(m)[0]
        b = base.iloc[sel]
        X = U.make_features(b, h, T)
        y = np.log(base.price.values[pos[sel]].astype(np.float64) / b.price.values.astype(np.float64))
        out[name] = (X, y, b[["price", "route_id", "dbd"]].reset_index(drop=True), base.price.values[pos[sel]].astype(np.float64))
    return out


def acc_block(y, p):
    ape = np.abs(p - y) / y
    return {"rows": int(len(y)), "accuracy_pct": round(100 - float(ape.mean()) * 100, 3), "within_10pct": round(float((ape <= .1).mean()) * 100, 2),
            "MAE_usd": round(float(np.abs(p - y).mean()), 2), "RMSE_usd": round(float(np.sqrt(((p - y) ** 2).mean())), 2)}


def export_trees(booster):
    d = booster.dump_model()
    feat, thr, left, right, dl, val = [], [], [], [], [], []
    roots, max_depth = [], 0

    def add(node, depth):
        nonlocal max_depth
        i = len(feat)
        max_depth = max(max_depth, depth)
        if "leaf_value" in node:
            feat.append(0); thr.append(np.inf); left.append(i); right.append(i); dl.append(True); val.append(node["leaf_value"])
            return i
        assert node["decision_type"] == "<=" and node["missing_type"] in ("None", "NaN"), node
        feat.append(node["split_feature"]); t = node["threshold"]; thr.append(t)
        dl.append(bool(node["default_left"]) if node["missing_type"] == "NaN" else bool(0.0 <= t))
        left.append(-1); right.append(-1); val.append(0.0)
        left[i] = add(node["left_child"], depth + 1); right[i] = add(node["right_child"], depth + 1)
        return i
    for t in d["tree_info"]:
        roots.append(add(t["tree_structure"], 0))
    return {"feat": np.array(feat, np.int32), "thr": np.array(thr, np.float64), "left": np.array(left, np.int32),
            "right": np.array(right, np.int32), "default_left": np.array(dl, bool), "value": np.array(val, np.float64),
            "roots": np.array(roots, np.int32), "max_depth": int(max_depth)}


PARAMS = dict(objective="l1", learning_rate=0.08, num_leaves=127, lambda_l2=10.0, min_child_samples=50, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, max_bin=127, verbose=-1)


def stage_train():
    import lightgbm as lgb
    base = pd.read_parquet(SAMPLE)
    base["fid_code"] = pd.factorize(base.fid)[0].astype(np.int64)
    log(f"base rows {len(base):,}, flights {base.fid_code.max() + 1:,}")
    T, air = make_tables()
    # naive (b): median price by (route, days-before-departure) on TRAIN-window quotes
    trn = base[base.sdn <= TRAIN_END]
    rdm = trn.groupby(["route_id", "dbd"]).price.median()
    rmed = trn.groupby("route_id").price.median()
    p = dict(PARAMS, num_threads=THREADS)
    res, best, trees_by_h, chk_all = {}, {}, {}, {}
    for h in HORIZONS:
        r = horizon_rows(base, h, T, None, {"train": (0, TRAIN_END, 0.6), "val": (TRAIN_END + 1, VAL_END, 0.5), "test": (VAL_END + 1, 999, 1.0)}, 1)
        (Xtr, ytr, _, _), (Xva, yva, _, _) = r["train"], r["val"]
        dtr = lgb.Dataset(Xtr.values, ytr, feature_name=U.FEATURES); dva = lgb.Dataset(Xva.values, yva, reference=dtr)
        m = lgb.train(p, dtr, 1500, valid_sets=[dva], callbacks=[lgb.early_stopping(40, verbose=False)])
        bi = int(m.best_iteration); best[h] = bi
        X, y, meta, tprice = r["test"]
        pred = meta.price.values * np.exp(m.predict(X.values, num_iteration=bi))
        naive = meta.price.values.astype(np.float64)
        bmed = rdm.reindex(pd.MultiIndex.from_arrays([meta.route_id.values, meta.dbd.values - h])).values
        bmed = np.where(np.isnan(bmed), meta.route_id.map(rmed).values, bmed)
        a, n, bb = acc_block(tprice, pred), acc_block(tprice, naive), acc_block(tprice, bmed)
        ch = tprice - naive; pc = pred - naive; nz = ch != 0
        d = {"model": a, "naive_last_price": n, "naive_route_days_out_median": bb, "delta_vs_naive_points": round(a["accuracy_pct"] - n["accuracy_pct"], 3),
             "share_price_unchanged_pct": round(float((~nz).mean()) * 100, 1),
             "direction_hit_rate_when_changed_pct": round(float((np.sign(pc[nz]) == np.sign(ch[nz])).mean()) * 100, 1),
             "best_iteration": bi, "train_rows": int(len(ytr)), "val_rows": int(len(yva))}
        d["verdict"] = ("better than carrying the last price forward" if d["delta_vs_naive_points"] >= 0.1 else
                        "no better than carrying the last price forward" if d["delta_vs_naive_points"] > -0.1 else
                        "worse than carrying the last price forward")
        res[str(h)] = d
        log(f"  h={h}: iters {bi}, test model {a['accuracy_pct']} last-price {n['accuracy_pct']} route-median {bb['accuracy_pct']} ({d['verdict']}), rows {a['rows']:,}")
        Xc = X.iloc[:3000].values
        d2 = export_trees(m); d2["roots"] = d2["roots"][:bi]
        chk_all[str(h)] = float(np.abs(U._Trees(d2).predict(Xc) - m.predict(Xc, num_iteration=bi)).max())
        del dtr, dva, Xtr, Xva, X
    json.dump({"generated": time.strftime("%Y-%m-%d %H:%M"), "type": "one LightGBM per horizon (1,2,3,5,7,10,14 days ahead); other horizons are interpolated; test window = target dates 2026-09-01..09-30",
               "horizons_trained": HORIZONS, "best_iterations": best, "numpy_vs_lightgbm_max_abs_diff_log": chk_all, "by_horizon": res,
               "summary": [{"horizon_days": int(h), "model_accuracy_pct": v["model"]["accuracy_pct"], "last_price_accuracy_pct": v["naive_last_price"]["accuracy_pct"],
                            "route_days_out_median_accuracy_pct": v["naive_route_days_out_median"]["accuracy_pct"], "delta_points": v["delta_vs_naive_points"],
                            "within_10pct_model": v["model"]["within_10pct"], "within_10pct_last_price": v["naive_last_price"]["within_10pct"],
                            "verdict": v["verdict"]} for h, v in res.items()],
               "note": "One seed, 10% flight sample. Rows: flights quoted both on the as-of day and on the target day. Fuel and route-market features are held at their as-of values for every horizon. Differences under about 0.3 points are not proven."},
              open(ROOT / "models/ts_horizon_metrics.json", "w"), indent=1)
    # ---- final models: refit on all target dates up to the last search day with the validated number of rounds
    tb = {}
    for h in HORIZONS:
        r = horizon_rows(base, h, T, None, {"all": (0, 999, 0.6)}, 3)["all"]
        rounds = max(10, int(best[h] * 1.15))
        mf = lgb.train(p, lgb.Dataset(r[0].values, r[1], feature_name=U.FEATURES), rounds)
        tb[h] = export_trees(mf)
        Xc = r[0].iloc[:3000].values
        chk = float(np.abs(U._Trees(tb[h]).predict(Xc) - mf.predict(Xc)).max())
        log(f"  final h={h}: {len(r[1]):,} rows, {rounds} rounds, numpy-vs-lightgbm diff {chk}")
    import joblib
    old = ROOT / "models/fare_model_ts.joblib"
    if old.exists() and not (ROOT / "models/fare_model_ts_nextday_v1.joblib").exists():
        old.rename(ROOT / "models/fare_model_ts_nextday_v1.joblib")      # my own earlier next-day model, kept
    air_names = dict(zip(air.iata, air.city))
    con_rt = pd.read_parquet(EXT / "route_names.parquet")
    meta = {"listed_as_of": "2026-09-30", "dep_min": None, "dep_max": None, "n_quotes_dataset": 38584829, "n_flights_dataset": 988708,
            "search_date_range": "2026-03-12 to 2026-09-30", "horizons_trained": HORIZONS, "rounds": {h: int(max(10, best[h] * 1.15)) for h in HORIZONS},
            "trained_on": "all target dates up to 2026-09-30, 10% flight sample (98,585 flights), one model per horizon",
            "sources": json.loads((EXT / "SOURCES.json").read_text())}
    joblib.dump({"trees_by_h": tb, "features": U.FEATURES, "families": U.FAMILIES, **T, "route_names": dict(zip(con_rt.route_id.astype(int), con_rt.route)),
                 "airports": air_names, "meta": meta, "lightgbm_params": PARAMS}, ROOT / "models/fare_model_ts.joblib", compress=3)
    log("model saved", round((ROOT / "models/fare_model_ts.joblib").stat().st_size / 1e6, 2), "MB")


def export_trees_check(m, best, Xc):
    # truncated model equivalent: compare against the first `best` trees
    d = export_trees(m)
    k = np.searchsorted(np.arange(len(d["roots"])), best)
    d2 = dict(d); d2["roots"] = d["roots"][:best]
    return float(np.abs(U._Trees(d2).predict(Xc) - m.predict(Xc, num_iteration=best)).max())


def stage_base():
    import duckdb
    con = duckdb.connect(str(DB))
    rt = con.sql("select route_id, route, origin, destination from rt").df()
    rt.to_parquet(EXT / "route_names.parquet")
    con.close()
    df = build_base("fid % 1000 < 100", "sample")
    df.to_parquet(SAMPLE)
    log("sample base", len(df))


def stage_snapshot():
    import duckdb
    df = build_base(f"fid in (select fid from q where sdn = {LAST_SDN})", "snap")
    df = df[df.sdn == LAST_SDN].reset_index(drop=True)
    con = duckdb.connect()
    arr = con.sql(f"""select route, airline, (departure_date - DATE '2026-01-01')::INT as ddn,
        (split_part(departure_time, ':', 1)::INT * 60 + split_part(departure_time, ':', 2)::INT) as dep_min,
        any_value(departure_time) as dep_time, arg_min(arrival_time, price_usd) as arr_time, arg_min(arrival_day_offset, price_usd) as arr_day_offset
        from '{RAW / 'flight_prices.parquet'}' where search_date = DATE '2026-09-30' group by all""").df()
    rt = pd.read_parquet(EXT / "route_names.parquet")
    con2 = duckdb.connect(str(DB), read_only=True)
    air = con2.sql("select airline_id, airline from air").df(); con2.close()
    df = df.merge(rt[["route_id", "route"]], on="route_id").merge(air, on="airline_id")
    df = df.merge(arr, on=["route", "airline", "ddn", "dep_min"], how="left")
    df["id"] = df.fid.map(lambda v: format(int(v), "016x"))
    df["dep_date"] = (U.EPOCH + pd.to_timedelta(df.ddn, unit="D")).dt.strftime("%Y-%m-%d")
    df = df.drop(columns=["fid", "route"]).rename(columns={})
    df["dep_time"] = df.dep_time.fillna(df.dep_min.map(lambda m: f"{m // 60:02d}:{m % 60:02d}"))
    for c in ("dbd", "stops", "dur", "n_changes", "n_obs", "days_since_change"):
        df[c] = df[c].astype("int32")
    df.to_parquet(SNAP, compression="zstd", index=False)
    import joblib
    M = joblib.load(ROOT / "models/fare_model_ts.joblib")
    M["meta"]["dep_min"], M["meta"]["dep_max"] = df.dep_date.min(), df.dep_date.max()
    joblib.dump(M, ROOT / "models/fare_model_ts.joblib", compress=3)
    log("snapshot rows", len(df), "size MB", round(SNAP.stat().st_size / 1e6, 1), "dep", df.dep_date.min(), df.dep_date.max())


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "all"
    if st in ("base", "all"): stage_base()
    if st in ("train", "all"): stage_train()
    if st in ("snapshot", "all"): stage_snapshot()
