"""US fare service for the FareSight demo (backend only; the Flask app wires the route).

    from us_service import us_meta, us_search, us_curve

Depends on numpy, pandas, pyarrow and joblib only (the LightGBM trees are stored as plain arrays and evaluated with
numpy, so no lightgbm / libomp is needed at serving time). Files (override with env US_SNAPSHOT / US_MODEL):
    data/external_us/snapshot.parquet   one row per flight seen on the last search day (2026-09-30), features as of that day
    models/fare_model_ts.joblib         multi-horizon model (trees as arrays) + calendar / fuel / climatology tables
Optional (read for us_meta only): models/ts_metrics.json, models/ts_horizon_metrics.json.

HONEST NOTE: this model barely improves on "the price stays the same" (see docs/TIMESERIES_REPORT.md). predicted_price
will usually be within a few percent of listed_price and may_drop flags are rare.
"""
import os
import json
import time
import threading
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAP_PATH = Path(os.environ.get("US_SNAPSHOT", ROOT / "data/external_us/snapshot.parquet"))
MODEL_PATH = Path(os.environ.get("US_MODEL", ROOT / "models/fare_model_ts.joblib"))
METRICS_PATH = ROOT / "models/ts_metrics.json"
HORIZON_PATH = ROOT / "models/ts_horizon_metrics.json"
EPOCH = pd.Timestamp("2026-01-01")
MAX_AHEAD = 14
MIN_DBD = 7          # the data never shows a quote closer than 7 days to departure: the model is not asked to extrapolate

DEFAULTS = {"predict_ahead_days": 1, "flag_threshold_pct": 10, "big_save_threshold_pct": 20, "big_save_min_usd": 20}

# ------------------------------------------------------------------ feature definition (shared with training)
BASICS = ["dbd", "search_dow", "dep_dow", "dep_hour", "dep_month", "airline_id", "route_id", "stops", "dur", "distance_mi", "ahead"]
HISTORY = ["last_price", "p_1d", "p_3d", "p_7d", "min3", "mean3", "std3", "min7", "mean7", "std7", "run_min", "run_max",
           "days_since_change", "n_changes", "n_obs", "lr_p1", "lr_p3", "lr_p7", "lr_min3", "lr_mean7", "lr_runmin", "lr_runmax"]
ROUTE_MKT = ["lr_route", "route_med7", "airline_rel7", "airline_rank7", "n_airlines", "n_flights_day", "day_coverage"]
CAL_COLS = ["is_holiday", "days_to_hol", "days_since_hol", "hol_weekend", "days_to_easter", "school_summer",
            "school_spring_break", "school_winter", "thanksgiving_window", "school_break_any", "is_weekend"]
CALENDAR = [f"{t}_{c}" for t in "sd" for c in CAL_COLS]
FUEL = ["fx_jet", "fx_brent", "fx_jet_chg7", "fx_jet_chg30", "fx_brent_chg7", "fx_brent_chg30"]
WX = ["tmax", "tmin", "precip", "snow", "wind"]
WEATHER = [f"wxl_{s}_{v}" for s in "od" for v in WX]
FAMILIES = {"basics": BASICS, "history": HISTORY, "route_market": ROUTE_MKT, "calendar": CALENDAR, "fuel": FUEL, "weather": WEATHER}
FEATURES = sum(FAMILIES.values(), [])
FAMILY_TITLES = {"basics": "Basics", "history": "Price history of this flight", "route_market": "Route market",
                 "calendar": "Holidays and calendar", "fuel": "Fuel prices", "weather": "Weather (climatology)"}
FAMILY_DESC = {
    "basics": "Days before departure on the day being predicted, weekdays, departure hour and month, airline, route, stops, duration, distance, and how many days ahead the prediction is.",
    "history": "The same flight's own earlier quotes: last price, price 1/3/7 days earlier, min/mean/std over the last 3 and 7 quotes, running min/max, days since last change, number of changes, and ratios of the last price to those.",
    "route_market": "Route median price over the last 7 search days, this airline's relative price level and rank on the route, number of competing airlines and flights that day, how complete that day's data was.",
    "calendar": "US federal holidays, days to the next / since the last holiday, holiday weekends, days to Easter, school-break windows (hand-set rules of thumb), for both the search date and the departure date.",
    "fuel": "US Gulf Coast jet fuel and Brent crude on the as-of date with 7 and 30-day changes (FRED / EIA public data).",
    "weather": "Typical (climatological, average of 2016-2025) tmax, tmin, rain, snow and wind at the origin and destination airports on the departure date (Open-Meteo / ERA5; non-commercial use only).",
}

_LABEL = {
    "dbd": ("Days before departure (on the predicted day)", "d"), "search_dow": ("Weekday of the predicted day (0=Mon)", "i"),
    "dep_dow": ("Departure weekday (0=Mon)", "i"), "dep_hour": ("Departure hour", "h"), "dep_month": ("Departure month", "i"),
    "airline_id": ("Airline code (internal)", "i"), "route_id": ("Route code (internal)", "i"), "stops": ("Stops", "i"),
    "dur": ("Duration (min)", "i"), "distance_mi": ("Route distance (miles)", "i"), "ahead": ("Prediction horizon (days ahead)", "i"),
    "last_price": ("Last observed price", "$"), "p_1d": ("Price about 1 day earlier", "$"), "p_3d": ("Price about 3 days earlier", "$"),
    "p_7d": ("Price about 7 days earlier", "$"), "min3": ("Lowest of the last 3 quotes", "$"), "mean3": ("Mean of the last 3 quotes", "$"),
    "std3": ("Std dev of the last 3 quotes", "$"), "min7": ("Lowest of the last 7 quotes", "$"), "mean7": ("Mean of the last 7 quotes", "$"),
    "std7": ("Std dev of the last 7 quotes", "$"), "run_min": ("Lowest price seen so far", "$"), "run_max": ("Highest price seen so far", "$"),
    "days_since_change": ("Days since the price last changed", "i"), "n_changes": ("Price changes so far", "i"),
    "n_obs": ("Days this flight has been quoted", "i"), "lr_p1": ("Log ratio last / 1 day earlier", "f"),
    "lr_p3": ("Log ratio last / 3 days earlier", "f"), "lr_p7": ("Log ratio last / 7 days earlier", "f"),
    "lr_min3": ("Log ratio last / lowest of last 3", "f"), "lr_mean7": ("Log ratio last / mean of last 7", "f"),
    "lr_runmin": ("Log ratio last / lowest so far", "f"), "lr_runmax": ("Log ratio last / highest so far", "f"),
    "lr_route": ("Log ratio last / route 7-day median", "f"), "route_med7": ("Route median price, last 7 search days", "$"),
    "airline_rel7": ("This airline's price vs route median, last 7 days", "f"), "airline_rank7": ("This airline's price rank on route (0 cheap - 1 dear)", "f"),
    "n_airlines": ("Airlines on this route and date", "i"), "n_flights_day": ("Flights on this route and date", "i"),
    "day_coverage": ("Completeness of the as-of day's data (1 = normal)", "f"),
    "fx_jet": ("Jet fuel, US Gulf Coast ($/gal)", "f"), "fx_brent": ("Brent crude ($/bbl)", "f"),
    "fx_jet_chg7": ("Jet fuel, 7-day change", "p"), "fx_jet_chg30": ("Jet fuel, 30-day change", "p"),
    "fx_brent_chg7": ("Brent, 7-day change", "p"), "fx_brent_chg30": ("Brent, 30-day change", "p"),
}
_CAL_LABEL = {"is_holiday": "federal holiday", "days_to_hol": "days to next holiday", "days_since_hol": "days since last holiday",
              "hol_weekend": "holiday weekend", "days_to_easter": "days from Easter", "school_summer": "summer break",
              "school_spring_break": "spring break", "school_winter": "winter break", "thanksgiving_window": "Thanksgiving window",
              "school_break_any": "any school break", "is_weekend": "weekend"}
for _t, _n in (("s", "predicted day"), ("d", "departure date")):
    for _c in CAL_COLS:
        _LABEL[f"{_t}_{_c}"] = (f"{_CAL_LABEL[_c].capitalize()} ({_n})", "i")
_WXL = {"tmax": "Typical max temp (C)", "tmin": "Typical min temp (C)", "precip": "Typical rain (mm)", "snow": "Typical snow (cm)", "wind": "Typical max wind (km/h)"}
for _s, _n in (("o", "origin"), ("d", "destination")):
    for _v in WX:
        _LABEL[f"wxl_{_s}_{_v}"] = (f"{_WXL[_v]}, {_n}", "f")


def _fmt(v, kind):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    v = float(v)
    if kind == "$":
        return round(v, 2)
    if kind == "p":
        return round(v * 100, 1)
    if kind == "f":
        return round(v, 3)
    return int(round(v)) if kind in ("i", "d") else round(v, 2)


def make_features(base, h, T):
    """base: DataFrame of as-of rows (snapshot columns); h: int or array of days ahead; T: tables from the model file.
    Returns float32 DataFrame with FEATURES columns. Used identically in training and serving."""
    n = len(base)
    h = np.broadcast_to(np.asarray(h, dtype=np.int64), (n,))
    sdn = base["sdn"].values.astype(np.int64); ddn = base["ddn"].values.astype(np.int64)
    X = {}
    X["dbd"] = base["dbd"].values.astype(np.float32) - h
    cal = T["cal"]; dn0 = T["cal_dn0"]
    tgt = sdn + h
    X["search_dow"] = cal["dow"][tgt - dn0].astype(np.float32)
    X["dep_dow"] = cal["dow"][ddn - dn0].astype(np.float32)
    X["dep_month"] = cal["month"][ddn - dn0].astype(np.float32)
    X["dep_hour"] = (base["dep_min"].values / 60.0).astype(np.float32)
    for c in ("airline_id", "route_id", "stops", "dur", "distance_mi"):
        X[c] = base[c].values.astype(np.float32)
    X["ahead"] = h.astype(np.float32)
    lp = base["price"].values.astype(np.float64)
    X["last_price"] = lp.astype(np.float32)
    for c in ("p_1d", "p_3d", "p_7d", "min3", "mean3", "std3", "min7", "mean7", "std7", "run_min", "run_max",
              "days_since_change", "n_changes", "n_obs"):
        X[c] = base[c].values.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        for nm, col in (("lr_p1", "p_1d"), ("lr_p3", "p_3d"), ("lr_p7", "p_7d"), ("lr_min3", "min3"), ("lr_mean7", "mean7"),
                        ("lr_runmin", "run_min"), ("lr_runmax", "run_max"), ("lr_route", "route_med7")):
            X[nm] = np.log(lp / base[col].values.astype(np.float64)).astype(np.float32)
    for c in ("route_med7", "airline_rel7", "airline_rank7", "n_airlines", "n_flights_day", "day_coverage"):
        X[c] = base[c].values.astype(np.float32)
    for j, c in enumerate(CAL_COLS):
        X[f"s_{c}"] = cal["vals"][tgt - dn0, j]
        X[f"d_{c}"] = cal["vals"][ddn - dn0, j]
    fx = T["fuel"]; fi = sdn - T["fuel_dn0"]
    for j, c in enumerate(FUEL):
        X[c] = fx[fi, j].astype(np.float32)
    doy = cal["doy"][ddn - dn0] - 1
    wcl = T["wx_clim"]                     # (airports, 365, 5)
    for side, idx in (("o", base["o_idx"].values.astype(int)), ("d", base["d_idx"].values.astype(int))):
        for j, v in enumerate(WX):
            X[f"wxl_{side}_{v}"] = wcl[idx, doy, j].astype(np.float32)
    return pd.DataFrame({c: np.asarray(X[c], dtype=np.float32) for c in FEATURES})


# ------------------------------------------------------------------ numpy tree ensemble
class _Trees:
    def __init__(self, d):
        self.feat, self.thr, self.left, self.right = d["feat"], d["thr"], d["left"], d["right"]
        self.dl, self.val, self.roots, self.depth = d["default_left"], d["value"], d["roots"], int(d["max_depth"])

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        idx = np.repeat(self.roots[:, None], n, axis=1)
        cols = np.arange(n)[None, :]
        for _ in range(self.depth + 1):
            f = self.feat[idx]
            x = X[cols, f]
            go_left = np.where(np.isnan(x), self.dl[idx], x <= self.thr[idx])
            idx = np.where(go_left, self.left[idx], self.right[idx])
        return self.val[idx].sum(axis=0)


# ------------------------------------------------------------------ lazy state
_lock = threading.Lock()
_S = None


def _load():
    global _S
    if _S is not None:
        return _S
    with _lock:
        if _S is not None:
            return _S
        import joblib
        M = joblib.load(MODEL_PATH)
        snap = pd.read_parquet(SNAP_PATH).reset_index(drop=True)
        key = snap.route_id.values.astype(np.int64) * 100000 + snap.ddn.values.astype(np.int64)
        order = np.argsort(key, kind="stable")
        ks = key[order]
        uk, start = np.unique(ks, return_index=True)
        end = np.append(start[1:], len(ks))
        index = {int(k): order[a:b] for k, a, b in zip(uk, start, end)}
        id_idx = {i: j for j, i in enumerate(snap["id"].values)}
        routes = {n: int(i) for i, n in M["route_names"].items()}
        _S = {"M": M, "snap": snap, "index": index, "id_idx": id_idx, "routes": routes, "trees": {int(h): _Trees(t) for h, t in M["trees_by_h"].items()},
              "T": {k: M[k] for k in ("cal", "cal_dn0", "fuel", "fuel_dn0", "wx_clim")}}
        return _S


def _cfg(config):
    c = dict(DEFAULTS)
    for k, v in (config or {}).items():
        if k in c and v is not None:
            c[k] = v
    c["predict_ahead_days"] = int(min(MAX_AHEAD, max(1, int(c["predict_ahead_days"]))))
    for k in ("flag_threshold_pct", "big_save_threshold_pct", "big_save_min_usd"):
        c[k] = float(c[k])
    return c


def _predict_curve(S, rows):
    """Predicted price for days_ahead 1..14 for every row (NaN where the target day would be < 7 days before departure).
    One model per trained horizon; other horizons are linearly interpolated in log-ratio. Returns (prices, features)."""
    n = len(rows)
    hs = np.arange(1, MAX_AHEAD + 1)
    trained = np.array(sorted(S["trees"]))
    # features for every (row, horizon 1..14); trees only need the trained horizons
    rep = rows.iloc[np.repeat(np.arange(n), len(hs))].reset_index(drop=True)
    hh = np.tile(hs, n)
    X = make_features(rep, hh, S["T"])
    ratio = np.zeros((n, len(trained)))
    for j, h in enumerate(trained):
        sel = np.arange(n) * len(hs) + (h - 1)
        ratio[:, j] = S["trees"][int(h)].predict(X.values[sel])          # log(price_{s+h} / price_s)
    full = np.vstack([np.interp(hs, trained, ratio[i]) for i in range(n)])
    price = rows["price"].values.astype(np.float64)[:, None] * np.exp(full)
    ok = X["dbd"].values.reshape(n, len(hs)) >= MIN_DBD
    return np.where(ok, price, np.nan), X


def _flight_dict(S, i, row, curve_row, Xrows, cfg):
    listed = float(row["price"])
    h = cfg["predict_ahead_days"]
    valid = ~np.isnan(curve_row)
    h_eff = min(h, int(valid.nonzero()[0].max()) + 1) if valid.any() else 0
    pred = float(curve_row[h_eff - 1]) if h_eff else listed
    # drop rule (mirrors the India side): lowest predicted price at or after the horizon, within the curve
    pts = [(d + 1, float(curve_row[d])) for d in range(h_eff - 1, MAX_AHEAD) if valid[d]] if h_eff else []
    drop_pct, best_d, best_p = 0.0, None, None
    if pts and listed > 0:
        d, p = min(pts, key=lambda t: (t[1], t[0]))
        drop_pct = max(0.0, round((listed - p) / listed * 100, 1)); best_d, best_p = d, p
    may = bool(drop_pct >= cfg["flag_threshold_pct"])
    save = round(max(0.0, listed - best_p), 2) if may else 0
    big = bool(may and (drop_pct >= cfg["big_save_threshold_pct"] or (cfg["big_save_min_usd"] > 0 and save >= cfg["big_save_min_usd"])))
    dbd0 = int(row["dbd"])
    xr = Xrows.iloc[(h_eff or 1) - 1]
    inputs = {}
    for fam, cols in FAMILIES.items():
        inputs[fam] = {"title": FAMILY_TITLES[fam], "items": [
            {"key": c, "label": _LABEL[c][0], "value": _fmt(xr[c], _LABEL[c][1])} for c in cols]}
    arr = row["arr_time"] if isinstance(row["arr_time"], str) else None
    return {
        "id": row["id"], "airline": row["airline"], "flight_code": f"{row['airline']} {row['dep_time']}",
        "source": row["origin"], "destination": row["destination"], "dep_date": row["dep_date"],
        "dep_time": row["dep_time"], "arr_time": arr, "arr_day_offset": int(row["arr_day_offset"]) if arr else None,
        "stops": int(row["stops"]), "duration_h": round(float(row["dur"]) / 60, 2), "cabin": None, "currency": "USD",
        "listed_price": round(listed, 2), "listed_as_of": S["M"]["meta"]["listed_as_of"], "days_left_at_listing": dbd0,
        "predicted_price": round(pred, 2), "predicted_horizon_days": h_eff, "horizon_requested_days": h,
        "change_pct": round((pred - listed) / listed * 100, 1) if listed > 0 else 0.0,
        "may_drop": may, "drop_pct": drop_pct if may else 0, "drop_best_days_left": (dbd0 - best_d) if may else None,
        "drop_best_days_ahead": best_d if may else None, "drop_price": round(best_p, 2) if may else None,
        "big_save": big, "save_usd": save,
        "curve": [{"days_ahead": d + 1, "days_left": dbd0 - (d + 1), "price": round(float(curve_row[d]), 2)}
                  for d in range(MAX_AHEAD) if valid[d]],
        "inputs": inputs,
    }


def us_search(origin, destination, date, cabin=None, config=None):
    """Flights for one origin-destination on one departure date. Out-of-range dates give an empty list, not an error."""
    S = _load(); cfg = _cfg(config)
    o, d = str(origin or "").upper().strip(), str(destination or "").upper().strip()
    base = {"ok": True, "currency": "USD", "listed_as_of": S["M"]["meta"]["listed_as_of"], "config": cfg, "flights": [], "count": 0,
            "cabin": None, "note": None}
    rid = S["routes"].get(f"{o}-{d}")
    if rid is None:
        return {**base, "ok": False, "error": f"no data for route {o}-{d}"}
    try:
        dt = pd.Timestamp(date)
    except Exception:
        return {**base, "ok": False, "error": f"bad date {date!r} (use YYYY-MM-DD)"}
    ddn = (dt.normalize() - EPOCH).days
    idx = S["index"].get(rid * 100000 + ddn)
    if idx is None or len(idx) == 0:
        m = S["M"]["meta"]
        base["note"] = f"no flights for {dt.date()}: the snapshot covers departures {m['dep_min']} to {m['dep_max']} (listed as of {m['listed_as_of']})"
        return base
    rows = S["snap"].iloc[idx].reset_index(drop=True)
    curve, X = _predict_curve(S, rows)
    flights = []
    for i in range(len(rows)):
        flights.append(_flight_dict(S, i, rows.iloc[i], curve[i], X.iloc[i * MAX_AHEAD:(i + 1) * MAX_AHEAD].reset_index(drop=True), cfg))
    flights.sort(key=lambda f: (f["dep_time"], f["airline"]))
    base.update(flights=flights, count=len(flights))
    return base


def us_curve(flight_id, config=None):
    """Curve (and the same fields as in a search result) for one flight id returned by us_search."""
    S = _load(); cfg = _cfg(config)
    j = S["id_idx"].get(flight_id)
    if j is None:
        return None
    rows = S["snap"].iloc[[j]].reset_index(drop=True)
    curve, X = _predict_curve(S, rows)
    return _flight_dict(S, 0, rows.iloc[0], curve[0], X, cfg)


def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def us_meta():
    S = _load(); M = S["M"]; m = M["meta"]; snap = S["snap"]
    pairs = sorted(S["routes"])
    origins = sorted({p.split("-")[0] for p in pairs}); dests = sorted({p.split("-")[1] for p in pairs})
    air = M["airports"]
    fam = [{"key": k, "title": FAMILY_TITLES[k], "columns": len(v), "description": FAMILY_DESC[k]} for k, v in FAMILIES.items()]
    tm = _json(METRICS_PATH); hm = _json(HORIZON_PATH)
    head = None
    if tm:
        try:
            r = tm["results"]["lgb_T"]
            head = {"split": "T (train to 2026-08-15, test September 2026), next search day",
                    "model_accuracy_pct": r["model_rows_with_history"]["accuracy_pct"],
                    "naive_last_price_accuracy_pct": r["naive_last_price_rows_with_history"]["accuracy_pct"],
                    "naive_route_days_out_median_accuracy_pct": r["naive_route_dbd_median_all_rows"]["accuracy_pct"],
                    "direction_given_a_move": tm.get("direction_classifier_T", {}).get("hit_rate_pct"),
                    "share_next_prices_unchanged_pct": r["direction_vs_last_observation"]["share_actually_unchanged"],
                    "feature_families": "history matters; route-market, calendar, fuel and weather add 0.2 points or less"}
        except Exception:
            head = None
    return {
        "currency": "USD", "listed_as_of": m["listed_as_of"], "origins": origins, "destinations": dests, "pairs": pairs,
        "airports": {a: air[a] for a in sorted(air) if a in origins or a in dests},
        "departure_date_range": {"min": m["dep_min"], "max": m["dep_max"]},
        "counts": {"flights_in_snapshot": int(len(snap)), "routes": len(pairs), "airlines": int(snap.airline.nunique()),
                   "quotes_in_dataset": m["n_quotes_dataset"], "flights_in_dataset": m["n_flights_dataset"],
                   "search_dates": m["search_date_range"]},
        "feature_families": fam, "feature_columns_total": len(FEATURES),
        "horizon": {"default_days": DEFAULTS["predict_ahead_days"], "min": 1, "max": MAX_AHEAD, "trained_horizons": sorted(S["trees"]),
                    "accuracy": (hm or {}).get("summary"), "note": (hm or {}).get("note")},
        "headline_metrics": head,
        "defaults": DEFAULTS,
        "honest_note": "The model barely improves on 'the price stays the same': predicted_price is usually within a few percent of listed_price and may_drop flags are rare. Cabin is not in the data; listed_price is the cheapest quote for the flight.",
        "sources": M["meta"]["sources"],
        "data_limits": ["US routes only (50), 2026 data; no seat availability; the quote is the cheapest fare found for the flight on that day.",
                        "Weather is a climatology (average of earlier years), never a forecast.",
                        "Fuel and route-market features are held at their as-of values for all horizons.",
                        "Open-Meteo free API data is for non-commercial use only."],
    }
