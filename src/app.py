"""Web app: demo UI (/) and configuration UI (/config) over the fare model."""
import datetime as dt
import json
import math
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, jsonify, render_template, request

from features import add_basic, COORDS

ROOT = Path(__file__).resolve().parents[1]
app = Flask(__name__, template_folder=str(ROOT / "templates"))

bundle = joblib.load(ROOT / "models/fare_model.joblib")
MODEL, ENC = bundle["model"], bundle["enc"]

DF = pd.read_parquet(ROOT / "data/merged.parquet")
DF = DF[DF["class"] != "First"]

# One itinerary per (route, flight code, class): the most common stops/times/duration.
CATALOG = (DF.groupby(["source", "destination", "class", "flight_code", "airline"])
             .agg(stops=("stops", lambda s: s.mode().iat[0]),
                  dep_bucket=("dep_bucket", lambda s: s.mode().iat[0]),
                  arr_bucket=("arr_bucket", lambda s: s.mode().iat[0]),
                  duration_h=("duration_h", "median"),
                  dep_hour=("dep_hour", "median"), arr_hour=("arr_hour", "median"),
                  n=("price", "size"))
             .reset_index())
CITIES = sorted(set(DF.source) | set(DF.destination))
CLASSES = ["Economy", "Premium Economy", "Business"]
MAX_DAYS = int(DF.days_left.max())
PAGE_SIZE = 10  # fixed: the demo shows 10 flights per page

# ---------------------------------------------------------------- configuration
CONFIG_PATH = ROOT / "config.json"
DEFAULTS = {
    "predict_at_days_left": 1,       # the predicted price is the model's estimate this many days before departure
    "price_period": 2023,            # price level the model predicts at (2022 or 2023 scrape)
    "listed_window_days": 3,         # listed price = median real fare within +/- this many days-left
    "curve_days": [45, 40, 30, 21, 14, 7, 3, 1],
    "flag_threshold_pct": 10,        # changes bigger than this are highlighted
    "default_sort": "listed",
    "show_curve": True,
}
SORTS = ["listed", "predicted", "change_asc", "change_desc", "airline"]


def load_config():
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text()))
        except (ValueError, OSError):
            pass
    return cfg


def validate_config(raw):
    """Return (clean_config, errors)."""
    cfg, errs = dict(DEFAULTS), []

    def intval(key, lo, hi):
        try:
            v = int(raw.get(key, DEFAULTS[key]))
        except (TypeError, ValueError):
            errs.append(f"{key} must be a whole number"); return
        if not lo <= v <= hi:
            errs.append(f"{key} must be between {lo} and {hi}"); return
        cfg[key] = v

    intval("predict_at_days_left", 1, 45)
    intval("listed_window_days", 0, 10)
    intval("flag_threshold_pct", 1, 100)
    if raw.get("price_period", DEFAULTS["price_period"]) in (2022, 2023):
        cfg["price_period"] = int(raw.get("price_period", DEFAULTS["price_period"]))
    else:
        errs.append("price_period must be 2022 or 2023")
    cd = raw.get("curve_days", DEFAULTS["curve_days"])
    try:
        cd = sorted({int(x) for x in cd}, reverse=True)
        if not 3 <= len(cd) <= 12 or cd[0] > 50 or cd[-1] < 1:
            raise ValueError
        cfg["curve_days"] = cd
    except (TypeError, ValueError):
        errs.append("curve_days must be 3 to 12 whole numbers between 1 and 50")
    if raw.get("default_sort", "listed") in SORTS:
        cfg["default_sort"] = raw.get("default_sort", "listed")
    else:
        errs.append(f"default_sort must be one of {SORTS}")
    cfg["show_curve"] = bool(raw.get("show_curve", True))
    return cfg, errs


# ---------------------------------------------------------------- prediction
def predict(rows: pd.DataFrame, days_left: int, travel: dt.date, period: int):
    X = rows.copy()
    X["days_left"] = days_left
    X["dow"] = travel.weekday()
    X["month"] = travel.month
    X["period"] = period
    feats = ENC.transform(add_basic(X))
    return np.exp(MODEL.predict(feats)), feats


def listed_price(route_df: pd.DataFrame, code: str, days_left: int, window: int) -> float:
    """Real observed fare for this flight near `days_left` (what a provider listed in the scrape)."""
    m = route_df[route_df.flight_code == code]
    near = m[(m.days_left - days_left).abs() <= window]
    return float((near if len(near) >= 3 else m).price.median())


def clean(v):
    """JSON-safe scalar (NaN -> None, numpy -> python)."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        if math.isnan(v):
            return None
        return int(v) if float(v).is_integer() else round(float(v), 2)
    return v


@app.get("/")
def demo():
    return render_template("index.html", cities=CITIES, classes=CLASSES, page="demo")


@app.get("/config")
def config_page():
    return render_template("config.html", page="config")


@app.post("/api/search")
def search():
    q = request.get_json(force=True)
    cfg = load_config()
    src, dst, cls = q.get("source"), q.get("destination"), q.get("class", "Economy")
    if src == dst or src not in CITIES or dst not in CITIES or cls not in CLASSES:
        return jsonify(error="Pick two different supported cities and a valid class."), 400
    try:
        travel = dt.date.fromisoformat(q["date"])
    except (KeyError, ValueError):
        return jsonify(error="Invalid date."), 400
    days_out = (travel - dt.date.today()).days
    if days_out < 1:
        return jsonify(error="Travel date must be in the future."), 400
    days_left = min(days_out, MAX_DAYS)  # the model has only seen up to MAX_DAYS ahead

    rows = CATALOG[(CATALOG.source == src) & (CATALOG.destination == dst) & (CATALOG["class"] == cls)]
    if rows.empty:
        return jsonify(error=f"No {cls} flights for {src} to {dst} in the training data."), 404

    period = cfg["price_period"]
    at_dep, feats = predict(rows, cfg["predict_at_days_left"], travel, period)
    curves = {d: predict(rows, d, travel, period)[0] for d in cfg["curve_days"]}
    route_df = DF[(DF.source == src) & (DF.destination == dst) & (DF["class"] == cls)]
    flights = []
    for i, (_, r) in enumerate(rows.iterrows()):
        f = feats.iloc[i]
        flights.append({
            "airline": r.airline, "code": r.flight_code, "stops": int(r.stops),
            "dep": r.dep_bucket, "arr": r.arr_bucket, "duration_h": round(float(r.duration_h), 1),
            "listed": round(listed_price(route_df, r.flight_code, days_left, cfg["listed_window_days"])),
            "predicted": round(float(at_dep[i])),
            "curve": [{"d": d, "p": round(float(curves[d][i]))} for d in cfg["curve_days"]],
            "inputs": {  # the exact feature values the model received for the departure prediction
                "airline": r.airline, "route": f"{src} to {dst}", "class": cls,
                "stops": clean(f.stops), "duration_h": clean(f.duration_h),
                "dep_hour": clean(f.dep_hour), "arr_hour": clean(f.arr_hour),
                "dep_bucket": r.dep_bucket, "arr_bucket": r.arr_bucket,
                "distance_km": clean(f.distance_km), "days_left": cfg["predict_at_days_left"],
                "weekday": travel.strftime("%A"), "month": travel.month, "period": period,
                "route_supply": clean(f.route_supply),
                "flight_price_level_inr": clean(math.exp(f.te_flight)) if not math.isnan(f.te_flight) else None,
                "route_airline_price_level_inr": clean(math.exp(f.te_route_airline)) if not math.isnan(f.te_route_airline) else None,
            },
        })
    return jsonify(source=src, destination=dst, date=travel.isoformat(), days_out=days_out,
                   clamped=days_out > MAX_DAYS, model_days=days_left, page_size=PAGE_SIZE,
                   config=cfg, flights=flights)


@app.route("/api/config", methods=["GET", "POST"])
def config_api():
    if request.method == "GET":
        return jsonify(config=load_config(), defaults=DEFAULTS, sorts=SORTS)
    raw = request.get_json(force=True)
    if raw.get("reset"):
        CONFIG_PATH.unlink(missing_ok=True)
        return jsonify(config=dict(DEFAULTS), errors=[])
    cfg, errs = validate_config(raw)
    if errs:
        return jsonify(errors=errs), 400
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    return jsonify(config=cfg, errors=[])


def _load_json(name):
    p = ROOT / "models" / name
    return json.loads(p.read_text()) if p.exists() else None


SOURCES = [
    ("Airfare ML (EaseMyTrip scrape, Jan-Mar 2023)", "yashdharme36/airfare-ml-predicting-flight-fares", "Main training data", True),
    ("Airlines Flights Data (EaseMyTrip scrape, 2022)", "rohitgrewal/airlines-flights-data", "Second period (no dates; scrape date assumed 2022-02-11)", True),
    ("Flight Price Dataset (2019)", "anshuman0427/flight-price-dataset", "Not used: no stops, duration or class", False),
    ("MakeMyTrip sample (Aug 2023)", "andrewgeorgeissac/flight-price-data-of-indian-cities-makemytrip", "Not used: about 400 rows, kept for sanity checks", False),
    ("Indian Domestic Airline schedules", "kabil007/indian-domestic-airline-dataset", "Not used yet: schedules only, no fares", False),
    ("Atlanta international fares (Hugging Face)", "egupta/atl-intl-flight-data-sql-db", "Not used: US routes", False),
]


def data_summary():
    per = []
    for p, g in DF.groupby("period"):
        per.append({"period": int(p), "rows": int(len(g)), "flights": int(g.flight_code.nunique()),
                    "routes": int((g.source + g.destination).nunique()), "airlines": int(g.airline.nunique()),
                    "first_journey": str(g.journey_date.min().date()), "last_journey": str(g.journey_date.max().date()),
                    "has_exact_times": bool(g.dep_hour.notna().any()),
                    "median_price": {c: int(g[g["class"] == c].price.median()) for c in CLASSES if (g["class"] == c).any()}})
    air = DF.groupby(["airline", "period"]).size().unstack(fill_value=0)
    return {"periods": per, "total_rows": int(len(DF)), "cities": CITIES,
            "airlines": [{"airline": a, **{str(int(p)): int(air.loc[a, p]) for p in air.columns}} for a in air.index],
            "sources": [{"name": n, "slug": s, "role": r, "used": u} for n, s, r, u in SOURCES]}


FEATURE_DOCS = {
    "airline": ("categorical", "Operating airline"),
    "source": ("categorical", "Origin city"),
    "destination": ("categorical", "Destination city"),
    "class": ("categorical", "Cabin: Economy, Premium Economy or Business"),
    "dep_bucket": ("categorical", "Departure time of day (4 buckets)"),
    "arr_bucket": ("categorical", "Arrival time of day (4 buckets)"),
    "stops": ("number", "0, 1 or 2+ stops"),
    "duration_h": ("number", "Total journey time in hours"),
    "days_left": ("number", "Days between booking and departure (1 to 50 in the data)"),
    "dow": ("number", "Weekday of travel (0 = Monday)"),
    "month": ("number", "Month of travel"),
    "period": ("number", "Which scrape the price level comes from (2022 or 2023)"),
    "dep_hour": ("number", "Exact departure hour (2023 scrape only; empty for 2022)"),
    "arr_hour": ("number", "Exact arrival hour (2023 scrape only; empty for 2022)"),
    "distance_km": ("number", "Great-circle distance between the two cities"),
    "te_flight": ("target encoding", "Typical log price of this flight code and class, computed out-of-fold"),
    "te_route_airline": ("target encoding", "Typical log price of this airline on this route and class, computed out-of-fold"),
    "route_supply": ("number", "Number of distinct flights on the route and class (competition proxy)"),
}
HP_KEYS = ["loss", "max_iter", "learning_rate", "max_leaf_nodes", "min_samples_leaf", "l2_regularization",
           "early_stopping", "validation_fraction", "n_iter_no_change", "random_state"]


@app.get("/api/meta")
def meta():
    params = MODEL.get_params()
    return jsonify(
        config=load_config(), defaults=DEFAULTS, sorts=SORTS, page_size=PAGE_SIZE, max_days=MAX_DAYS,
        data=data_summary(),
        model={"type": "HistGradientBoostingRegressor (scikit-learn)", "target": "log(price in INR)",
               "rounds_fitted": int(MODEL.n_iter_), "n_features": len(ENC.features),
               "hyperparameters": {k: params[k] for k in HP_KEYS},
               "features": [{"name": f, "kind": FEATURE_DOCS.get(f, ("", ""))[0], "doc": FEATURE_DOCS.get(f, ("", ""))[1]}
                            for f in ENC.features]},
        metrics=_load_json("metrics.json"), experiments=_load_json("experiments.json"),
        importance=_load_json("importance.json"))


if __name__ == "__main__":
    app.run(port=5055, debug=False)
