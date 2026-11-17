"""Web app: demo UI (/) and configuration UI (/config) over the fare model."""
import datetime as dt
import json
import re
import math
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, jsonify, render_template, request

from features import add_basic, COORDS
from drop_logic import future_series, drop_info, big_save_rule

ROOT = Path(__file__).resolve().parents[1]
app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"), static_url_path="/static")
app.json.sort_keys = False   # keep field order (the US feature families are shown in the order the model groups them)

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
_MANGLED = re.compile(r"^(\d)\.0+E-(\d+)$")


def display_code(code):
    """The one place that turns a stored flight code into the code a traveller reads.

    The 2022 scrape has IndiGo codes damaged by a spreadsheet ('6.00E-152' for '6E-152'; 80 codes, 4,705 rows) and
    one code with no recoverable number ('0.00E+00', 51 rows). Display repairs the first shape and returns None for
    the second (the UI then says the number is not available). The stored `code` stays the unique key everywhere;
    this is for display only."""
    c = str(code or "").strip()
    m = _MANGLED.match(c)
    if m:
        return f"{m.group(1)}E-{m.group(2)}"
    if re.match(r"^\d+\.\d+E[+-]\d+$", c):
        return None
    return c or None


def _accuracy():
    try:
        m = json.loads((ROOT / "models/metrics.json").read_text())
        return {k: {"accuracy": v["accuracy_pct"], "within10": v["within_10pct"], "rows": v["rows"]}
                for k, v in zip("AEC", (m["A_same_period_random_20pct"], m["E_2022_plus_10pct_of_2023"], m["C_train_2022_test_2023"]))}
    except Exception:
        return None


ACCURACY = _accuracy()   # the three held-out scores for the deployed model (models/metrics.json)


def accuracy_for(key):
    """The three held-out scores of the engine that made the prices (manifest = models/algorithms.json), else the deployed model's."""
    e = next((e for e in engine_list() if e["key"] == key), None)
    if e and isinstance(e.get("accuracy"), dict) and ACCURACY:
        return {k: {"accuracy": e["accuracy"][k], "within10": (e.get("within10") or {}).get(k, ACCURACY[k]["within10"]), "rows": ACCURACY[k]["rows"]} for k in "AEC" if k in e["accuracy"]}
    return ACCURACY
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
    "include_defunct": False,        # show airlines that no longer operate (historical data only)
    "algorithm": "xgboost",          # prediction engine for the India market (see /api/engines); falls back to hgb if it cannot load
    "flag_threshold_pct": 10,        # changes bigger than this are highlighted
    "big_save_threshold_pct": 20,    # a "may drop" flight is a big save at or above this predicted fall (%)
    "big_save_min_inr": 1500,        # ...or at or above this rupee saving (0 switches the rupee test off)
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
    intval("big_save_threshold_pct", 5, 80)
    intval("big_save_min_inr", 0, 50000)
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
    cfg["include_defunct"] = bool(raw.get("include_defunct", False))
    alg = raw.get("algorithm", DEFAULTS["algorithm"])
    cfg["algorithm"] = alg if alg in engine_keys() else DEFAULTS["algorithm"]   # an unknown key falls back to the default
    return cfg, errs


# ---------------------------------------------------------------- prediction
def predict(rows: pd.DataFrame, days_left: int, travel: dt.date, period: int, engine=None):
    X = rows.copy()
    X["days_left"] = days_left
    X["dow"] = travel.weekday()
    X["month"] = travel.month
    X["period"] = period
    feats = ENC.transform(add_basic(X))
    return np.exp((engine or MODEL).predict(feats)), feats


def predict_many(rows, days, travel, period, engine=None):
    """Predict every requested days-before-departure for all rows in ONE model call (about 8x faster than one call per
    day, because the model has a large fixed cost per call). Returns ({days: prices}, {days: encoded features})."""
    days = list(dict.fromkeys(days)); n = len(rows)
    big = pd.concat([rows.assign(days_left=d, dow=travel.weekday(), month=travel.month, period=period) for d in days], ignore_index=True)
    feats = ENC.transform(add_basic(big))
    out = np.exp((engine or MODEL).predict(feats))
    return ({d: out[i * n:(i + 1) * n] for i, d in enumerate(days)},
            {d: feats.iloc[i * n:(i + 1) * n].reset_index(drop=True) for i, d in enumerate(days)})


# Airlines whose brand no longer sells flights. They stay in the historical data but are hidden unless asked for.
DEFUNCT = {"GO FIRST": "GO FIRST stopped flying in 2023", "Vistara": "Vistara merged into Air India in November 2024",
           "AirAsia": "AirAsia India merged into Air India Express in 2024"}
_SEARCH_CACHE = {}          # model outputs per search, so repeating a search (or changing only thresholds) is instant
_CACHE_MAX = 24


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


def add_drop_fields(f, days_left, p0, cfg):
    """May-drop / big-save flags (see drop_logic.py for the rule).

    days_left: today's days before departure (clamped to the model range); p0: the model's price for today.
    The card's big slot is always FareSight at the chosen horizon (`card_*`); a tagged flight adds the expected low
    (`drop_*`), a second price that is below the listed fare, as is the big slot (the rule requires both).
    """
    pts = [(c["d"], c["p"]) for c in f["curve"]]
    extra = [(cfg["predict_at_days_left"], f["predicted"])]
    series = future_series(days_left, p0, pts, extra)
    di = drop_info(f["listed"], series, cfg["flag_threshold_pct"], cfg["big_save_threshold_pct"], cfg["big_save_min_inr"], headline=f["predicted"])
    f["p0"] = round(p0)
    f["may_drop"] = di["may_drop"]
    f["drop_best_days_left"] = di["drop_best_days_left"]
    f["drop_price"] = round(di["drop_price"]) if di["may_drop"] else None
    f["drop_pair_days_left"] = di["low_pair"]
    # percentages come from the rounded prices that are shown, so the tag, the second line and the API agree
    f["drop_pct"] = round((f["listed"] - f["drop_price"]) / f["listed"] * 100, 1) if di["may_drop"] else 0
    f["save_inr"] = int(f["listed"] - f["drop_price"]) if di["may_drop"] else 0
    f["big_save"] = bool(di["may_drop"] and big_save_rule(f["drop_pct"], f["save_inr"], cfg["big_save_threshold_pct"], cfg["big_save_min_inr"]))
    f["card_price"], f["card_kind"], f["card_days_left"] = f["predicted"], "horizon", cfg["predict_at_days_left"]
    f["card_change_pct"] = round((f["predicted"] - f["listed"]) / f["listed"] * 100, 1) if f["listed"] else 0.0
    f["low_change_pct"] = -f["drop_pct"] if di["may_drop"] else None



# ---------------------------------------------------------------- prediction engines (India only)
# `engines` (src/engines.py) lists and loads the switchable algorithms. The deployed model (key "hgb") is always the
# app's own MODEL, so choosing it never loads a second copy. Development only: FARESIGHT_MOCK=1 falls back to
# src/engines_mock.py when models/engines/manifest.json does not exist yet (flagged "mock": true).
_ENG = {"mod": None, "mock": False, "err": None, "tried": 0.0}
HGB_ENTRY = {"key": "hgb", "label": "First model (HistGradientBoosting)", "family": "Gradient boosting",
             "description": "The first model built for this project: thousands of small decision trees, each fixing the mistakes of the ones before.",
             "available": True, "unavailable_reason": None}
# how each engine is presented. Everything else is a neutral "also evaluated" engine.
TIERS = {"xgboost": ("recommended", "Recommended", None), "hgb": ("reliable", "Reliable", None)}
CAUTION = "Included for comparison. Less accurate on this data; prices may look unusual."
CAUTION_KEYS = {"ridge", "mlp", "random_forest", "extra_trees"}


def engines_module():
    import time
    if _ENG["mod"] is not None:
        return _ENG["mod"]
    if _ENG["err"] and time.time() - _ENG["tried"] < 20:
        return None
    _ENG["tried"] = time.time()
    try:
        import engines
        n_real = len(engines.list_engines())
        if n_real < 2 and os.environ.get("FARESIGHT_MOCK") == "1":
            raise RuntimeError(f"only {n_real} real engine listed; using the mock for development")
        _ENG.update(mod=engines, mock=False, err=None)
    except Exception as e:
        _ENG["err"] = f"{type(e).__name__}: {e}"
        if os.environ.get("FARESIGHT_MOCK") == "1":
            try:
                import engines_mock
                engines_mock.set_base(MODEL)
                _ENG.update(mod=engines_mock, mock=True)
            except Exception as e2:
                _ENG["err"] += f" (mock also failed: {e2})"
    return _ENG["mod"]


def engine_list():
    mod = engines_module()
    if mod is None:
        return [dict(HGB_ENTRY)]
    try:
        lst = mod.list_engines()
    except Exception:
        return [dict(HGB_ENTRY)]
    out = []
    for e in lst:
        e = {**HGB_ENTRY, **e, "label": HGB_ENTRY["label"], "description": HGB_ENTRY["description"], "available": True, "unavailable_reason": None} if e["key"] == "hgb" else dict(e)
        t = TIERS.get(e["key"])
        e["tier"], e["tier_label"] = (t[0], t[1]) if t else (("caution", "Also evaluated") if e["key"] in CAUTION_KEYS else (None, "Also evaluated"))
        e["caution"] = CAUTION if e["key"] in CAUTION_KEYS else None
        out.append(e)
    return out or [dict(HGB_ENTRY)]


def engine_keys():
    return {e["key"] for e in engine_list()}


def resolve_engine(key):
    """(callable engine or None for the deployed model, info dict). Falls back to the deployed model with a reason."""
    req = next((e for e in engine_list() if e["key"] == key), {})
    info = {"key": "hgb", "label": HGB_ENTRY["label"], "requested": key, "requested_label": req.get("label") or key, "fallback": None, "mock": _ENG["mock"]}
    if key in (None, "hgb"):
        return None, info
    mod = engines_module()
    if mod is None:
        info["fallback"] = "The engine files are not installed on this server."
        return None, info
    try:
        eng = mod.load_engine(key)
    except Exception as e:
        info["fallback"] = str(e)
        return None, info
    entry = next((e for e in engine_list() if e["key"] == key), {})
    info.update(key=key, label=getattr(eng, "label", None) or entry.get("label", key))
    return eng, info


@app.get("/api/engines")
def engines_api():
    cfg = load_config()
    lst = engine_list()
    want = cfg["algorithm"]; entry = next((e for e in lst if e["key"] == want), None)
    ok = bool(entry and entry.get("available", True))
    active = want if ok else "hgb"
    for e in lst:
        e["active"] = e["key"] == active
    fallback = None if ok else {"wanted": want, "wanted_label": (entry or {}).get("label") or want,
                                "reason": (entry or {}).get("unavailable_reason") or "this engine is not installed on this server"}
    return jsonify(active=active, engines=lst, mock=_ENG["mock"], us_note="The US market uses its own price-history model.",
                   installed=_ENG["mod"] is not None, fallback=fallback, default=DEFAULTS["algorithm"])


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

    rows = CATALOG[(CATALOG.source == src) & (CATALOG.destination == dst) & (CATALOG["class"] == cls)].reset_index(drop=True)
    if rows.empty:
        return jsonify(error=f"No {cls} flights for {src} to {dst} in the training data."), 404

    period = cfg["price_period"]
    engine, eng_info = resolve_engine(cfg["algorithm"])
    ckey = (src, dst, cls, travel, period, days_left, cfg["predict_at_days_left"], tuple(cfg["curve_days"]),
            cfg["listed_window_days"], eng_info["key"])
    base = _SEARCH_CACHE.get(ckey)
    if base is None:
        wanted = [cfg["predict_at_days_left"], days_left] + list(cfg["curve_days"])
        prices, feats_by_day = predict_many(rows, wanted, travel, period, engine)
        feats = feats_by_day[cfg["predict_at_days_left"]]
        route_df = DF[(DF.source == src) & (DF.destination == dst) & (DF["class"] == cls)]
        groups = {code: (g.days_left.values, g.price.values) for code, g in route_df.groupby("flight_code")}

        def listed_for(code):
            d, p = groups[code]; near = p[np.abs(d - days_left) <= cfg["listed_window_days"]]
            return float(np.median(near if len(near) >= 3 else p))

        base = []
        for i, r in enumerate(rows.itertuples(index=False)):
            f = feats.iloc[i]
            base.append({
                "airline": r.airline, "code": r.flight_code, "stops": int(r.stops),
                "dep": r.dep_bucket, "arr": r.arr_bucket, "duration_h": round(float(r.duration_h), 1),
                "listed": round(listed_for(r.flight_code)),
                "predicted": round(float(prices[cfg["predict_at_days_left"]][i])),
                "curve": [{"d": d, "p": round(float(prices[d][i]))} for d in cfg["curve_days"]],
                "_p0": float(prices[days_left][i]),
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
        _SEARCH_CACHE[ckey] = base
        while len(_SEARCH_CACHE) > _CACHE_MAX:
            _SEARCH_CACHE.pop(next(iter(_SEARCH_CACHE)))
    flights, hidden = [], 0
    for b in base:
        defunct = b["airline"] in DEFUNCT
        if defunct and not cfg["include_defunct"]:
            hidden += 1
            continue
        f = {k: v for k, v in b.items() if k != "_p0"}
        f["defunct"] = defunct
        f["defunct_note"] = DEFUNCT.get(b["airline"]) if defunct else None
        add_drop_fields(f, days_left, b["_p0"], cfg)
        f["display_code"] = display_code(b["code"])
        flights.append(f)
    seen = {}
    for f in flights:                                    # same displayed number, different stored flights: say which records each is from
        seen.setdefault(f["display_code"], []).append(f)
    for dc, group in seen.items():
        for f in group:
            f["period_hint"] = "from older 2022 records" if (dc and len(group) > 1 and f["code"] != dc) else None
    if not flights:
        return jsonify(error=f"No {cls} flights for {src} to {dst} from airlines that operate today. Switch on "
                             "\"Include airlines that no longer operate\" in Model & Insights to see historical data."), 404
    return jsonify(source=src, destination=dst, date=travel.isoformat(), days_out=days_out,
                   clamped=days_out > MAX_DAYS, model_days=days_left, page_size=PAGE_SIZE,
                   config=cfg, engine=eng_info, hidden_defunct=hidden, accuracy=accuracy_for(eng_info["key"]), flights=flights)


@app.route("/api/config", methods=["GET", "POST"])
def config_api():
    if request.method == "GET":
        return jsonify(config=load_config(), defaults=DEFAULTS, sorts=SORTS)
    raw = request.get_json(force=True)
    if raw.get("reset"):
        CONFIG_PATH.unlink(missing_ok=True)
        return jsonify(config=dict(DEFAULTS), errors=[])
    if set(raw) <= {"algorithm"}:                        # a bare engine switch keeps every other setting
        raw = {**load_config(), **raw}
    elif "algorithm" not in raw:                          # a settings save that omits the engine keeps the current one
        raw = {**raw, "algorithm": load_config()["algorithm"]}
    cfg, errs = validate_config(raw)
    if errs:
        return jsonify(errors=errs), 400
    eng = None
    if "algorithm" in request.get_json(force=True) and cfg["algorithm"] != "hgb":
        import time
        t0 = time.time()
        _eng, info = resolve_engine(cfg["algorithm"])      # warm-load now, so the first search is not slow
        if info["fallback"]:
            return jsonify(errors=[f"Cannot use {cfg['algorithm']}: {info['fallback']}"], engine=info), 409
        eng = {**info, "load_seconds": round(time.time() - t0, 2)}
    elif "algorithm" in request.get_json(force=True):
        eng = {"key": "hgb", "label": HGB_ENTRY["label"], "load_seconds": 0, "fallback": None}
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    return jsonify(config=cfg, errors=[], engine=eng)


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
        importance=_load_json("importance.json"), engines=engine_list(), default_engine=DEFAULTS["algorithm"])


# ---------------------------------------------------------------- algorithm comparison (read-only)
def _read_results(name):
    p = ROOT / "models" / name
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except (ValueError, OSError):
        return None


@app.get("/api/algorithms")
def algorithms():
    """Results written by the algorithm-comparison run. Never written by the app.

    Development only: FARESIGHT_MOCK=1 with FARESIGHT_MOCK_FILE=/path/to.json serves that file
    as the baseline set when models/algorithms.json does not exist yet (flagged "mock": true).
    """
    base, enr = _read_results("algorithms.json"), _read_results("algorithms_enriched.json")
    mock = False
    if base is None and os.environ.get("FARESIGHT_MOCK") == "1" and os.environ.get("FARESIGHT_MOCK_FILE"):
        try:
            base, mock = json.loads(Path(os.environ["FARESIGHT_MOCK_FILE"]).read_text()), True
        except (ValueError, OSError):
            pass
    sets = {k: v for k, v in (("baseline18", base), ("enriched", enr)) if v}
    return jsonify(status="ready" if sets else "pending", mock=mock, sets=sets,
                   message=None if sets else "No comparison results yet. They appear here once models/algorithms.json exists.")


# ---------------------------------------------------------------- United States market (optional, lazy)
# us_service is imported on first use, so the app starts and India works when the US model files are missing.
# Development only: FARESIGHT_MOCK=1 falls back to src/us_mock.py (sample data, flagged "mock": true) when the real
# service cannot load.
_US = {"mod": None, "err": None, "mock": False, "tried": 0.0}
US_MISSING = "US data is not installed on this server."


def us_module():
    """Return the US service module, or None. Failures are retried at most every 20 seconds."""
    import time
    if _US["mod"] is not None:
        return _US["mod"]
    if _US["err"] and time.time() - _US["tried"] < 20:
        return None
    _US["tried"] = time.time()
    try:
        import us_service
        us_service.us_meta()                      # forces the lazy model and snapshot load
        _US.update(mod=us_service, err=None, mock=False)
    except Exception as e:                        # missing files, missing optional libraries, anything
        _US["err"] = f"{type(e).__name__}: {e}"
        if os.environ.get("FARESIGHT_MOCK") == "1":
            try:
                import us_mock
                _US.update(mod=us_mock, mock=True)
            except Exception as e2:
                _US["err"] += f" (mock also failed: {e2})"
    return _US["mod"]


def apply_us_drop_rule(res):
    """Recompute may_drop / big_save for US flights with the shared rule (drop_logic.py), in place.

    The service's own flags are replaced. The service's curve starts at its first day ahead, so that first point
    is the as-of model price (P0) and the later points are the future (at most 14 days after the listing date, the
    tested range). The card's big slot is FareSight at the horizon (`card_*`); a tagged flight adds the expected low.
    """
    c = res.get("config", {})
    for f in res.get("flights", []):
        curve = [(p["days_left"], p["price"]) for p in f.get("curve", []) if p.get("price") is not None]
        listed = f["listed_price"]
        di = {"may_drop": False}
        if len(curve) >= 2:
            series = future_series(curve[0][0], curve[0][1], curve[1:])
            di = drop_info(listed, series, c.get("flag_threshold_pct", 10), c.get("big_save_threshold_pct", 20),
                           c.get("big_save_min_usd", 20), headline=f["predicted_price"])
            f["p0"] = round(curve[0][1], 2)
        else:
            f["p0"] = None
        dbd0 = f.get("days_left_at_listing")
        f["may_drop"] = di["may_drop"]
        f["drop_best_days_left"] = di.get("drop_best_days_left")
        f["drop_best_days_ahead"] = (dbd0 - di["drop_best_days_left"]) if di["may_drop"] and dbd0 is not None else None
        f["drop_price"] = round(di["drop_price"], 2) if di["may_drop"] else None
        f["drop_pair_days_left"] = di.get("low_pair")
        f["drop_pct"] = round((listed - f["drop_price"]) / listed * 100, 1) if di["may_drop"] else 0
        f["save_usd"] = round(listed - f["drop_price"], 2) if di["may_drop"] else 0
        big_min = c.get("big_save_min_usd", 20)
        f["big_save"] = bool(di["may_drop"] and big_save_rule(f["drop_pct"], f["save_usd"], c.get("big_save_threshold_pct", 20), big_min))
        f["card_price"], f["card_kind"] = f["predicted_price"], "horizon"
        f["card_days_left"] = (dbd0 - (f.get("predicted_horizon_days") or 0)) if dbd0 is not None else None
        f["card_change_pct"] = round((f["predicted_price"] - listed) / listed * 100, 1) if listed else 0.0
        f["low_change_pct"] = -f["drop_pct"] if di["may_drop"] else None


def finite_json(o):
    """Replace NaN and infinity with None, recursively. Browsers cannot parse the bare NaN that Python's JSON writes,
    and some US inputs are missing (for example price history on a flight's first quote)."""
    if isinstance(o, float) or isinstance(o, np.floating):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, dict):
        return {k: finite_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite_json(v) for v in o]
    return o


@app.get("/api/us/meta")
def us_meta_api():
    mod = us_module()
    if mod is None:
        return jsonify(available=False, error=US_MISSING), 503
    try:
        meta = mod.us_meta()
    except Exception as e:
        return jsonify(available=False, error=US_MISSING, detail=str(e)), 503
    return jsonify(finite_json(dict(available=True, mock=_US["mock"], **{k: v for k, v in meta.items() if k != "mock"})))


@app.post("/api/us/search")
def us_search_api():
    import time
    mod = us_module()
    if mod is None:
        return jsonify(available=False, error=US_MISSING), 503
    q = request.get_json(force=True, silent=True)
    if not isinstance(q, dict):
        return jsonify(error="Send a JSON body with source, destination and date."), 400
    src, dst, date = q.get("source"), q.get("destination"), q.get("date")
    if not src or not dst or not date:
        return jsonify(error="Pick an origin, a destination and a departure date."), 400
    if str(src).upper() == str(dst).upper():
        return jsonify(error="Pick two different airports."), 400
    cfg = load_config()
    ucfg = {"flag_threshold_pct": cfg["flag_threshold_pct"], "big_save_threshold_pct": cfg["big_save_threshold_pct"]}
    try:
        if q.get("ahead") is not None:
            ucfg["predict_ahead_days"] = int(q["ahead"])
        t0 = time.time()
        res = mod.us_search(src, dst, date, q.get("cabin"), ucfg)
        res["elapsed_ms"] = round((time.time() - t0) * 1000)
    except (TypeError, ValueError) as e:
        return jsonify(error=f"Invalid input: {e}"), 400
    except Exception:
        app.logger.exception("us_search failed")
        return jsonify(error="The US search failed on the server. Try another route or date."), 500
    if not res.get("ok", True):
        return jsonify(error=res.get("error", "Search failed.")), 404
    apply_us_drop_rule(res)
    res["mock"] = _US["mock"]
    return jsonify(finite_json(res))


# ---------------------------------------------------------------- Price Lab (what-if) ----------------------------------
import lab as _lab

_LAB = {}


def lab_spec():
    """Description of the 18 inputs for the Price Lab: label, meaning, widget, limits and options (built once)."""
    if "spec" in _LAB:
        return _LAB["spec"]
    cats = ENC.cats
    opts = lambda vals: [{"value": v, "label": str(v)} for v in vals]
    dists = [float(distance_km_pair(a, b)) for a in COORDS for b in COORDS if a != b]
    top_supply = int(ENC.supply_map.max())
    out = []
    for key, label, hlp, widget, derived in _lab.SPECS:
        sp = {"key": key, "label": label, "help": hlp, "derived": derived, "widget": widget}
        if key in ("airline", "source", "destination", "class", "dep_bucket", "arr_bucket"):
            sp.update(type="select", options=opts(cats[key]))
        elif key == "dow":
            sp.update(type="select", options=[{"value": i, "label": n} for i, n in enumerate(_lab.WEEKDAYS)])
        elif key == "month":
            sp.update(type="select", options=[{"value": i + 1, "label": n} for i, n in enumerate(_lab.MONTHS)])
        elif key == "period":
            sp.update(type="select", options=[{"value": 2022, "label": "2022 fare level"}, {"value": 2023, "label": "2023 fare level (latest)"}])
        else:
            rng = {"days_left": (1, MAX_DAYS, 1, True), "duration_h": (0.5, 50, 0.25, False), "stops": (0, 2, 1, True), "dep_hour": (0, 24, 0.25, False),
                   "arr_hour": (0, 24, 0.25, False), "distance_km": (round(min(dists) * 0.9), round(max(dists) * 1.1), 10, False),
                   "route_supply": (1, top_supply, 1, True), "flight_price_level_inr": (float(DF.price.min()), float(DF.price.max()), 50, False),
                   "route_airline_price_level_inr": (float(DF.price.min()), float(DF.price.max()), 50, False)}[key]
            sp.update(type="number", min=rng[0], max=rng[1], step=rng[2], integer=rng[3], nullable=key in ("dep_hour", "arr_hour"))
        out.append(sp)
    _LAB["spec"] = out
    return out


def distance_km_pair(a, b):
    import features as _f
    return float(_f.distance_km(pd.Series([a]), pd.Series([b])).iloc[0])


def lab_reference():
    """A typical flight in the data: most common route/airline/class/windows, median numbers, average typical prices."""
    if "ref" in _LAB:
        return _LAB["ref"]
    mode = lambda c: DF[c].mode().iat[0]
    route = DF.groupby(["source", "destination"]).size().idxmax()
    g = DF[(DF.source == route[0]) & (DF.destination == route[1])]
    ref = {"airline": g[~g.airline.isin(DEFUNCT)].airline.mode().iat[0], "source": route[0], "destination": route[1], "class": "Economy",
           "dep_bucket": mode("dep_bucket"), "arr_bucket": mode("arr_bucket"), "stops": int(DF.stops.median()),
           "duration_h": float(round(DF.duration_h.median(), 2)), "days_left": int(DF.days_left.median()), "dow": int(DF.dow.mode().iat[0]) if "dow" in DF else 3,
           "month": int(DF.month.mode().iat[0]) if "month" in DF else 2, "period": 2023,
           "dep_hour": float(round(DF.dep_hour.median(), 2)), "arr_hour": float(round(DF.arr_hour.median(), 2)),
           "distance_km": round(distance_km_pair(*route), 1), "route_supply": int(ENC.supply_map.median()),
           "flight_price_level_inr": float(round(math.exp(np.log(DF.price).mean()), 0)), "route_airline_price_level_inr": float(round(math.exp(np.log(DF.price).mean()), 0))}
    _LAB["ref"] = ref
    return ref


def lab_frame(vals):
    """One encoded row, built with the same encoder the search uses; typical-price, supply and distance are taken as given."""
    row = {"source": vals["source"], "destination": vals["destination"], "class": vals["class"], "flight_code": "_lab", "airline": vals["airline"],
           "stops": vals["stops"], "dep_bucket": vals["dep_bucket"], "arr_bucket": vals["arr_bucket"], "duration_h": vals["duration_h"],
           "dep_hour": np.nan if vals["dep_hour"] is None else vals["dep_hour"], "arr_hour": np.nan if vals["arr_hour"] is None else vals["arr_hour"],
           "days_left": vals["days_left"], "dow": vals["dow"], "month": vals["month"], "period": vals["period"]}
    feats = ENC.transform(add_basic(pd.DataFrame([row])))
    feats["distance_km"] = vals["distance_km"]
    feats["te_flight"] = math.log(max(vals["flight_price_level_inr"], 1.0))
    feats["te_route_airline"] = math.log(max(vals["route_airline_price_level_inr"], 1.0))
    feats["route_supply"] = vals["route_supply"]
    return feats[ENC.features].astype(float)


@app.get("/lab")
def lab_page():
    return render_template("lab.html", page="lab", cities=CITIES, classes=CLASSES)


@app.get("/walkthrough")
def walkthrough_page():
    return render_template("walkthrough.html", page="walk")


@app.get("/how")
def how_page():
    return render_template("how.html", page="how")


@app.get("/api/facts")
def facts_api():
    """Numbers the plain-English pages quote, read from the models/ files so they never drift from the rest of the app."""
    alg = _load_json("algorithms.json") or {}
    median = next((r for r in alg.get("results", []) if "median lookup" in r.get("algorithm", "").lower()), None)
    per = {int(p): int(n) for p, n in DF.groupby("period").size().items()}
    out = {"metrics": _load_json("metrics.json"), "importance": _load_json("importance.json"), "engines": engine_list(),
           "median_lookup": median, "rows": {"total": int(len(DF)), "by_scrape": per, "routes": int((DF.source + DF.destination).nunique()), "cities": CITIES,
                                            "airlines": int(DF.airline.nunique())},
           "us": None}
    mod = us_module()
    if mod is not None:
        try:
            m = mod.us_meta(); out["us"] = {"routes": len(m["pairs"]), "columns": m["feature_columns_total"], "headline": m.get("headline_metrics"), "listed_as_of": m["listed_as_of"],
                                            "counts": m.get("counts"), "families": [{"title": f["title"], "columns": f["columns"]} for f in m["feature_families"]]}
        except Exception:
            out["us"] = None
    return jsonify(finite_json(out))


@app.get("/api/lab/spec")
def lab_spec_api():
    return jsonify(finite_json({"inputs": lab_spec(), "reference": lab_reference(), "note": "Effects are the model's estimates for this one flight, not rules of the airline."}))


@app.post("/api/whatif")
def whatif_api():
    """Predicted price for a full set of edited inputs, with a breakdown whose steps add up to price - base."""
    q = request.get_json(force=True, silent=True)
    if not isinstance(q, dict):
        return jsonify(error="Send a JSON body with the input values."), 400
    spec, ref = lab_spec(), lab_reference()
    try:
        vals, warns = _lab.sanitize(q.get("values"), spec, ref)
        orig = _lab.sanitize(q["original"], spec, ref)[0] if q.get("original") else None
    except _lab.BadValue as e:
        return jsonify(error=str(e)), 400
    key = q.get("engine") or load_config()["algorithm"]
    engine, info = resolve_engine(key)
    model = engine or MODEL
    price_of = lambda df: np.exp(np.asarray(model.predict(df[ENC.features] if len(df.columns) != len(ENC.features) else df), dtype=float))
    try:
        frame = lab_frame(vals)
        price = float(price_of(frame)[0])
        method, base, eff = "sampled", None, None
        if info["key"] == "xgboost":
            try:
                import xgboost as xgb, engines as _eng
                X = frame[engine.features]
                booster = engine.model.get_booster()
                c = booster.predict(xgb.DMatrix(_eng._xgb_frame(X, engine.cat_sizes), enable_categorical=True), pred_contribs=True)[0]
                base, per = _lab.shap_to_rupees(c[:-1], float(c[-1]))
                base = price - float(np.sum(per))          # absorb float32 rounding so the steps add up exactly
                eff = dict(zip(engine.features, [float(v) for v in per])); method = "shap"
            except Exception:
                method = "sampled"
        if method == "sampled":
            refrow = lab_frame(ref).iloc[0].to_dict(); xrow = frame.iloc[0].to_dict()
            rows, plan, cols = _lab.sampled_shapley(None, refrow, xrow)
            prices = price_of(pd.DataFrame(rows)[ENC.features])
            base, price_s, eff = _lab.finish_shapley(prices, plan, cols)
            price = float(price_s)
        steps = []
        inv = {v: k for k, v in _lab.TE.items()}
        for sp in spec:
            col = _lab.TE.get(sp["key"], sp["key"])
            steps.append({"key": sp["key"], "label": sp["label"], "effect": round(float(eff[col]), 2)})
        out = {"engine": {"key": info["key"], "label": info["label"], "fallback": info["fallback"]}, "method": method, "price": round(price, 2),
               "base_price": round(float(base), 2), "steps": steps, "values": vals, "warnings": warns}
        if orig is not None:
            o_price = float(price_of(lab_frame(orig))[0]); out["original_price"] = round(o_price, 2); out["difference"] = round(price - o_price, 2)
            changed = [sp["key"] for sp in spec if vals[sp["key"]] != orig[sp["key"]]]
            moves = []
            if changed:
                fr = pd.concat([lab_frame({**orig, k: vals[k]}) for k in changed], ignore_index=True)
                for k, p in zip(changed, price_of(fr)):
                    moves.append({"key": k, "label": next(sp["label"] for sp in spec if sp["key"] == k), "from": orig[k], "to": vals[k], "delta": round(float(p) - o_price, 2)})
            out["moves"] = moves
            out["biggest"] = max(moves, key=lambda m: abs(m["delta"])) if moves else None
        return jsonify(finite_json(out))
    except Exception:
        app.logger.exception("whatif failed")
        return jsonify(error="The what-if calculation failed on the server. Try different values."), 500


@app.post("/api/trace")
def trace_api():
    """The XGBoost calculation for one edited flight, tree by tree, plus the normal what-if breakdown.
    Same body as /api/whatif ({values, engine}). Other engines get available=false and the breakdown only."""
    rv = whatif_api()
    resp, status = (rv if isinstance(rv, tuple) else (rv, 200))
    if status != 200:
        return rv
    wi = resp.get_json()
    out = {"available": False, "engine": wi["engine"]["key"], "breakdown": wi}
    try:
        import engines as _eng
        t = _eng.trace(wi["engine"]["key"], lab_frame(wi["values"]))
        out.update(t)
        out["breakdown"] = wi
        if t.get("available") and abs(t["price"] - wi["price"]) > 1.0:
            app.logger.error("trace %s and whatif %s disagree", t["price"], wi["price"])
    except Exception as e:
        app.logger.warning("trace unavailable: %s", e)
        out["reason"] = "The XGBoost trace could not be built on this server. The breakdown below still works."
    if not out.get("available") and wi["engine"]["key"] != "xgboost":
        out["reason"] = "The tree-by-tree view is only available for the XGBoost engine. The breakdown below still works."
    return jsonify(finite_json(out))


def _warm_default_engine():
    try:
        resolve_engine(DEFAULTS["algorithm"])      # load the default engine now so the first search is not slow
    except Exception:
        pass


import threading as _threading
_threading.Thread(target=_warm_default_engine, daemon=True).start()


if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 5055)), debug=False)
