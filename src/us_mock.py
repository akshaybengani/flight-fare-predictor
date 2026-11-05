"""MOCK of us_service, for building the UI while the real service is not ready.

Only used when FARESIGHT_MOCK=1 and the real `us_service` cannot be loaded (see src/app.py). Same function names and
output shape as us_service (field names, grouped `inputs`, curve points). Numbers are random and deterministic per
route and date; they are NOT results. Responses carry mock=True and the UI labels them as sample data.
"""
import random
import datetime as dt

from us_service import FAMILIES, FAMILY_TITLES, FAMILY_DESC, _LABEL, MAX_AHEAD, MIN_DBD, DEFAULTS

ASOF = dt.date(2026, 9, 30)
DEP_MIN, DEP_MAX = dt.date(2026, 10, 7), dt.date(2026, 11, 14)
AIRPORTS = {"ATL": "Atlanta", "BOS": "Boston", "DEN": "Denver", "DFW": "Dallas-Fort Worth", "JFK": "New York JFK",
            "LAX": "Los Angeles", "MIA": "Miami", "ORD": "Chicago O'Hare", "SEA": "Seattle", "SFO": "San Francisco"}
PAIRS = [f"{a}-{b}" for a in sorted(AIRPORTS) for b in sorted(AIRPORTS) if a != b][:50]
AIRLINES = ["AA", "DL", "UA", "WN", "B6", "AS"]


def us_meta():
    origins = sorted({p.split("-")[0] for p in PAIRS}); dests = sorted({p.split("-")[1] for p in PAIRS})
    fams = [{"key": k, "title": FAMILY_TITLES[k], "columns": len(v), "description": FAMILY_DESC[k]} for k, v in FAMILIES.items()]
    return {
        "mock": True, "currency": "USD", "listed_as_of": ASOF.isoformat(), "origins": origins, "destinations": dests, "pairs": PAIRS,
        "airports": {a: {"city": AIRPORTS[a]} for a in AIRPORTS},
        "departure_date_range": {"min": DEP_MIN.isoformat(), "max": DEP_MAX.isoformat()},
        "counts": {"flights_in_snapshot": 4200, "routes": len(PAIRS), "airlines": len(AIRLINES), "quotes_in_dataset": 12000000,
                   "flights_in_dataset": 90000, "search_dates": "2026-01-01 to 2026-09-30"},
        "feature_families": fams, "feature_columns_total": sum(f["columns"] for f in fams),
        "horizon": {"default_days": 1, "min": 1, "max": MAX_AHEAD,
                    "accuracy": [{"days_ahead": d, "model_accuracy_pct": round(97.5 - d * .25, 2), "naive_last_price_accuracy_pct": round(97.3 - d * .27, 2)} for d in (1, 2, 3, 5, 7, 10, 14)],
                    "note": "Sample numbers for layout only."},
        "headline_metrics": {"split": "SAMPLE (mock)", "model_accuracy_pct": 97.4, "naive_last_price_accuracy_pct": 97.2,
                             "naive_route_days_out_median_accuracy_pct": 88.1, "direction_given_a_move": 61.0,
                             "share_next_prices_unchanged_pct": 72.0, "feature_families": "sample text"},
        "defaults": DEFAULTS,
        "honest_note": "The model barely improves on 'the price stays the same': predicted_price is usually within a few percent of listed_price and may_drop flags are rare. Cabin is not in the data; listed_price is the cheapest quote for the flight.",
        "sources": [{"name": "Sample source", "licence": "n/a (mock)"}],
        "data_limits": ["Mock data: nothing here is real.", "Open-Meteo free API data is for non-commercial use only."],
    }


def _hhmm(m):
    return f"{int(m // 60) % 24:02d}:{int(m % 60):02d}"


def us_search(origin, destination, date, cabin=None, config=None):
    cfg = dict(DEFAULTS); cfg.update({k: v for k, v in (config or {}).items() if k in cfg and v is not None})
    base = {"ok": True, "currency": "USD", "listed_as_of": ASOF.isoformat(), "config": cfg, "flights": [], "count": 0,
            "cabin": None, "note": None, "mock": True}
    o, d = str(origin or "").upper(), str(destination or "").upper()
    if f"{o}-{d}" not in PAIRS:
        return {**base, "ok": False, "error": f"no data for route {o}-{d}"}
    try:
        day = dt.date.fromisoformat(str(date))
    except ValueError:
        return {**base, "ok": False, "error": f"bad date {date!r} (use YYYY-MM-DD)"}
    if not DEP_MIN <= day <= DEP_MAX:
        base["note"] = f"no flights for {day}: the snapshot covers departures {DEP_MIN} to {DEP_MAX} (listed as of {ASOF})"
        return base
    rnd = random.Random(f"{o}{d}{day}")
    dbd0 = (day - ASOF).days
    flights = []
    for i in range(rnd.randint(7, 24)):
        air = rnd.choice(AIRLINES); dep = rnd.randint(300, 1320); dur = rnd.randint(90, 360); stops = rnd.choice([0, 0, 1, 1, 2])
        listed = round(rnd.uniform(70, 520), 2); arr = dep + dur + 60 * stops
        price, walk = listed, []
        drift = rnd.choice([0, 0, 0, -0.02, 0.01, -0.04])
        for h in range(1, MAX_AHEAD + 1):
            price *= 1 + drift + rnd.gauss(0, 0.012)
            if rnd.random() < 0.04: price *= rnd.choice([0.85, 0.8, 1.12])
            walk.append(price if dbd0 - h >= MIN_DBD else float("nan"))
        valid = [p == p for p in walk]
        h = int(min(cfg["predict_ahead_days"], max([j + 1 for j, v in enumerate(valid) if v] or [0])))
        pred = walk[h - 1] if h else listed
        pts = [(j + 1, walk[j]) for j in range(max(h - 1, 0), MAX_AHEAD) if valid[j]]
        drop_pct, bd, bp = 0.0, None, None
        if pts:
            dd, pp = min(pts, key=lambda t: (t[1], t[0])); drop_pct = max(0.0, round((listed - pp) / listed * 100, 1)); bd, bp = dd, pp
        may = drop_pct >= cfg["flag_threshold_pct"]; save = round(max(0.0, listed - bp), 2) if may else 0
        big = bool(may and (drop_pct >= cfg["big_save_threshold_pct"] or (cfg["big_save_min_usd"] > 0 and save >= cfg["big_save_min_usd"])))
        inputs = {}
        for fam, cols in FAMILIES.items():
            inputs[fam] = {"title": FAMILY_TITLES[fam], "items": [
                {"key": c, "label": _LABEL[c][0], "value": round(rnd.uniform(0, 1) * (listed if _LABEL[c][1] == "$" else 10), 2)} for c in cols]}
        flights.append({
            "id": f"{o}-{d}-{day}-{i}", "airline": air, "flight_code": f"{air} {_hhmm(dep)}", "source": o, "destination": d,
            "dep_date": str(day), "dep_time": _hhmm(dep), "arr_time": _hhmm(arr), "arr_day_offset": int(arr // 1440), "stops": stops,
            "duration_h": round(dur / 60, 2), "cabin": None, "currency": "USD", "listed_price": listed, "listed_as_of": ASOF.isoformat(),
            "days_left_at_listing": dbd0, "predicted_price": round(pred, 2), "predicted_horizon_days": h, "horizon_requested_days": cfg["predict_ahead_days"],
            "change_pct": round((pred - listed) / listed * 100, 1), "may_drop": may, "drop_pct": drop_pct if may else 0,
            "drop_best_days_left": (dbd0 - bd) if may else None, "drop_best_days_ahead": bd if may else None,
            "drop_price": round(bp, 2) if may else None, "big_save": big, "save_usd": save,
            "curve": [{"days_ahead": j + 1, "days_left": dbd0 - (j + 1), "price": round(walk[j], 2)} for j in range(MAX_AHEAD) if valid[j]],
            "inputs": inputs})
    flights.sort(key=lambda f: (f["dep_time"], f["airline"]))
    base.update(flights=flights, count=len(flights))
    return base
