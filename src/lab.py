"""Price Lab: what-if predictions and a plain breakdown of one flight's price.

Pure helpers (no Flask). `app.py` builds the encoded feature frame from the user's values through the same encoder the
search uses, and passes a `predict_prices(frame) -> ndarray` function in.

Breakdown method
  * XGBoost: exact TreeSHAP contributions (log-price), shared out into rupees so the steps add up to price - base.
  * every other engine: sampled Shapley values (random orders of switching each input from a typical flight to this
    flight, all predicted in one batch). Each order adds up exactly; the average over orders is an estimate.
"""
import math

import numpy as np

# key, plain label, one-sentence meaning, widget, derived?
SPECS = [
    ("flight_price_level_inr", "Typical price of this flight", "The average fare the training data shows for this flight number. Calculated from past data.", "money", True),
    ("route_airline_price_level_inr", "Typical price of this airline on the route", "The average fare the data shows for this airline on this route. Calculated from past data.", "money", True),
    ("days_left", "Days before departure", "How many days ahead of the trip the price is for. Fares usually rise as the date gets close.", "slider", False),
    ("duration_h", "Journey time (hours)", "Total time from take-off to landing, including any stops. Longer journeys often cost more.", "slider", False),
    ("airline", "Airline", "Different airlines price the same trip differently.", "select", False),
    ("source", "From", "The city you leave from.", "select", False),
    ("destination", "To", "The city you fly to.", "select", False),
    ("class", "Class of travel", "Business costs several times more than Economy.", "select", False),
    ("stops", "Number of stops", "Non-stop flights are usually dearer than flights with a stop.", "slider", False),
    ("dep_hour", "Departure time (hour of day)", "When the plane leaves, as an hour from 0 to 24. Leave empty if the data has no exact time.", "slider", False),
    ("arr_hour", "Arrival time (hour of day)", "When the plane lands, as an hour from 0 to 24. Leave empty if the data has no exact time.", "slider", False),
    ("dep_bucket", "Departure time window", "A rough part of the day for take-off.", "select", False),
    ("arr_bucket", "Arrival time window", "A rough part of the day for landing.", "select", False),
    ("distance_km", "Distance (km)", "Straight-line distance between the two cities. Calculated from the route.", "number", True),
    ("route_supply", "Number of flights on the route", "How many different flights serve this route and class: a rough measure of competition. Calculated from past data.", "number", True),
    ("dow", "Weekday of travel", "The day of the week the plane leaves.", "select", False),
    ("month", "Month of travel", "The month the plane leaves.", "select", False),
    ("period", "Which scrape the fare level comes from", "The data was collected in 2022 and 2023, and 2023 fares are about 55% higher. Choose the fare level to price at.", "select", False),
]
MODEL_ORDER = ["airline", "source", "destination", "class", "dep_bucket", "arr_bucket", "stops", "duration_h", "days_left", "dow", "month",
               "period", "dep_hour", "arr_hour", "distance_km", "te_flight", "te_route_airline", "route_supply"]
# the two typical-price inputs reach the model as logs under these names
TE = {"flight_price_level_inr": "te_flight", "route_airline_price_level_inr": "te_route_airline"}
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


class BadValue(ValueError):
    """A value the user typed cannot be used; the message is shown to them."""


def sanitize(values, spec, reference):
    """Return (clean values, warnings). Missing keys take the reference value; numbers are clamped to the allowed range."""
    clean, warns = {}, []
    if not isinstance(values, dict):
        raise BadValue("Send the values as an object of input names and values.")
    for s in spec:
        k = s["key"]; v = values.get(k, reference[k])
        if s["type"] == "select":
            opts = s["options"]; ok = [o["value"] for o in opts]
            if isinstance(v, str) and v.lstrip("-").isdigit() and not isinstance(ok[0], str):
                v = int(v)
            if v not in ok:
                raise BadValue(f"{s['label']} must be one of: " + ", ".join(str(o["label"]) for o in opts) + ".")
            clean[k] = v
            continue
        if v is None or v == "":
            if s.get("nullable"):
                clean[k] = None; continue
            raise BadValue(f"{s['label']} needs a number.")
        try:
            x = float(v)
        except (TypeError, ValueError):
            raise BadValue(f"{s['label']} must be a number, not {str(v)[:20]!r}.")
        if not math.isfinite(x):
            raise BadValue(f"{s['label']} must be a real number.")
        lo, hi = s["min"], s["max"]
        if x < lo or x > hi:
            warns.append(f"{s['label']} was changed from {x:g} to {min(max(x, lo), hi):g}, the range the model has seen.")
            x = min(max(x, lo), hi)
        clean[k] = round(x) if s.get("integer") else x
    if clean["source"] == clean["destination"]:
        raise BadValue("From and To must be two different cities.")
    return clean, warns


def shap_to_rupees(contribs, bias):
    """contribs: log-price contributions; bias: log base. Returns (base_price, effects in rupees summing to price - base)."""
    c = np.asarray(contribs, dtype=float); s = float(c.sum())
    base = math.exp(bias)
    factor = base * math.expm1(s) / s if abs(s) > 1e-9 else base
    return base, c * factor


def sampled_shapley(predict_prices, ref, x, n_perm=20, seed=7):
    """ref, x: dict feature -> value (model columns). Returns (base_price, final_price, effects dict summing to final - base)."""
    cols = list(x)
    rng = np.random.default_rng(seed)
    rows = [dict(ref)]; plan = []
    for _ in range(n_perm):
        order = list(rng.permutation(len(cols))); cur = dict(ref)
        for j in order:
            cur[cols[j]] = x[cols[j]]; rows.append(dict(cur))
        plan.append(order)
    return rows, plan, cols


def finish_shapley(prices, plan, cols):
    base = float(prices[0]); eff = {c: 0.0 for c in cols}; i = 1
    for order in plan:
        prev = base
        for j in order:
            cur = float(prices[i]); eff[cols[j]] += cur - prev; prev = cur; i += 1
    n = len(plan)
    return base, float(prices[i - 1]), {c: v / n for c, v in eff.items()}
