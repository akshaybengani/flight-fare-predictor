"""Tests for the shared "may drop" rule (drop_logic.py) and the invariant the UI relies on.

Run:  cd src && ../.venv/bin/python tests_drop_logic.py        (also works with pytest)

Unit tests need nothing else. The invariant tests call the real app through Flask's test client for India and the
US (the US part is skipped if the US service is not installed). Nothing is written anywhere.
"""
import datetime as dt
import itertools
import sys

from drop_logic import sustained_low, future_series, drop_info, big_save_rule

FLAG, BIG, BIG_MIN = 10, 20, 1500


def info(listed, today, p0, pts, extra=()):
    return drop_info(listed, future_series(today, p0, pts, extra), FLAG, BIG, BIG_MIN)


# ------------------------------------------------------------------ unit tests
def test_one_point_dip_does_not_tag():
    # the 7-day dip is a single point: the level held at two adjacent checkpoints stays at 100
    r = info(100, 40, 100, [(30, 100), (21, 100), (14, 100), (7, 55), (3, 100), (1, 100)])
    assert not r["may_drop"] and r["drop_pct"] == 0 and r["drop_price"] is None


def test_sustained_dip_tags():
    r = info(100, 40, 100, [(30, 95), (21, 60), (14, 62), (7, 100), (3, 100)])
    assert r["may_drop"] and r["drop_best_days_left"] == 14 and r["drop_price"] == 62
    assert r["low_pair"] == [21, 14] and r["drop_pct"] == 38.0 and r["save"] == 38


def test_dip_before_today_is_ignored():
    # today is 14 days out; the cheap points are at 45, 40, 30, 21 days out (the past) and must not count
    r = info(100, 14, 100, [(45, 10), (40, 10), (30, 10), (21, 10), (7, 100), (3, 100)])
    assert not r["may_drop"]
    s = future_series(14, 100, [(45, 10), (21, 10), (14, 10), (7, 90)])
    assert [d for d, _ in s] == [14, 7]            # today's own point and the future only


def test_must_be_below_both_listed_and_model_today():
    # low is 20% below the listed fare but not below the model's own price for today: no tag
    assert not info(100, 30, 82, [(21, 80), (14, 80), (7, 90)])["may_drop"]
    # low is below the model's price for today but not below the listed fare: no tag
    assert not info(70, 30, 100, [(21, 65), (14, 65), (7, 90)])["may_drop"]
    # below both by the threshold: tag
    assert info(100, 30, 100, [(21, 85), (14, 85), (7, 90)])["may_drop"]


def test_card_price_must_also_fall():
    pts = [(21, 85), (14, 85), (7, 90)]
    assert drop_info(100, future_series(30, 100, pts), FLAG, BIG, BIG_MIN, headline=95)["may_drop"]
    assert not drop_info(100, future_series(30, 100, pts), FLAG, BIG, BIG_MIN, headline=100)["may_drop"]     # the big slot is level
    assert not drop_info(100, future_series(30, 100, pts), FLAG, BIG, BIG_MIN, headline=140)["may_drop"]     # the big slot is a rise
    assert not drop_info(100, future_series(30, 100, pts), FLAG, BIG, BIG_MIN, headline=99.5)["may_drop"]    # a fall under 1% is not visible


def test_default_engine_and_fallback():
    import app as appmod
    assert appmod.DEFAULTS["algorithm"] == "xgboost"
    eng, info = appmod.resolve_engine("no_such_engine")           # an unknown or missing engine never errors: it falls back, visibly
    assert eng is None and info["key"] == "hgb" and info["fallback"], info


def test_needs_two_future_points():
    assert sustained_low([(30, 100), (21, 10)]) is None
    assert not info(100, 30, 100, [(21, 10)])["may_drop"]


def test_horizon_point_is_used_when_it_is_a_future_point():
    # curve point at 21 days plus the horizon prediction at 1 day form the pair; the held level is the dearer of the two
    assert sustained_low([(30, 100), (21, 92), (1, 50)]) == (1, 92, (21, 1))   # day of the later point of the pair
    assert not info(100, 30, 100, [(21, 92)], extra=[(1, 50)])["may_drop"]        # 8% below: under the threshold
    assert info(100, 30, 100, [(21, 88)], extra=[(1, 50)])["may_drop"]            # 12% below: tagged
    # a duplicate day keeps the first value
    assert future_series(30, 100, [(7, 60)], [(7, 99)])[1] == (7, 60)


def test_big_save_thresholds():
    # 25% below listed: big by percentage
    assert info(1000, 30, 1000, [(21, 750), (14, 750), (7, 900)])["big_save"]
    # 12% and 1,800 rupees: big, because the amount comes with at least a 10% fall
    r = info(15000, 30, 15000, [(21, 13200), (14, 13200), (7, 14500)])
    assert r["may_drop"] and r["big_save"] and r["drop_pct"] < BIG
    # 12% and 120 rupees: a plain may-drop
    r = info(1000, 30, 1000, [(21, 880), (14, 880), (7, 950)])
    assert r["may_drop"] and not r["big_save"]
    # rupee test off (0): the 12% / 1,800 case is no longer big
    r = drop_info(15000, future_series(30, 15000, [(21, 13200), (14, 13200), (7, 14500)]), FLAG, BIG, 0)
    assert r["may_drop"] and not r["big_save"]


def test_amount_alone_is_not_a_big_save():
    # a 6% fall on a dear fare is 1,800 rupees, but with the May drop limit lowered to 5% it must NOT become a big save
    r = drop_info(30000, future_series(30, 30000, [(21, 28200), (14, 28200), (7, 29000)]), 5, BIG, BIG_MIN)
    assert r["may_drop"] and r["drop_pct"] == 6.0 and r["save"] == 1800 and not r["big_save"]
    assert big_save_rule(9.9, 5000, 20, 1500) is False and big_save_rule(10, 1500, 20, 1500) is True and big_save_rule(20, 1, 20, 1500) is True


# ------------------------------------------------------------------ invariant tests on the real app
def check_flight(f, price_key, save_key, where):
    listed = f["listed"] if "listed" in f else f["listed_price"]
    assert f["card_kind"] == "horizon" and f["card_price"] == f[price_key], f"big slot is not the FareSight price: {where}"
    if f["may_drop"]:
        # a tagged card visibly falls in BOTH places: the big FareSight price and the expected low
        assert rnd(f["card_change_pct"]) <= -1, f"tagged card's FareSight price does not fall: {where} {f['card_change_pct']}"
        assert f["drop_price"] < listed and rnd(f["drop_pct"]) >= 1, where
        assert f["drop_pct"] >= FLAG - 0.05, where
        assert f["drop_best_days_left"] is not None and f["drop_pair_days_left"], where
        assert f[save_key] > 0, where
        assert abs(f["drop_pct"] - round((listed - f["drop_price"]) / listed * 100, 1)) < 0.051, f"drop_pct is not from the shown prices: {where}"
        assert abs(f["drop_pct"] + f["low_change_pct"]) < 0.051, where
        assert f["drop_price"] <= f["p0"] * (1 - FLAG / 100) + 1e-6 + (0.51 if "listed" in f else 0.006), f"low not below the model's own price today: {where}"
    else:
        assert not f["big_save"] and f["drop_pct"] == 0 and f["drop_price"] is None and f[save_key] == 0 and f["low_change_pct"] is None, where
    if f["big_save"]:
        assert f["may_drop"] and f["drop_pct"] >= 10 - 0.05, f"a big save must be at least a 10% fall: {where}"


def run_invariants():
    from app import app, CITIES                      # loads the models (slow, once)
    client = app.test_client()
    today = dt.date.today()
    stats = {"india_searches": 0, "india_flights": 0, "india_tagged": 0, "us_searches": 0, "us_flights": 0, "us_tagged": 0}
    pairs = [("Delhi", "Mumbai"), ("Mumbai", "Delhi"), ("Ahmedabad", "Mumbai"), ("Bangalore", "Delhi"), ("Kolkata", "Delhi"),
             ("Chennai", "Hyderabad"), ("Hyderabad", "Bangalore")]
    for (s, d), days, cls in itertools.product(pairs, (2, 5, 14, 23, 40, 70), ("Economy", "Business")):
        date = (today + dt.timedelta(days=days)).isoformat()
        r = client.post("/api/search", json={"source": s, "destination": d, "date": date, "class": cls})
        if r.status_code != 200:
            continue
        j = r.get_json(); stats["india_searches"] += 1
        stats["filtered_lists"] = stats.get("filtered_lists", 0) + 1
        stats["filtered_flights"] = stats.get("filtered_flights", 0) + check_filtered_list(j["flights"], "predicted", f"India {s}-{d} {date} {cls}")
        for f in j["flights"]:
            check_flight(f, "predicted", "save_inr", f"India {s}-{d} {date} {cls} {f['code']}")
            stats["india_flights"] += 1; stats["india_tagged"] += f["may_drop"]
    meta = client.get("/api/us/meta")
    if meta.status_code == 200:
        pairs_us = meta.get_json()["pairs"]; rng = meta.get_json()["departure_date_range"]
        lo = dt.date.fromisoformat(rng["min"]); hi = dt.date.fromisoformat(rng["max"])
        dates = [lo + dt.timedelta(days=k) for k in (3, 25, 40, 70, 110, 150) if lo + dt.timedelta(days=k) <= hi]
        for p, day in itertools.product(pairs_us, dates):
            o, t = p.split("-")
            r = client.post("/api/us/search", json={"source": o, "destination": t, "date": day.isoformat()})
            assert r.status_code == 200, (p, day, r.status_code)
            j = r.get_json(); stats["us_searches"] += 1
            stats["filtered_lists"] = stats.get("filtered_lists", 0) + 1
            stats["filtered_flights"] = stats.get("filtered_flights", 0) + check_filtered_list(j["flights"], "predicted_price", f"US {p} {day}")
            for f in j["flights"]:
                check_flight(f, "predicted_price", "save_usd", f"US {p} {day} {f['id']}")
                stats["us_flights"] += 1; stats["us_tagged"] += f["may_drop"]
    else:
        print("US service not installed: US invariant tests skipped")
    return stats


# ------------------------------------------------------------------ acceptance: what the filtered list shows
def rnd(v):
    """Round half away from zero, as index.html does (rnd)."""
    import math
    return int(math.copysign(math.floor(abs(v) + 0.5), v))


def ui_list(flights, price_key, drop_only, big_only, sort="drop", page_size=10):
    """The list the page builds for the "Price may drop" / "Big saves" filters, sorting and pager.

    Mirrors index.html (view(), render(), ticket()) field for field: the filters use may_drop and big_save, the
    "Biggest expected drop" sort uses -drop_pct with big saves first, the count in the label is the number of
    may_drop (or big_save) flights, the card's change is rnd(card_change_pct) and the tag says
    "about round(drop_pct)%".
    """
    v = [f for f in flights if (not drop_only or f["may_drop"]) and (not big_only or f["big_save"])]
    if sort == "drop":
        v.sort(key=lambda f: -f["drop_pct"] - (1000 if f["big_save"] else 0))
    pages = [v[i:i + page_size] for i in range(0, len(v), page_size)] or [[]]
    return v, pages


def check_filtered_list(flights, price_key, where):
    for drop_only, big_only in ((True, False), (False, True), (True, True)):
        v, pages = ui_list(flights, price_key, drop_only, big_only)
        label_count = sum(1 for f in flights if (f["big_save"] if big_only else f["may_drop"]))
        listed = sum(len(p) for p in pages)
        assert listed == len(v) == label_count, f"filter count {label_count} != flights listed {listed}: {where}"
        for page in pages:
            assert len(page) <= 10, where
        for f in v:
            big_slot = rnd(f["card_change_pct"]); low = rnd(f["low_change_pct"])
            assert big_slot < 0, f"filtered card's FareSight price does not fall ({big_slot}%): {where}"
            assert low < 0 and rnd(f["drop_pct"]) == -low, f"tag says {rnd(f['drop_pct'])}% but the expected-low line says {low}%: {where}"
        keys = [-f["drop_pct"] - (1000 if f["big_save"] else 0) for f in v]
        assert keys == sorted(keys), f"'Biggest expected later drop' order broken: {where}"
        if v and not drop_only:
            assert all(f["big_save"] for f in v), where
    v, _ = ui_list(flights, price_key, True, False)
    seen_plain = False
    for f in v:
        if not f["big_save"]:
            seen_plain = True
        assert not (seen_plain and f["big_save"]), f"a big save is listed after a plain May drop: {where}"
    return len(v)


def run_engine_invariants():
    """The same invariants for every loadable India engine (the engine is chosen in config; patched in memory here,
    nothing is written). With FARESIGHT_MOCK=1 and fewer than two real engines this runs against the mock engines."""
    import app as appmod
    client = appmod.app.test_client()
    today = dt.date.today(); orig = appmod.load_config; done = {}
    try:
        for e in appmod.engine_list():
            if not e.get("available", True):
                done[e["key"]] = "unavailable: " + str(e.get("unavailable_reason"))
                continue
            appmod.load_config = lambda key=e["key"]: {**orig(), "algorithm": key}
            n = tagged = 0
            for (s, d), days, cls in itertools.product([("Delhi", "Mumbai"), ("Ahmedabad", "Mumbai"), ("Kolkata", "Delhi")], (5, 23, 40), ("Economy",)):
                date = (today + dt.timedelta(days=days)).isoformat()
                r = client.post("/api/search", json={"source": s, "destination": d, "date": date, "class": cls})
                assert r.status_code == 200, (e["key"], s, d, date, r.status_code)
                j = r.get_json()
                assert j["engine"]["key"] == e["key"] and not j["engine"]["fallback"], (e["key"], j["engine"])
                for f in j["flights"]:
                    check_flight(f, "predicted", "save_inr", f"engine {e['key']} {s}-{d} {date} {f['code']}")
                    n += 1; tagged += f["may_drop"]
                check_filtered_list(j["flights"], "predicted", f"engine {e['key']} {s}-{d} {date}")
            done[e["key"]] = f"{n} flights, {tagged} tagged"
    finally:
        appmod.load_config = orig
    return done


def run_defunct_invariants():
    """Airlines that no longer operate: hidden by default, shown and marked on request; filters and counts follow."""
    import app as appmod
    client = appmod.app.test_client(); today = dt.date.today(); orig = appmod.load_config; out = {}
    try:
        for show in (False, True):
            appmod.load_config = lambda show=show: {**orig(), "include_defunct": show}
            n = bad = 0
            for (s, d), days in itertools.product([("Delhi", "Mumbai"), ("Mumbai", "Delhi"), ("Kolkata", "Delhi"), ("Bangalore", "Delhi")], (5, 23, 40)):
                date = (today + dt.timedelta(days=days)).isoformat()
                r = client.post("/api/search", json={"source": s, "destination": d, "date": date, "class": "Economy"})
                if r.status_code != 200:
                    continue
                j = r.get_json()
                for f in j["flights"]:
                    check_flight(f, "predicted", "save_inr", f"defunct={show} {s}-{d} {date} {f['code']}")
                    is_def = f["airline"] in appmod.DEFUNCT
                    assert f["defunct"] == is_def and (show or not is_def), (show, f["airline"])
                    n += 1; bad += is_def
                check_filtered_list(j["flights"], "predicted", f"defunct={show} {s}-{d} {date}")
                assert (j["hidden_defunct"] == 0) == show or show, j["hidden_defunct"]
            out["shown" if show else "hidden"] = f"{n} flights, {bad} from airlines that no longer operate"
    finally:
        appmod.load_config = orig
    return out


def test_defunct_airlines():
    out = run_defunct_invariants(); assert out["hidden"].endswith("0 from airlines that no longer operate"), out


def test_invariants_for_every_engine():
    assert run_engine_invariants()


def test_invariants_on_real_app():
    st = run_invariants()
    assert st["india_flights"] > 1000, st


if __name__ == "__main__":
    fails = 0
    for name, fn in [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and n not in ("test_invariants_on_real_app", "test_invariants_for_every_engine", "test_defunct_airlines")]:
        try:
            fn(); print("ok  ", name)
        except AssertionError as e:
            fails += 1; print("FAIL", name, e)
    try:
        print("ok   defunct airlines hidden / shown", run_defunct_invariants())
        print("ok   invariants for every loadable engine", run_engine_invariants())
        st = run_invariants(); print("ok   invariants on the real app", st)
    except AssertionError as e:
        fails += 1; print("FAIL invariants:", e)
    print("FAILED" if fails else "ALL PASSED"); sys.exit(1 if fails else 0)
