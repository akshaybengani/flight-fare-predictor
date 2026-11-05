"""Self-test for src/us_service.py (not a pytest suite):  python src/us_service_selftest.py"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import us_service as U   # noqa: E402

fails = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name, extra)
    if not cond:
        fails.append(name)


t = time.time(); meta = U.us_meta(); print(f"us_meta cold (loads model + snapshot): {time.time() - t:.2f}s")
check("meta pairs == 50", len(meta["pairs"]) == 50, len(meta["pairs"]))
check("meta has families", len(meta["feature_families"]) == 6 and meta["feature_columns_total"] == len(U.FEATURES))
check("meta date range", meta["departure_date_range"]["min"] and meta["departure_date_range"]["max"], meta["departure_date_range"])
check("meta horizon accuracy", bool(meta["horizon"]["accuracy"]))
print("  counts:", meta["counts"])
print("  horizon accuracy:", [(r["horizon_days"], r["model_accuracy_pct"], r["last_price_accuracy_pct"], r["verdict"]) for r in (meta["horizon"]["accuracy"] or [])])

d0 = meta["departure_date_range"]["min"]
cases = [("LAX", "SFO", "2026-11-15"), ("JFK", "LAX", "2026-12-23"), ("ORD", "DEN", "2027-01-20"), ("ATL", "DFW", "2026-10-08"),
         ("DCA", "BOS", "2027-03-01"), ("SEA", "LAX", d0)]
times = []
for o, d, date in cases:
    for k in range(2):                       # first call warm-up for that route, second timed
        t = time.time(); r = U.us_search(o, d, date); dt = time.time() - t
    times.append(dt)
    fl = r["flights"]
    print(f"search {o}-{d} {date}: {len(fl)} flights, {dt * 1000:.0f} ms, may_drop={sum(f['may_drop'] for f in fl)}, big_save={sum(f['big_save'] for f in fl)}")
    check(f"{o}-{d} {date} returns flights", r["ok"] and len(fl) > 0)
    if fl:
        f = fl[0]
        need = ["id", "airline", "flight_code", "source", "destination", "dep_time", "arr_time", "stops", "duration_h", "cabin", "currency",
                "listed_price", "listed_as_of", "predicted_price", "change_pct", "may_drop", "drop_pct", "drop_best_days_left", "drop_price",
                "big_save", "save_usd", "curve", "inputs"]
        check("  fields present", all(k in f for k in need), [k for k in need if k not in f])
        check("  prices sane", 15 < f["listed_price"] < 6000 and 15 < f["predicted_price"] < 8000, (f["listed_price"], f["predicted_price"]))
        check("  inputs grouped", set(f["inputs"]) == set(U.FAMILIES) and sum(len(v["items"]) for v in f["inputs"].values()) == len(U.FEATURES))
        g = U.us_curve(f["id"])
        check("  us_curve matches search", g is not None and g["curve"] == f["curve"])
check("avg search under 1 s after warm-up", sum(times) / len(times) < 1.0, f"{sum(times) / len(times) * 1000:.0f} ms avg, max {max(times) * 1000:.0f} ms")

r = U.us_search("LAX", "XXX", "2026-11-15"); check("unknown route -> ok False, no flights", (not r["ok"]) and r["flights"] == [], r.get("error"))
r = U.us_search("LAX", "SFO", "2028-01-01"); check("date after data -> empty list, ok", r["ok"] and r["flights"] == [], r.get("note"))
r = U.us_search("LAX", "SFO", "2026-03-01"); check("date before data -> empty list, ok", r["ok"] and r["flights"] == [])
r = U.us_search("LAX", "SFO", "not-a-date"); check("bad date -> ok False", not r["ok"])
r = U.us_search("lax", "sfo", "2026-11-15", config={"predict_ahead_days": 14, "flag_threshold_pct": 1}); check("lower-case codes, horizon 14, config", r["ok"] and r["config"]["predict_ahead_days"] == 14 and len(r["flights"]) > 0)
r2 = U.us_search("LAX", "SFO", "2026-11-15", config={"predict_ahead_days": 99}); check("horizon clamped to 14", r2["config"]["predict_ahead_days"] == 14)
r3 = U.us_search("LAX", "SFO", "2026-10-09"); print("near-departure flights (dbd 9 at listing):", len(r3["flights"]), [len(f["curve"]) for f in r3["flights"][:3]])
check("unknown flight id -> None", U.us_curve("deadbeef") is None)
print("\nFAILED:" if fails else "\nALL PASSED", fails)
sys.exit(1 if fails else 0)
