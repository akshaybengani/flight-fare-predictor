"""Self-test for the switchable engines: load every engine in this interpreter, predict a sample frame, check shapes,
finiteness, price range and a business > economy sanity, and print timing. Engines whose library is not installed are
reported as SKIPPED (not failed). Exit code 1 if any available engine fails.

    python src/engines_selftest.py
"""
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engines  # noqa: E402
from features import add_basic  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = engines.ROOT


def route_frame(df, src, dst, klass, days_left=21, dow=3, month=12, period=None):
    rows = df[(df.source == src) & (df.destination == dst) & (df["class"] == klass) & (df.period == df.period.max())]
    rows = rows.drop_duplicates("flight_code").copy()
    rows["days_left"], rows["dow"], rows["month"], rows["period"] = days_left, dow, month, int(df.period.max() if period is None else period)
    return rows


def explain_checks(df, ENC):
    """explain(): steps add up to the price (<= 1 rupee), finite, ordered, fast, plausible directions; reference_values()."""
    import time as _t
    fails = 0
    rv = engines.reference_values()
    names = [f["name"] for f in rv["features"]]
    try:
        assert names == list(ENC.features), "reference_values() order differs from ENC.features"
        assert all(f["type"] in ("number", "category") and f["label"] for f in rv["features"])
        assert all(f.get("categories") for f in rv["features"] if f["type"] == "category")
        assert [f["name"] for f in rv["features"] if f["derived"]] == ["te_flight", "te_route_airline", "route_supply"]
        print(f"\nreference_values OK: {len(names)} features in model order; derived = te_flight, te_route_airline, route_supply")
    except AssertionError as ex:
        fails += 1
        print(f"reference_values FAILED: {ex}")
    rows = ENC.transform(add_basic(df.sample(20, random_state=7)))
    big = ENC.transform(add_basic(df.sample(400, random_state=5)))
    print(f"\n{'explain':14} {'status':8} {'ms/row':>7} {'max |sum-price|':>16}  detail")
    for e in engines.list_engines():
        key = e["key"]
        if not e["available"]:
            continue
        try:
            eng = engines.load_engine(key)
            engines.explain(key, rows.iloc[[0]])  # warm-up (builds caches)
            ts, errs = [], []
            for i in range(len(rows)):
                one = rows.iloc[[i]]
                t = _t.perf_counter()
                out = engines.explain(key, one)
                ts.append(_t.perf_counter() - t)
                steps = out["steps"]
                assert out["method"] == ("shap" if key == "xgboost" else "sensitivity"), out["method"]
                assert 1 <= len(steps) <= 13, len(steps)
                vals = [out["base_price"], out["price"]] + [s["effect_inr"] for s in steps]
                assert np.isfinite(vals).all(), "NaN/inf in explanation"
                errs.append(abs(out["base_price"] + sum(s["effect_inr"] for s in steps) - out["price"]))
                assert abs(out["price"] - float(np.exp(eng.predict(one)[0]))) < 0.02, "price differs from the engine's prediction"
                known = [s["effect_inr"] for s in steps if s["feature"] != "other"]
                assert known == sorted(known, key=abs, reverse=True) or key == "xgboost", "steps not ordered by |effect|"
            assert max(errs) <= 1.0, f"steps do not add up (max error {max(errs):.2f} rupees)"
            limit = 0.1 if key == "xgboost" else 0.5
            assert float(np.median(ts)) <= limit, f"too slow: median {np.median(ts) * 1000:.0f} ms (limit {limit * 1000:.0f})"
            a, b = big.copy(), big.copy()
            a["days_left"], b["days_left"] = 30, 3
            ratio = np.exp(eng.predict(a)).mean() / np.exp(eng.predict(b)).mean()
            assert ratio < 1.05, f"booking 30 days ahead is {ratio:.2f}x the price of 3 days ahead (earlier should not cost more)"
            print(f"{key:14} {'OK':8} {np.median(ts) * 1000:7.0f} {max(errs):16.2e}  method={out['method']}, 30d/3d price ratio {ratio:.2f}")
        except Exception as ex:
            fails += 1
            print(f"{key:14} {'FAILED':8} {'':>7} {'':>16}  {type(ex).__name__}: {ex}")
    return fails


def main():
    bundle = joblib.load(ROOT / "models/fare_model.joblib")
    ENC = bundle["enc"]
    df = add_basic(pd.read_parquet(ROOT / "data/merged.parquet"))
    df = df[df["class"] != "First"].reset_index(drop=True)
    routes = [("Delhi", "Mumbai"), ("Mumbai", "Bangalore"), ("Delhi", "Hyderabad"), ("Chennai", "Kolkata")]
    frames = {}
    for s, d in routes:
        for k in ("Economy", "Business"):
            r = route_frame(df, s, d, k)
            if len(r):
                frames[(s, d, k)] = ENC.transform(add_basic(r))
    sample = frames[("Delhi", "Mumbai", "Economy")]
    batch = ENC.transform(df.sample(5000, random_state=3))
    print(f"engines manifest: {engines.MANIFEST}\nsample search: Delhi -> Mumbai Economy, {len(sample)} flights; batch {len(batch)} rows\n")
    hdr = f"{'engine':14} {'status':8} {'load s':>7} {'ms/1k':>8} {'mean fare':>10} {'biz/eco':>8}  detail"
    print(hdr)
    fails = 0
    ref_mean = {}
    for e in engines.list_engines():
        key = e["key"]
        if not e["available"]:
            print(f"{key:14} {'SKIPPED':8} {'':>7} {'':>8} {'':>10} {'':>8}  {e['unavailable_reason']}")
            continue
        try:
            engines.clear_cache()
            t = time.perf_counter()
            eng = engines.load_engine(key)
            load_s = time.perf_counter() - t
            assert engines.load_engine(key) is eng, "cache must return the same object"
            out = eng.predict(sample)
            assert isinstance(out, np.ndarray) and out.shape == (len(sample),), f"bad shape {getattr(out, 'shape', None)}"
            assert np.isfinite(out).all(), "NaN/inf in sample search"
            price = np.exp(out)
            assert 300 < price.min() and price.max() < 300_000, f"implausible fares {price.min():.0f}..{price.max():.0f}"
            best = 1e9
            for _ in range(2):
                t = time.perf_counter()
                ob = eng.predict(batch)
                best = min(best, time.perf_counter() - t)
            assert ob.shape == (len(batch),) and np.isfinite(ob).all(), "batch prediction not finite"
            ratios = []
            for s, d in routes:
                if (s, d, "Economy") in frames and (s, d, "Business") in frames:
                    ratios.append(np.exp(eng.predict(frames[(s, d, "Business")])).mean() / np.exp(eng.predict(frames[(s, d, "Economy")])).mean())
            assert ratios and all(r > 1 for r in ratios), f"business not above economy: {ratios}"
            ref = ref_mean.setdefault("v", price.mean() if key == "hgb" else None)
            warn = ""
            if ref and not (0.5 < price.mean() / ref < 2.0):
                warn = f"  [WARN: sample fare {price.mean() / ref:.1f}x the HGB engine; this engine extrapolates badly outside the training months (Jan-Apr)]"
            print(f"{key:14} {'OK':8} {load_s:7.2f} {best * 1e6 / len(batch):8.1f} {price.mean():10.0f} {np.mean(ratios):8.2f}  "
                  f"fares {price.min():.0f}-{price.max():.0f}{warn}")
        except Exception as ex:
            fails += 1
            print(f"{key:14} {'FAILED':8} {'':>7} {'':>8} {'':>10} {'':>8}  {type(ex).__name__}: {ex}")
    # ---- explain() and reference_values()
    fails += explain_checks(df, ENC)
    # cache bound check
    engines.clear_cache()
    for e in engines.list_engines():
        if e["available"]:
            engines.load_engine(e["key"])
    assert len(engines._cache) <= engines.MAX_CACHED, "cache exceeded MAX_CACHED"
    # helpful error for an unknown engine
    try:
        engines.load_engine("nope")
        fails += 1
        print("unknown engine did not raise")
    except engines.EngineUnavailable as ex:
        print(f"\nunknown-engine error OK: {ex}")
    print("\nRESULT:", "all available engines passed" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
