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
