"""Switchable prediction engines (algorithms) behind one interface.

    from engines import list_engines, load_engine
    eng = load_engine("xgboost")
    log_price = eng.predict(feats)      # feats = the DataFrame from ENC.transform(...) (columns = ENC.features)
    price = np.exp(log_price)

Every engine takes the SAME encoded frame the app already builds with the deployed encoder and does its own
frame conversion (native categoricals, one-hot + scaling, NaN fill) internally. Engines are loaded lazily, cached
(at most MAX_CACHED in memory, least-recently-used evicted) and loading is thread-safe. Heavy libraries
(xgboost / lightgbm / catboost) are imported only when that engine is loaded; a clear EngineUnavailable error
says what is missing. See docs/ENGINES.md.
"""
import ctypes
import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "models" / "engines"
MANIFEST = ENGINE_DIR / "manifest.json"
MAX_CACHED = 2
CATS = ["airline", "source", "destination", "class", "dep_bucket", "arr_bucket"]  # same as features.CATS
DEFAULT_ENGINE = "xgboost"
# python modules that must be importable to LOAD each engine (besides numpy, pandas, scikit-learn, joblib)
EXTRA_LIBS = {"xgboost": ["xgboost"], "lightgbm": ["lightgbm"], "catboost": ["catboost"]}
PIP_NAMES = {"xgboost": "xgboost", "lightgbm": "lightgbm", "catboost": "catboost"}


_omp_done = False


def _preload_libomp():
    """macOS: the xgboost / lightgbm wheels link @rpath/libomp.dylib, which exists only if Homebrew's libomp is installed
    (/opt/homebrew/opt/libomp). Without it, we reuse the libomp that scikit-learn ships in sklearn/.dylibs. Its install
    name is not '@rpath/libomp.dylib', so a plain preload would not satisfy the later lookup; we therefore make a copy
    with the install name rewritten (needs Xcode command-line tools: install_name_tool, codesign) once into a cache dir
    and load THAT by absolute path before xgboost is imported. This works inside a running Flask process (no env vars
    or re-exec). No-op on Linux (wheels need libgomp1, normally present) and when Homebrew's libomp exists."""
    global _omp_done
    if _omp_done or sys.platform != "darwin":
        return
    _omp_done = True
    if any(Path(p).exists() for p in ("/opt/homebrew/opt/libomp/lib/libomp.dylib", "/usr/local/opt/libomp/lib/libomp.dylib")):
        return
    spec = importlib.util.find_spec("sklearn")
    if not spec or not spec.submodule_search_locations:
        return
    src = Path(list(spec.submodule_search_locations)[0]) / ".dylibs" / "libomp.dylib"
    if not src.exists():
        return
    shim = Path.home() / ".cache" / "flight-poc" / "libomp_shim" / "libomp.dylib"
    try:
        if not shim.exists():
            shim.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, shim)
            subprocess.run(["install_name_tool", "-id", "@rpath/libomp.dylib", str(shim)], check=True, capture_output=True)
            subprocess.run(["codesign", "-f", "-s", "-", str(shim)], check=True, capture_output=True)
        ctypes.CDLL(str(shim), mode=ctypes.RTLD_GLOBAL)
    except (OSError, subprocess.SubprocessError):
        pass  # the import below then fails with xgboost's own message; docs/ENGINES.md says: brew install libomp


class EngineUnavailable(RuntimeError):
    """The engine's file is missing or a library it needs is not installed."""


_lock = threading.RLock()
_cache = OrderedDict()


def _read_manifest():
    if not MANIFEST.exists():
        raise EngineUnavailable(f"{MANIFEST} not found. Build the engines with: python src/build_engines.py")
    return json.loads(MANIFEST.read_text())


def _missing_libs(key):
    return [m for m in EXTRA_LIBS.get(key, []) if importlib.util.find_spec(m) is None]


def list_engines():
    """Manifest entries (list of dicts) plus 'available' / 'unavailable_reason' for THIS interpreter."""
    out = []
    for e in _read_manifest()["engines"]:
        e = dict(e)
        path = ROOT / e["file"]
        miss = _missing_libs(e["key"])
        reason = None
        if not path.exists():
            reason = f"model file {e['file']} is missing (run python src/build_engines.py --only {e['key']})"
        elif miss:
            reason = "missing Python package(s): " + ", ".join(PIP_NAMES.get(m, m) for m in miss) + f" (pip install {' '.join(miss)})"
        e["available"], e["unavailable_reason"] = reason is None, reason
        out.append(e)
    return out


def _xgb_frame(X, cat_sizes):
    X = X.copy()
    for c in CATS:
        X[c] = pd.Categorical.from_codes(X[c].values.astype(int), categories=list(range(cat_sizes[c])))
    return X


def _cat_frame(X):
    X = X.copy()
    for c in CATS:
        X[c] = X[c].astype(int)
    return X


class Engine:
    """Wraps a fitted model + the frame conversion it needs. predict(feats) -> predicted LOG price (ndarray)."""

    def __init__(self, key, label, kind, model, features, cat_sizes, load_seconds=None, clip=None):
        self.key, self.label, self.kind, self.model = key, label, kind, model
        self.clip = clip or {}  # {column: [min, max]} seen in training; linear models must not extrapolate
        self.features, self.cat_sizes, self.load_seconds = list(features), cat_sizes, load_seconds

    def predict(self, feats):
        X = feats[self.features] if isinstance(feats, pd.DataFrame) else pd.DataFrame(np.asarray(feats), columns=self.features)
        for c, (lo, hi) in self.clip.items():
            if c in X.columns:
                X = X.assign(**{c: X[c].clip(lo, hi)})  # e.g. a December search is treated like the nearest trained month (April)
        k = self.kind
        if k in ("hgb", "lightgbm", "ridge", "mlp"):
            out = self.model.predict(X)
        elif k == "xgboost":
            out = self.model.predict(_xgb_frame(X, self.cat_sizes))
        elif k == "catboost":
            out = self.model.predict(_cat_frame(X))
        elif k in ("random_forest", "extra_trees"):
            out = self.model.predict(X.fillna(-1))
        else:
            raise EngineUnavailable(f"unknown engine kind {k!r}")
        return np.asarray(out, dtype=float)

    __call__ = predict


def _load(key):
    entry = next((e for e in _read_manifest()["engines"] if e["key"] == key), None)
    if entry is None:
        raise EngineUnavailable(f"unknown engine {key!r}; known: {[e['key'] for e in _read_manifest()['engines']]}")
    path = ROOT / entry["file"]
    if not path.exists():
        raise EngineUnavailable(f"engine {key!r}: model file {entry['file']} is missing. Rebuild: python src/build_engines.py --only {key}")
    miss = _missing_libs(key)
    if miss:
        raise EngineUnavailable(f"engine {key!r} needs Python package(s) not installed here: {', '.join(miss)} (pip install {' '.join(miss)}). "
                                "On macOS lightgbm/xgboost also need libomp (brew install libomp); on Linux libgomp (apt install libgomp1).")
    if key in EXTRA_LIBS:
        _preload_libomp()
    t = time.perf_counter()
    try:
        payload = joblib.load(path)
    except ModuleNotFoundError as e:
        raise EngineUnavailable(f"engine {key!r} could not be unpickled, missing module: {e.name}") from e
    except Exception as e:  # version mismatch etc.
        raise EngineUnavailable(f"engine {key!r} failed to load ({type(e).__name__}: {e}). Engines must be loaded with the library versions "
                                "they were built with (see docs/ENGINES.md).") from e
    if key == "hgb":  # the deployed bundle {"model", "enc", "nums"}: wrap it, features come from its encoder
        feats, sizes = payload["enc"].features, {}
        model = payload["model"]
    else:
        feats, sizes, model = payload["features"], payload.get("cat_sizes", {}), payload["model"]
    return Engine(key, entry["label"], key, model, feats, sizes, load_seconds=time.perf_counter() - t,
                  clip=None if key == "hgb" else payload.get("clip"))


def load_engine(key=DEFAULT_ENGINE):
    """Return the cached engine, loading it if needed (thread-safe; at most MAX_CACHED engines stay in memory)."""
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
        eng = _load(key)
        _cache[key] = eng
        while len(_cache) > MAX_CACHED:
            _cache.popitem(last=False)
        return eng


def clear_cache():
    with _lock:
        _cache.clear()


# ------------------------------------------------------------------------------------------------ explanations
REF_PARQUET = ENGINE_DIR / "reference.parquet"
REF_JSON = ENGINE_DIR / "reference_values.json"
TOP_STEPS = 12
_ref_cache = {}
_base_cache = {}


def reference_values():
    """Per-feature facts for building editable input fields, in the order the model receives them (ENC.features).

    Returns {"order_note", "target", "reference_sample", "features": [...]}. Each feature has name, order, label (plain English),
    description (one sentence; reuses FEATURE_DOCS from src/app.py, read-only, at build time), type ("number" or "category"),
    derived (True for te_flight, te_route_airline, route_supply: typical price levels and route supply computed from the
    training data, not something a traveller types), and
      category: categories (names; the model frame holds the position in this list as the code) and typical (most common)
      number:   min, max (observed in training), typical (median), integer (True if whole numbers); te_* also carry
                typical_price_inr / min_price_inr / max_price_inr because their raw unit is the natural log of a price.
    Built by `python src/build_engines.py --build-reference`."""
    if not REF_JSON.exists():
        raise EngineUnavailable(f"{REF_JSON} not found. Build it with: python src/build_engines.py --build-reference")
    with _lock:
        if "values" not in _ref_cache:
            _ref_cache["values"] = json.loads(REF_JSON.read_text())
    return json.loads(json.dumps(_ref_cache["values"]))


def _reference():
    """(reference frame, {feature: median or mode code}, {feature: info}) from the fixed 2,000-row training sample."""
    with _lock:
        if "ref" not in _ref_cache:
            if not REF_PARQUET.exists():
                raise EngineUnavailable(f"{REF_PARQUET} not found. Build it with: python src/build_engines.py --build-reference")
            ref = pd.read_parquet(REF_PARQUET)
            info = {f["name"]: f for f in reference_values()["features"]}
            fill = {}
            for c in ref.columns:
                fill[c] = float(ref[c].mode().iloc[0]) if info[c]["type"] == "category" else float(ref[c].median())
            _ref_cache["ref"] = (ref, fill, info)
        return _ref_cache["ref"]


def _display(info, value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if info["type"] == "category":
        i = int(value)
        return info["categories"][i] if 0 <= i < len(info["categories"]) else None
    return int(value) if info.get("integer") and float(value).is_integer() else round(float(value), 4)


def _step(f, info, value, effect):
    s = {"feature": f, "label": info["label"], "value": _display(info, value), "effect_inr": effect}
    if f.startswith("te_") and s["value"] is not None:
        s["value_display"] = f"about Rs {np.exp(float(value)):,.0f} usual fare"
    return s


def _finish(method, base, price, parts, info, X, extra=None):
    """parts: [(feature or None, effect)] -> steps sorted by |effect|, top TOP_STEPS + 'all other features', rounded to
    paise so that base_price + sum(effect_inr) == price to within 0.01 rupee."""
    parts = sorted(parts, key=lambda p: -abs(p[1]))
    top, rest = parts[:TOP_STEPS], parts[TOP_STEPS:]
    steps = [_step(f, info[f], X[f].iloc[0], e) for f, e in top]
    if rest:
        steps.append({"feature": "other", "label": f"All {len(rest)} other features", "value": None, "effect_inr": sum(e for _, e in rest)})
    base, price = round(float(base), 2), round(float(price), 2)
    for s in steps:
        s["effect_inr"] = round(float(s["effect_inr"]), 2)
    resid = round(price - base - sum(s["effect_inr"] for s in steps), 2)
    if steps and resid:  # rounding crumbs go to the smallest step
        steps[-1]["effect_inr"] = round(steps[-1]["effect_inr"] + resid, 2)
    out = {"method": method, "base_price": base, "price": price, "steps": steps}
    out.update(extra or {})
    return out


def _reference_mean_price(eng):
    with _lock:
        if eng.key not in _base_cache:
            ref = _reference()[0]
            _base_cache[eng.key] = float(np.exp(eng.predict(ref)).mean())
        return _base_cache[eng.key]


_pool = None


def _shap_chunked(eng, booster, dm, n_it):
    """Exact TreeSHAP for one row, ~4x faster: contributions are additive over trees, so the trees are split into chunks
    scored in parallel threads on sliced copies of the booster (xgboost releases the GIL); the bias column carries a constant per call, measured once
    against a single full call and subtracted."""
    global _pool
    from concurrent.futures import ThreadPoolExecutor
    k = max(1, min(4, (__import__("os").cpu_count() or 2) // 2))
    if k == 1:
        return booster.predict(dm, pred_contribs=True, iteration_range=(0, n_it))[0]
    if _pool is None:
        _pool = ThreadPoolExecutor(k)
    if not hasattr(eng, "_shap_slices"):  # sliced copies of the booster (contributions only support a range starting at tree 0)
        edges = np.linspace(0, n_it, k + 1).astype(int)
        eng._shap_slices = [booster[int(x):int(y)] for x, y in zip(edges[:-1], edges[1:])]
    parts = list(_pool.map(lambda bs: bs.predict(dm, pred_contribs=True)[0], eng._shap_slices))
    out = np.sum(parts, axis=0)
    if not hasattr(eng, "_bias_fix"):  # one-time calibration of the per-call constant
        full = booster.predict(dm, pred_contribs=True, iteration_range=(0, n_it))[0]
        eng._bias_fix = float(full[-1] - out[-1])
        assert np.allclose(full[:-1], out[:-1], atol=1e-4), "chunked SHAP disagrees with the full call"
    out = out.copy()
    out[-1] += eng._bias_fix
    return out


def explain(engine_key, feats):
    """Explain one prediction in rupees. `feats` = one-row DataFrame in the format the app feeds to engines (ENC.features columns).

    Returns {"method": "shap" | "sensitivity", "base_price", "price", "steps": [{feature, label, value, effect_inr[, value_display]}], ...};
    steps are ordered by absolute effect, truncated to the top 12 plus an "all other features" step, and
    base_price + sum(effect_inr) == price (to within 0.01 rupee).

    XGBoost -> method "shap": exact tree contributions (Booster.predict(pred_contribs=True)) in log-price space. They are
      converted to rupees by a multiplicative walk starting at exp(bias), features taken in order of |log contribution|:
      effect_k = exp(bias + c_1..c_k) - exp(bias + c_1..c_(k-1)). base_price is exp(bias), the model's baseline price (about the
      typical training fare; geometric rather than arithmetic mean); the engine's arithmetic mean over the reference sample is
      returned separately as "reference_mean_price". Exact for this model: sum of contributions + bias = the model's log price.
    Every other engine -> method "sensitivity" (a what-if, not exact attribution, and it ignores interactions): for each feature,
      replace its value by the median (numbers) or mode (categories) of the fixed 2,000-row reference sample of training rows,
      re-predict, and take price - price_without_that_feature as its effect. The raw effects rarely add up to price - base_price
      (base_price = the engine's mean predicted price over the reference sample), so the gap is spread over the features in
      proportion to |effect| (identical to rescaling when all effects agree in sign). Cheap: one batched predict of 19 rows.
    Categories are handled in each engine's own frame conversion because we always go through Engine.predict()."""
    eng = load_engine(engine_key)
    ref, fill, info = _reference()
    if not isinstance(feats, pd.DataFrame) or len(feats) < 1:
        raise ValueError("feats must be a one-row DataFrame with the model's feature columns")
    X = feats[eng.features].iloc[[0]].reset_index(drop=True)
    if eng.kind == "xgboost":
        import xgboost as xgb
        booster = eng.model.get_booster()
        bi = getattr(eng.model, "best_iteration", None)
        kw = {"iteration_range": (0, int(bi) + 1)} if bi is not None else {}
        dm = xgb.DMatrix(_xgb_frame(X, eng.cat_sizes), enable_categorical=True)
        n_it = int(bi) + 1 if bi is not None else int(booster.num_boosted_rounds())
        contrib = _shap_chunked(eng, booster, dm, n_it)
        bias, c = float(contrib[-1]), contrib[:-1]
        order = np.argsort(-np.abs(c))
        parts, cum, prev = [], bias, float(np.exp(bias))
        for i in order:
            cum += float(c[i])
            cur = float(np.exp(cum))
            parts.append((eng.features[i], cur - prev))
            prev = cur
        price = float(np.exp(eng.predict(X)[0]))  # the engine's own price; any float32 crumb (about 1 rupee) is absorbed by the smallest step
        return _finish("shap", np.exp(bias), price, parts, info, X, {"reference_mean_price": round(_reference_mean_price(eng), 2),
                                                                   "unit_note": "effects are in rupees, applied one after another (multiplicative walk), not independent"})
    rows = [X.iloc[0].copy()]
    for f in eng.features:
        r = X.iloc[0].copy()
        r[f] = fill[f]
        rows.append(r)
    batch = pd.DataFrame(rows, columns=eng.features).reset_index(drop=True)
    for c in batch.columns:
        batch[c] = batch[c].astype(X[c].dtype) if X[c].dtype.kind in "iuf" and not batch[c].isna().any() else batch[c]
    prices = np.exp(eng.predict(batch))
    price, raw = float(prices[0]), prices[0] - prices[1:]
    base = _reference_mean_price(eng)
    gap = (price - base) - float(raw.sum())
    w = np.abs(raw)
    adj = raw + gap * (w / w.sum() if w.sum() > 0 else np.full(len(raw), 1.0 / len(raw)))
    return _finish("sensitivity", base, price, list(zip(eng.features, map(float, adj))), info, X)
