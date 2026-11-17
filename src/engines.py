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
