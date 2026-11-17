"""MOCK of the engines module, for building the engine switch while models/engines/ does not exist.

Used only when FARESIGHT_MOCK=1 and the real `engines` module cannot list engines (see app.py). Every mock engine
wraps the deployed model and bends its log-price output by a small deterministic amount, so switching changes the
numbers. The manifest numbers are placeholders, not results; the UI labels them "Sample data".
"""
import hashlib
import time

import numpy as np

_base = {"model": None}


def set_base(model):
    _base["model"] = model


_ENGINES = [
    ("hgb", "HistGradientBoosting", "Gradient boosting", "The deployed model.", 44, 0.9, 31, (93.0, 87.0, 66.2), 0.0, None),
    ("xgboost", "XGBoost", "Gradient boosting", "Boosted trees with level-wise growth.", 30, 0.6, 41, (93.4, 87.4, 67.7), 0.03, None),
    ("lightgbm", "LightGBM", "Gradient boosting", "Leaf-wise boosted trees, fast to train.", 12, 0.3, 26, (93.2, 87.5, 65.8), -0.02, None),
    ("catboost", "CatBoost", "Gradient boosting", "Boosting with built-in handling of categories.", 25, 0.4, 35, (93.0, 87.2, 67.4), 0.04, "missing Python package(s): catboost (sample reason)"),
    ("random_forest", "Random Forest", "Bagged trees", "Many independent deep trees, averaged. Requested by the portfolio manager.", 880, 4.5, 880, (92.0, 85.9, 60.2), -0.08, None),
    ("extra_trees", "Extra Trees", "Bagged trees", "Like a random forest with random split points.", 1200, 6.0, 1200, (91.6, 85.4, 58.9), 0.07, None),
    ("ridge", "Ridge regression", "Linear", "A straight-line model, the simplest baseline.", 1, 0.05, 3, (76.2, 70.5, 52.0), 0.15, None),
    ("mlp", "Neural net (MLP)", "Neural network", "A small multilayer perceptron.", 3, 0.2, 180, (85.6, 78.2, 57.3), -0.12, "model file models/engines/mlp.joblib is missing (sample reason)"),
]


def list_engines():
    """Same shape as the real manifest entries: accuracy / within10 / mae dicts keyed by scenario A, E, C."""
    out = []
    for key, label, fam, desc, mb, load_s, ms, (a, e, c), _bias, reason in _ENGINES:
        out.append({"key": key, "label": label, "family": fam, "description": desc, "size_mb": mb, "load_seconds": load_s,
                    "predict_ms_per_1k": ms, "accuracy": {"A": a, "E": e, "C": c},
                    "within10": {"A": round(a - 15, 1), "E": round(e - 31, 1), "C": round(c - 46, 1)},
                    "mae": {"A": 1500 + int((93 - a) * 60), "E": 2700 + int((87 - e) * 60), "C": 5900 + int((66 - c) * 60)},
                    "notes": "Sample numbers for layout only.", "available": reason is None, "unavailable_reason": reason})
    return out


class _MockEngine:
    def __init__(self, key, bias):
        self.key, self.bias = key, bias
        self.label = next(e[1] for e in _ENGINES if e[0] == key)

    def predict(self, feats):
        base = np.asarray(_base["model"].predict(feats), dtype=float)
        h = np.array([int(hashlib.md5(f"{self.key}{i}".encode()).hexdigest()[:4], 16) / 65535 - .5 for i in range(len(base))])
        return base + self.bias + h * abs(self.bias) * 0.6


def load_engine(key="hgb"):
    for k, _l, _f, _d, _m, load_s, _ms, _sc, bias, reason in _ENGINES:
        if k == key:
            if reason:
                raise RuntimeError(reason)
            time.sleep(min(load_s, 1.2))      # make the loading state visible
            return _MockEngine(key, bias)
    raise RuntimeError(f"unknown engine {key!r}")
