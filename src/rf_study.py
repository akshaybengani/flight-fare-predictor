"""Random Forest deep-dive (requested by the portfolio manager): size/accuracy trade-offs and feature importances.
Uses the same scenario A data/encoder/metric as compare_models.py. Writes models/rf_study.json. Run after compare_models.py:
    python src/rf_study.py
"""
import json, os, time, warnings
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from compare_models import ROOT, NT, load_df, prepare, metrics, model_mb

warnings.filterwarnings("ignore")
base = ROOT / "models/algo_runs/baseline18"
tuned = json.loads((base / "random_forest_A.json").read_text())["params"]
df = load_df("baseline18", False)
data = prepare("A", df, "baseline18")
tr, te = data["tr"], data["te"]
Xtr, Xte = tr.X.fillna(-1), te.X.fillna(-1)
CONFIGS = [("reference (tuned, 120 trees)", dict(n_estimators=120, **tuned)),
           ("60 trees", dict(n_estimators=60, **tuned)),
           ("30 trees", dict(n_estimators=30, **tuned)),
           ("min_samples_leaf=5", dict(n_estimators=120, **dict(tuned, min_samples_leaf=5))),
           ("min_samples_leaf=10", dict(n_estimators=120, **dict(tuned, min_samples_leaf=10))),
           ("min_samples_leaf=20", dict(n_estimators=120, **dict(tuned, min_samples_leaf=20))),
           ("max_samples=0.5", dict(n_estimators=120, max_samples=0.5, **tuned)),
           ("max_samples=0.25", dict(n_estimators=120, max_samples=0.25, **tuned)),
           ("30 trees + leaf 10 + max_samples 0.5", dict(n_estimators=30, max_samples=0.5, **dict(tuned, min_samples_leaf=10)))]
out = {"tuned_params_A": tuned, "n_train": len(tr.y), "n_test": len(te.y), "tradeoffs": []}
ref = None
for name, p in CONFIGS:
    t = time.time()
    m = RandomForestRegressor(n_jobs=NT, random_state=0, **p).fit(Xtr, tr.y)
    ts = time.time() - t
    s = metrics(np.exp(m.predict(Xte)), te.price)
    row = {"config": name, "params": p, "accuracy": s["accuracy"], "within10": s["within10"], "mae": s["mae"],
           "train_seconds": round(ts, 1), "model_mb": round(model_mb(m), 1)}
    out["tradeoffs"].append(row); print(row, flush=True)
    if ref is None:
        ref = m
        # impurity importance + held-out permutation importance (accuracy points lost) on a 30k test sample
        out["impurity_importance"] = dict(sorted(zip(Xtr.columns, map(lambda v: round(float(v), 4), m.feature_importances_)), key=lambda kv: -kv[1]))
        idx = np.random.RandomState(1).choice(len(te.y), 30000, replace=False)
        Xs, ps = Xte.iloc[idx], te.price[idx]
        b = metrics(np.exp(m.predict(Xs)), ps)["accuracy"]
        rng = np.random.RandomState(0); perm = {}
        for c in Xs.columns:
            d = []
            for _ in range(2):
                Xp = Xs.copy(); Xp[c] = Xp[c].values[rng.permutation(len(Xp))]
                d.append(b - metrics(np.exp(m.predict(Xp)), ps)["accuracy"])
            perm[c] = round(float(np.mean(d)), 3)
        out["permutation_importance_accuracy_pts"] = dict(sorted(perm.items(), key=lambda kv: -kv[1]))
    del m
    (ROOT / "models/rf_study.json").write_text(json.dumps(out, indent=2))
print("RF STUDY DONE")
