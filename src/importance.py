"""Permutation importance of the saved model: how many accuracy points are lost when one feature is shuffled."""
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from features import add_basic

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]


def accuracy(model, X, price):
    pred = np.exp(model.predict(X))
    return 100 - (np.abs(pred - price) / price).mean() * 100


def main():
    b = joblib.load(ROOT / "models/fare_model.joblib")
    model, enc = b["model"], b["enc"]
    df = add_basic(pd.read_parquet(ROOT / "data/merged.parquet"))
    df = df[(df["class"] != "First") & (df.period == 2023)].sample(20000, random_state=1)
    X, price = enc.transform(df), df.price.values
    base = accuracy(model, X, price)
    rng = np.random.RandomState(0)
    out = []
    for c in X.columns:
        drops = []
        for _ in range(2):
            Xp = X.copy()
            Xp[c] = rng.permutation(Xp[c].values)
            drops.append(base - accuracy(model, Xp, price))
        out.append({"feature": c, "accuracy_drop_pts": round(float(np.mean(drops)), 2)})
    out.sort(key=lambda r: -r["accuracy_drop_pts"])
    (ROOT / "models/importance.json").write_text(json.dumps({"baseline_accuracy_pct": round(float(base), 2),
                                                             "sample_rows": len(df), "features": out}, indent=2))
    print("DONE", base)


if __name__ == "__main__":
    main()
