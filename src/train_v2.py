"""Ablation ladder for the external features (v2). Does NOT touch the v1 model files.

    python src/train_v2.py ladder            # fast-GBM ablation ladder, all rungs x 3 scenarios -> models/experiments_v2.json
    python src/train_v2.py best <rung>       # full train.py GBM on one rung: 3 scenarios, final model, permutation importance
Scenarios and metric are the same as train.py (A same-period random 20%, E 2022 + 10% of 2023 flight-days, C 2022 -> 2023).
"""
import sys
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor

from features import Encoder, add_basic
from train import NUMS, TE, GBM, score
from enrich import groups, nonconstant

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
FAST = dict(GBM, learning_rate=0.1, max_iter=1500)  # ladder only: ~2x faster than the final config, same for every rung
ALL4 = ["calendar", "weather_lag", "fuel_fx", "airport_static"]
RUNGS = {
    "baseline": [], "baseline_seed1": [],  # seed1 = noise floor (only GBM random_state changes)
    "+calendar": ["calendar"], "+weather_lag": ["weather_lag"], "+weather_actual": ["weather_actual"],
    "+fuel_fx": ["fuel_fx"], "+airport_static": ["airport_static"],
    "+all_booking_time": ALL4, "+all": ALL4 + ["weather_actual"],
}


def fit(train, nums, te, gbm):
    enc = Encoder(nums, te=te, supply=True)
    X = enc.fit_transform(train)
    m = HistGradientBoostingRegressor(categorical_features=enc.cat_idx, **gbm).fit(X, np.log(train.price))
    return m, enc


def load():
    df = add_basic(pd.read_parquet(ROOT / "data/enriched.parquet"))
    return df[df["class"] != "First"].reset_index(drop=True)


def scenarios(df):
    """yield (name, train, test, base_nums, te) exactly as train.py builds them"""
    d22, d23 = df[df.period == 2022], df[df.period == 2023]
    te_idx = d23.sample(frac=0.2, random_state=0).index
    yield "A_same_period_random_20pct", d23.drop(te_idx), d23.loc[te_idx], NUMS, TE
    rng = np.random.RandomState(0)
    keys = d23.flight_code + d23.journey_date.astype(str)
    pick = set(rng.choice(keys.unique(), size=int(keys.nunique() * 0.1), replace=False))
    mk = keys.isin(pick)
    yield "E_2022_plus_10pct_of_2023", pd.concat([d22, d23[mk]]), d23[~mk], NUMS, TE
    yield "C_train_2022_test_2023", d22, d23, [n for n in NUMS if n not in ("period", "dep_hour", "arr_hour")], ["te_route_airline"]


def run_rung(df, rung, gbm, keep=None):
    g = groups(df.columns)
    new = [c for grp in RUNGS[rung] for c in g[grp]]
    res = {"n_new_features_requested": len(new)}
    for name, tr, te_, nums, te in scenarios(df):
        use = nonconstant(tr, new)  # a column that never varies in training cannot help; drop it
        t = time.time()
        m, enc = fit(tr, nums + use, te, dict(gbm, random_state=1) if rung == "baseline_seed1" else gbm)
        res[name] = dict(score(m, enc, te_), new_features_used=len(use), seconds=round(time.time() - t))
        print(rung, name, json.dumps(res[name]), flush=True)
        if keep is not None:
            keep[name] = (m, enc, te_, use)
    return res


def ladder():
    out_p = ROOT / "models/experiments_v2.json"
    out = json.loads(out_p.read_text()) if out_p.exists() else {"config": "fast GBM: lr 0.1, max_iter 1500 (else = train.py GBM)", "rungs": {}}
    df = load()
    for rung in RUNGS:
        if rung in out["rungs"]:
            continue
        out["rungs"][rung] = run_rung(df, rung, FAST)
        out_p.write_text(json.dumps(out, indent=2))
    print("LADDER DONE", flush=True)


def acc(model, X, price):
    return 100 - (np.abs(np.exp(model.predict(X)) - price) / price).mean() * 100


def perm_importance(model, enc, test, use, n=30000, reps=3):
    """Held-out permutation importance: accuracy points lost when a feature (or a whole group) is shuffled."""
    t = test.sample(min(n, len(test)), random_state=1)
    X, price = enc.transform(t), t.price.values
    base = acc(model, X, price)
    rng = np.random.RandomState(0)

    def drop(cols):
        d = []
        for _ in range(reps):
            Xp = X.copy()
            idx = rng.permutation(len(Xp))  # shuffle rows jointly so a group keeps its internal correlation
            Xp[cols] = Xp[cols].values[idx]
            d.append(base - acc(model, Xp, price))
        return round(float(np.mean(d)), 3)

    g = groups(list(X.columns))
    feats = sorted(({"feature": c, "accuracy_drop_pts": drop([c])} for c in use), key=lambda r: -r["accuracy_drop_pts"])
    grp = {k: drop([c for c in v if c in X.columns]) for k, v in g.items() if any(c in X.columns for c in v)}
    return {"baseline_accuracy_pct": round(float(base), 2), "sample_rows": len(t), "group_drop_pts": grp, "new_features": feats}


def best(rung):
    df = load()
    keep = {}
    res = run_rung(df, rung, GBM, keep)
    metrics = {k: v for k, v in res.items() if k[0] in "ACE" and k[1] == "_"}
    metrics["rung"] = rung
    (ROOT / "models/metrics_v2.json").write_text(json.dumps(metrics, indent=2))
    imp = {"rung": rung, "note": "held-out permutation importance (rows the model never trained on), 3 shuffles"}
    for name in ("A_same_period_random_20pct", "E_2022_plus_10pct_of_2023", "C_train_2022_test_2023"):
        m, enc, te_, use = keep[name]
        imp[name] = perm_importance(m, enc, te_, use)
    (ROOT / "models/importance_v2.json").write_text(json.dumps(imp, indent=2))
    print("importance written", flush=True)
    g = groups(df.columns)
    use = nonconstant(df, [c for grp in RUNGS[rung] for c in g[grp]])
    m, enc = fit(df, NUMS + use, TE, GBM)
    joblib.dump({"model": m, "enc": enc, "nums": NUMS + use, "rung": rung}, ROOT / "models/fare_model_v2.joblib")
    print("DONE", flush=True)


if __name__ == "__main__":
    {"ladder": ladder, "best": lambda: best(sys.argv[2])}[sys.argv[1]]()
