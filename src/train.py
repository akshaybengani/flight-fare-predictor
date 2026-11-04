"""Train and evaluate the fare model under several splits (same-period -> cross-period)."""
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor

from features import Encoder, add_basic

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
NUMS = ["stops", "duration_h", "days_left", "dow", "month", "period", "dep_hour", "arr_hour", "distance_km"]
TE = ["te_flight", "te_route_airline"]
GBM = dict(max_iter=3500, learning_rate=0.05, max_leaf_nodes=255, min_samples_leaf=5, l2_regularization=1.0,
           early_stopping=True, validation_fraction=0.1, n_iter_no_change=40, random_state=0)


def fit(train, nums=NUMS, te=TE):
    enc = Encoder(nums, te=te, supply=True)
    X = enc.fit_transform(train)
    # model log-price so errors are relative (an INR 500 miss matters less on a business fare)
    m = HistGradientBoostingRegressor(categorical_features=enc.cat_idx, **GBM).fit(X, np.log(train.price))
    return m, enc


def score(m, enc, test):
    pred = np.exp(m.predict(enc.transform(test)))
    y = test.price.values
    ape = np.abs(pred - y) / y
    return {"rows": len(test), "MAE_inr": round(float(np.abs(pred - y).mean())),
            "accuracy_pct": round(100 - ape.mean() * 100, 2), "within_10pct": round((ape <= 0.10).mean() * 100, 1)}


def main(final_only=False):
    df = add_basic(pd.read_parquet(ROOT / "data/merged.parquet"))
    df = df[df["class"] != "First"].reset_index(drop=True)
    d22, d23 = df[df.period == 2022], df[df.period == 2023]
    res = {}

    if final_only:  # skip the evaluation scenarios; keep the committed metrics.json
        m, enc = fit(df)
        joblib.dump({"model": m, "enc": enc, "nums": NUMS}, ROOT / "models/fare_model.joblib")
        print("DONE (final model only)", flush=True)
        return

    # A: random 80/20 of 2023 rows (same period; other days of the same flights are in training)
    te_idx = d23.sample(frac=0.2, random_state=0).index
    m, enc = fit(d23.drop(te_idx)); res["A_same_period_random_20pct"] = score(m, enc, d23.loc[te_idx])
    print("A_same_period_random_20pct", json.dumps(res["A_same_period_random_20pct"]), flush=True)

    # E: recalibration. 2022 plus a random 10% of 2023 flight-days; test on the other 2023 flight-days
    rng = np.random.RandomState(0)
    keys = d23.flight_code + d23.journey_date.astype(str)
    pick = set(rng.choice(keys.unique(), size=int(keys.nunique() * 0.1), replace=False))
    mk = keys.isin(pick)
    m, enc = fit(pd.concat([d22, d23[mk]])); res["E_2022_plus_10pct_of_2023"] = score(m, enc, d23[~mk])
    print("E_2022_plus_10pct_of_2023", json.dumps(res["E_2022_plus_10pct_of_2023"]), flush=True)

    # C: cross-period. train 2022 only, test all of 2023 (no period feature possible, no 2023 flights seen)
    # (2022 has no exact times, so those columns are dropped here)
    m, enc = fit(d22, nums=[n for n in NUMS if n not in ("period", "dep_hour", "arr_hour")], te=["te_route_airline"])
    res["C_train_2022_test_2023"] = score(m, enc, d23)
    print("C_train_2022_test_2023", json.dumps(res["C_train_2022_test_2023"]), flush=True)

    # final model: all real data
    m, enc = fit(df)
    joblib.dump({"model": m, "enc": enc, "nums": NUMS}, ROOT / "models/fare_model.joblib")
    (ROOT / "models/metrics.json").write_text(json.dumps(res, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    import sys
    (ROOT / "models").mkdir(exist_ok=True)
    main(final_only="--final-only" in sys.argv)
