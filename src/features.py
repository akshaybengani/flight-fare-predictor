"""Feature engineering shared by training and the web app."""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

CATS = ["airline", "source", "destination", "class", "dep_bucket", "arr_bucket"]
BASE_NUMS = ["stops", "duration_h", "days_left", "dow", "month"]

# lat, lon of the metros in the training data
COORDS = {"Delhi": (28.61, 77.21), "Mumbai": (19.08, 72.88), "Bangalore": (12.97, 77.59),
          "Chennai": (13.08, 80.27), "Hyderabad": (17.39, 78.49), "Kolkata": (22.57, 88.36),
          "Ahmedabad": (23.02, 72.57)}

# target-encoding keys: (name, columns)
TE_KEYS = [("te_flight", ["flight_code", "class"]),
           ("te_route_airline", ["source", "destination", "airline", "class"])]


def distance_km(src, dst):
    a = np.radians(src.map(lambda c: COORDS[c][0]).astype(float)); b = np.radians(dst.map(lambda c: COORDS[c][0]).astype(float))
    lon = np.radians(src.map(lambda c: COORDS[c][1]).astype(float) - dst.map(lambda c: COORDS[c][1]).astype(float))
    return 6371 * np.arccos(np.clip(np.sin(a) * np.sin(b) + np.cos(a) * np.cos(b) * np.cos(lon), -1, 1))


def add_basic(df):
    df = df.copy()
    df["distance_km"] = distance_km(df.source, df.destination)
    return df


class Encoder:
    """Fit on training rows only. Target encodings are out-of-fold on the training set to avoid leakage."""

    def __init__(self, nums, te=(), supply=False):
        self.nums, self.te, self.supply = list(nums), list(te), supply

    @property
    def features(self):
        extra = list(self.te) + (["route_supply"] if self.supply else [])
        return CATS + self.nums + extra

    def _stats(self, df):
        y = np.log(df.price)
        out = {}
        for name, cols in TE_KEYS:
            if name in self.te:
                out[name] = y.groupby([df[c] for c in cols]).mean()
        return out

    def _apply(self, df, stats):
        res = {}
        for name, cols in TE_KEYS:
            if name in self.te:
                idx = pd.MultiIndex.from_frame(df[cols]) if len(cols) > 1 else pd.Index(df[cols[0]])
                res[name] = stats[name].reindex(idx).values
        return res

    def fit_transform(self, df):
        self.cats = {c: sorted(df[c].unique()) for c in CATS}
        self.stats = self._stats(df)
        te = {n: np.full(len(df), np.nan) for n in self.te}
        for tr, va in KFold(5, shuffle=True, random_state=0).split(df):
            part = self._apply(df.iloc[va], self._stats(df.iloc[tr]))
            for n in te:
                te[n][va] = part[n]
        # supply: number of distinct flights per route/class (competition proxy)
        self.supply_map = df.groupby(["source", "destination", "class"]).flight_code.nunique()
        return self._build(df, te)

    def transform(self, df):
        return self._build(df, self._apply(df, self.stats))

    def _build(self, df, te):
        X = df[CATS + self.nums].copy()
        for c in CATS:
            X[c] = pd.Categorical(X[c], categories=self.cats[c]).codes
        for n in self.te:
            X[n] = te[n]
        if self.supply:
            idx = pd.MultiIndex.from_frame(df[["source", "destination", "class"]])
            X["route_supply"] = self.supply_map.reindex(idx).values
        return X[self.features]

    @property
    def cat_idx(self):
        return [self.features.index(c) for c in CATS]
