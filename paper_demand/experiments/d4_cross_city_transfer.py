"""
d4_cross_city_transfer.py — train an IMD-augmented model on city A,
test it on city B. Measures the transferability of the IMD-based
demand model from one network to another.

For each ordered pair (train_city, test_city) :
  * Build the (station, hour) demand panel for both cities
  * Engineer temporal features
  * Merge per-station IMD features computed from OSM / Open-Elevation
  * Train LightGBM on the train city's data
  * Evaluate on the test city's hold-out (last 20%)

The transferability score is the test R² achieved on city B using
a model trained on city A's IMD-augmented features. We report :
  - In-distribution R² (train on A's train, test on A's test)
  - Transfer R² (train on A's full, test on B's full)
  - The transfer gap, ΔR² = R²_in - R²_transfer

A transfer gap close to 0 indicates the IMD-4 framework captures
features that generalise across regulatory and geographic contexts.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error
from pandas.api.types import is_numeric_dtype

import sys
sys.path.insert(0, str(Path(__file__).parent))
from d3_multicity_benchmark import (
    build_panel, add_temporal_features, FEATS_T, FEATS_IMD,
    IMD_INTL_DIR, OUT_DIR, _encode, evaluate,
)

warnings.filterwarnings("ignore")

LYFT_CITIES = ["dc_capitalbikeshare", "chicago_divvy", "boston_bluebikes",
               "sf_baywheels"]


def load_city(city: str) -> pd.DataFrame:
    """Load + merge IMD + add temporal features for one city."""
    imd_path = IMD_INTL_DIR / f"{city}.parquet"
    imd = pd.read_parquet(imd_path)
    imd["station_id"] = imd["station_id"].astype(str)
    panel = build_panel(city)
    if panel.empty:
        return pd.DataFrame()
    df = panel.merge(imd[["station_id"] + FEATS_IMD], on="station_id", how="left")
    df = df.dropna(subset=FEATS_IMD).copy()
    df = add_temporal_features(df)
    df["city"] = city
    return df


def train_lgb(train, features):
    Xt, _ = _encode(train, train, features)
    cat_cols = [c for c in features if not is_numeric_dtype(train[c])]
    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        min_child_samples=30, reg_lambda=0.5, random_state=42,
        n_jobs=-1, verbose=-1,
    )
    model.fit(Xt, train["log_demand"].values,
              categorical_feature=cat_cols if cat_cols else "auto")
    return model


def predict(model, df, features):
    _, Xv = _encode(df, df, features)
    return model.predict(Xv)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cities", nargs="*", default=LYFT_CITIES)
    args = p.parse_args()

    # Load all cities once
    panels = {}
    for c in args.cities:
        print(f"\nLoading {c}...")
        df = load_city(c)
        if df.empty:
            print(f"  ✗ empty")
            continue
        panels[c] = df
        print(f"  ✓ {len(df):,} bins, {df['station_id'].nunique()} stations")

    features = FEATS_T + FEATS_IMD
    results = []

    # In-distribution baseline + cross-city transfers
    for train_city, test_city in product(panels.keys(), panels.keys()):
        if train_city == test_city:
            # in-distribution split
            df = panels[train_city].sort_values("datetime_hour").reset_index(drop=True)
            cut = int(0.8 * len(df))
            train, test = df.iloc[:cut], df.iloc[cut:]
            mode = "in_distribution"
        else:
            train = panels[train_city]
            test = panels[test_city]
            mode = "transfer"

        t0 = time.time()
        model = train_lgb(train, features)
        y_pred = predict(model, test, features)
        m = evaluate(f"{train_city}_to_{test_city}",
                     test["log_demand"].values, y_pred)
        m["mode"] = mode
        m["train_city"] = train_city
        m["test_city"] = test_city
        m["fit_seconds"] = round(time.time() - t0, 1)
        results.append(m)
        print(f"  {mode:18s}  {train_city:25s} -> {test_city:25s}  "
              f"R²={m['r2_trips']:+.4f}  MAE={m['mae_trips']:.3f}")

    summary = pd.DataFrame(results)
    summary.to_csv(OUT_DIR / "d4_transfer_results.csv", index=False)
    print(f"\n=== TRANSFER MATRIX ===")
    pivot = summary.pivot(index="train_city", columns="test_city", values="r2_trips")
    print(pivot.round(3))
    print(f"\nSaved {OUT_DIR / 'd4_transfer_results.csv'}")


if __name__ == "__main__":
    main()
