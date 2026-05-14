"""
tier2_pseudo_flow.py — convert GBFS station_status snapshots to
hourly pseudo-trip counts per station, for the multi-city French
extension of the IMD-augmented demand benchmark.

Source : the bikeshare-data-explorer/data/status_snapshots/ directory
populated by an active polling daemon (one parquet per day per city,
with fields fetched_at, station_id, num_bikes_available,
num_docks_available, is_renting, is_returning).

Method : for each (station, hour) bin, sum the negative deltas in
num_bikes_available between consecutive polls. A drop from N to N-k
bikes between two polls a few minutes apart is treated as k checkouts.
This is a conservative pseudo-trip estimator -- it underestimates
trip volume when bikes are checked out and returned within the same
polling interval, and is contaminated by operator rebalancing.

Output : one Parquet per city under
  paper_demand/data_collection/tier2_pseudo_flow/<city>.parquet
with columns (station_id, datetime_hour, pseudo_trips, n_observations).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

SNAPSHOTS_ROOT = Path("/home/rohanfosse/Bureau/Recherche/bikeshare-data-explorer/data/status_snapshots")
OUT_ROOT = Path(__file__).parent / "tier2_pseudo_flow"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def list_cities(min_days: int = 1) -> list[str]:
    """Return city directories with at least `min_days` of daily snapshots."""
    if not SNAPSHOTS_ROOT.exists():
        return []
    cities = []
    for sub in sorted(SNAPSHOTS_ROOT.iterdir()):
        if not sub.is_dir():
            continue
        day_files = [p for p in sub.glob("*.parquet")
                     if p.name != "station_info.parquet"]
        if len(day_files) >= min_days:
            cities.append(sub.name)
    return cities


def load_city_snapshots(city: str) -> pd.DataFrame | None:
    """Concatenate all daily snapshot parquets for one city."""
    city_dir = SNAPSHOTS_ROOT / city
    if not city_dir.exists():
        return None
    frames = []
    for path in sorted(city_dir.glob("*.parquet")):
        if path.name == "station_info.parquet":
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            print(f"      [warn] {path.name}: {type(e).__name__}: {e}")
            continue
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    return df


def compute_pseudo_flow(df: pd.DataFrame, hourly_floor: int = 0) -> pd.DataFrame:
    """For each (station, hour), estimate the pseudo-trip count from
    consecutive Δ num_bikes_available. Negative Δ ⇒ checkout.
    """
    df = df.copy()
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    df = df.sort_values(["station_id", "fetched_at"]).reset_index(drop=True)

    # Compute per-station Δ
    df["bikes_prev"] = df.groupby("station_id")["num_bikes_available"].shift(1)
    df["delta_bikes"] = df["num_bikes_available"] - df["bikes_prev"]

    # Time-gap filter: only count deltas within a "reasonable" gap (< 60 min)
    df["time_prev"] = df.groupby("station_id")["fetched_at"].shift(1)
    df["gap_min"] = (df["fetched_at"] - df["time_prev"]).dt.total_seconds() / 60
    valid = df["gap_min"] < 60

    # Checkout estimator: count only negative deltas (bikes leaving)
    df["pseudo_checkouts"] = np.where(
        valid & (df["delta_bikes"] < 0),
        -df["delta_bikes"],
        0,
    )

    # Bin by (station, hour)
    df["datetime_hour"] = df["fetched_at"].dt.floor("h")
    out = (
        df.groupby(["station_id", "datetime_hour"], as_index=False)
        .agg(pseudo_trips=("pseudo_checkouts", "sum"),
             n_observations=("fetched_at", "count"))
    )
    if hourly_floor:
        out = out[out["pseudo_trips"] >= hourly_floor]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cities", nargs="*", default=None,
                   help="City slugs (folder names under status_snapshots/). Default: all.")
    p.add_argument("--min-days", type=int, default=1,
                   help="Minimum number of daily snapshots required to include a city")
    args = p.parse_args()

    all_cities = list_cities(min_days=args.min_days)
    if args.cities:
        selected = [c for c in all_cities if c in args.cities]
    else:
        selected = all_cities

    if not selected:
        print(f"No cities to process. Available with ≥{args.min_days} days: {all_cities}")
        return

    print(f"Processing {len(selected)} cities → {OUT_ROOT}")
    print(f"Source root : {SNAPSHOTS_ROOT}\n")

    summary = []
    for city in selected:
        t0 = time.time()
        df = load_city_snapshots(city)
        if df is None or df.empty:
            print(f"  ✗ {city}: no snapshots")
            continue

        pf = compute_pseudo_flow(df)
        out_path = OUT_ROOT / f"{city}.parquet"
        pf.to_parquet(out_path, index=False)
        elapsed = time.time() - t0
        summary.append({
            "city": city,
            "n_raw": int(len(df)),
            "n_stations": int(df["station_id"].nunique()),
            "n_hour_bins": int(len(pf)),
            "pseudo_trips_total": int(pf["pseudo_trips"].sum()),
            "pseudo_trips_mean_per_hour": float(pf["pseudo_trips"].mean()),
            "snapshot_span_h": float(
                (pd.to_datetime(df["fetched_at"], utc=True).max() -
                 pd.to_datetime(df["fetched_at"], utc=True).min()).total_seconds() / 3600
            ),
            "elapsed_s": round(elapsed, 1),
        })
        s = summary[-1]
        print(f"  ✓ {city:25s}  stations={s['n_stations']:4d}  "
              f"hours={s['n_hour_bins']:6d}  trips={s['pseudo_trips_total']:7d}  "
              f"span={s['snapshot_span_h']:.0f}h  ({s['elapsed_s']}s)")

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT_ROOT / "_summary.csv", index=False)
    print(f"\nWrote {len(summary_df)} city files + _summary.csv to {OUT_ROOT}")
    print(f"Total pseudo-trips across all cities: {summary_df['pseudo_trips_total'].sum():,}")


if __name__ == "__main__":
    main()
