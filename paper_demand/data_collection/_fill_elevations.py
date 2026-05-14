"""Vectorised elevation + roughness fill for the IMD-international cities."""
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from paper_demand.data_collection.imd_international import (
    get_elevation_batch, CACHE_DIR
)

IMD_INTL_DIR = Path("/home/rohanfosse/Bureau/Recherche/imd-national-catalogue/paper_demand/data_collection/imd_international")


def compute_roughness_vectorised(lat: np.ndarray, lon: np.ndarray,
                                  elev: np.ndarray, k: int = 5) -> np.ndarray:
    """For each station i, compute the std of elevations among its k nearest neighbours."""
    n = len(lat)
    # Pairwise squared euclidean (in lat/lon space) — good enough for KNN ranking
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    d2 = dlat**2 + dlon**2
    np.fill_diagonal(d2, np.inf)
    knn_idx = np.argpartition(d2, k, axis=1)[:, :k]
    rough = np.array([np.std(elev[knn_idx[i]]) for i in range(n)])
    return np.nan_to_num(rough, nan=0.0)


def main():
    cities = sys.argv[1:] if len(sys.argv) > 1 else [
        "chicago_divvy", "boston_bluebikes", "sf_baywheels", "nyc_citibike"
    ]
    for c in cities:
        p = IMD_INTL_DIR / f"{c}.parquet"
        if not p.exists():
            print(f"  ✗ {c} : no file")
            continue
        df = pd.read_parquet(p)
        n = len(df)
        t0 = time.time()
        coords = list(zip(df["lat"].astype(float), df["lng"].astype(float)))
        elevs = get_elevation_batch(coords)
        df["elevation_m"] = elevs

        lat = df["lat"].astype(float).values
        lng = df["lng"].astype(float).values
        elev = df["elevation_m"].astype(float).fillna(0).values
        df["topography_roughness_index"] = compute_roughness_vectorised(
            lat, lng, elev, k=5
        )
        df.to_parquet(p, index=False)
        filled = df["elevation_m"].notna().sum()
        print(f"  ✓ {c:25s}  {filled}/{n} elev  "
              f"mean_elev={df['elevation_m'].mean():.1f}m  "
              f"mean_rough={df['topography_roughness_index'].mean():.2f}  "
              f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
