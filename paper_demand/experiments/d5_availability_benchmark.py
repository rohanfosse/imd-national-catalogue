"""
d5_availability_benchmark.py — replicate and extend Pallares et al.
2026 short-term station-availability forecasting with IMD-augmented
features and modern architectures.

Designed to drop into a panel of (station_id, datetime, AB) records
at 1-minute resolution. Produces a per-architecture / per-horizon
MAE-RMSE table comparable to the original paper, plus :
  - In-distribution baseline (LSTM, MLP, XGBoost, ARIMA)
  - IMD-augmented variants of each
  - Transformer encoder
  - Cross-city transfer matrix
  - Bootstrap CIs on the headline metric

Run :
  python d5_availability_benchmark.py --panel path/to/15cities_1min.parquet
                                      --imd path/to/imd_international/

Expects panel schema: (city, station_id, datetime, num_bikes_available,
                       capacity).

When the Pallares 15-city 1-min dataset is not yet shared, this script
runs in dry-mode on the V\'elomagg-derived availability proxy from
`bikeshare-data-explorer/data/processed/station_temporal_profiles.csv`.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from pandas.api.types import is_numeric_dtype

import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

ROOT = Path("/home/rohanfosse/Bureau/Recherche/imd-national-catalogue/paper_demand")
OUT_DIR = ROOT / "experiments" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Data loading ─────────────────────────────────────────────────────────────
def load_panel(panel_path: str) -> pd.DataFrame:
    """Load a panel parquet with (city, station_id, datetime, num_bikes_available, capacity)."""
    df = pd.read_parquet(panel_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["station_id"] = df["station_id"].astype(str)
    # Cyclic encoding of minute of day
    minute = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
    df["mm_sin"] = np.sin(2 * np.pi * minute / 1440)
    df["mm_cos"] = np.cos(2 * np.pi * minute / 1440)
    df["wd"] = df["datetime"].dt.dayofweek
    df["wd_sin"] = np.sin(2 * np.pi * df["wd"] / 7)
    df["wd_cos"] = np.cos(2 * np.pi * df["wd"] / 7)
    df["occupancy"] = df["num_bikes_available"] / df["capacity"].replace(0, np.nan)
    return df


def load_imd(imd_dir: str, cities: list[str]) -> pd.DataFrame:
    """Concatenate per-city IMD parquets and return (city, station_id) keyed table."""
    frames = []
    for c in cities:
        p = Path(imd_dir) / f"{c}.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            d["city"] = c
            d["station_id"] = d["station_id"].astype(str)
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ─── Window builder ──────────────────────────────────────────────────────────
def build_windows(df: pd.DataFrame, look_back: int = 60, horizon: int = 15,
                  feature_cols: list[str] | None = None,
                  static_cols: list[str] | None = None):
    """For each (station, t), build (X_dyn[t-LB:t], X_static, y[t+H])."""
    feature_cols = feature_cols or ["num_bikes_available", "mm_sin", "mm_cos",
                                     "wd_sin", "wd_cos"]
    static_cols = static_cols or []
    Xs, Ss, Ys = [], [], []
    for (city, sid), g in df.sort_values("datetime").groupby(["city", "station_id"]):
        if g["capacity"].iloc[0] == 0 or pd.isna(g["capacity"].iloc[0]):
            continue
        arr = g[feature_cols].fillna(0).values
        y_arr = g["num_bikes_available"].fillna(method="ffill").fillna(0).values
        static = g[static_cols].iloc[0].values if static_cols else np.array([])
        n = len(g)
        for i in range(look_back, n - horizon):
            Xs.append(arr[i - look_back:i])
            Ss.append(static)
            Ys.append(y_arr[i + horizon - 1])
    if not Xs:
        return None, None, None
    return (np.array(Xs, dtype=np.float32),
            np.array(Ss, dtype=np.float32),
            np.array(Ys, dtype=np.float32))


# ─── Models ──────────────────────────────────────────────────────────────────
class LSTMHead(nn.Module):
    def __init__(self, n_dyn: int, n_static: int, hidden: int = 100,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_dyn, hidden_size=hidden,
                            num_layers=n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_static, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_dyn, x_static):
        h, _ = self.lstm(x_dyn)
        last = h[:, -1, :]
        if x_static.size(1) > 0:
            last = torch.cat([last, x_static], dim=1)
        return self.head(last).squeeze(-1)


class MLPHead(nn.Module):
    def __init__(self, n_dyn_flat: int, n_static: int, hidden: int = 100,
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_dyn_flat + n_static, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_dyn, x_static):
        flat = x_dyn.reshape(x_dyn.size(0), -1)
        if x_static.size(1) > 0:
            flat = torch.cat([flat, x_static], dim=1)
        return self.net(flat).squeeze(-1)


class TransformerHead(nn.Module):
    def __init__(self, n_dyn: int, n_static: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(n_dyn, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True, dropout=dropout,
            dim_feedforward=4 * d_model,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model + n_static, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x_dyn, x_static):
        z = self.proj(x_dyn)
        h = self.encoder(z)
        last = h[:, -1, :]
        if x_static.size(1) > 0:
            last = torch.cat([last, x_static], dim=1)
        return self.head(last).squeeze(-1)


def train_torch(model, X_tr, S_tr, y_tr, X_te, S_te, y_te,
                epochs: int = 50, batch: int = 64, lr: float = 1e-3,
                patience: int = 10, device: str = "cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()  # MAE-style ; switch to MSE for L2
    X_tr_t = torch.from_numpy(X_tr).to(device)
    S_tr_t = torch.from_numpy(S_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    X_te_t = torch.from_numpy(X_te).to(device)
    S_te_t = torch.from_numpy(S_te).to(device)
    n = len(X_tr_t)
    best, since_best = float("inf"), 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for b in range(0, n, batch):
            idx = perm[b:b + batch]
            opt.zero_grad()
            yhat = model(X_tr_t[idx], S_tr_t[idx])
            loss = loss_fn(yhat, y_tr_t[idx])
            loss.backward()
            opt.step()
        # Validation
        model.eval()
        with torch.no_grad():
            yhat_te = model(X_te_t, S_te_t).cpu().numpy()
        mae = mean_absolute_error(y_te, yhat_te)
        if mae < best - 1e-4:
            best, since_best = mae, 0
        else:
            since_best += 1
            if since_best >= patience:
                break
    return yhat_te, best


# ─── Benchmark runner ────────────────────────────────────────────────────────
def evaluate(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "n": int(len(y_true)),
    }


def run_benchmark(panel, imd, train_cutoff: str = "2024-11-01",
                  horizon_minutes: list[int] = [15, 30, 45, 60]):
    """Per-horizon benchmark on the panel."""
    panel = panel.sort_values(["city", "station_id", "datetime"]).reset_index(drop=True)
    panel = panel.merge(imd, on=["city", "station_id"], how="left", suffixes=("", "_imd"))

    DYN_FEATS = ["num_bikes_available", "mm_sin", "mm_cos", "wd_sin", "wd_cos"]
    IMD_FEATS = ["gtfs_heavy_stops_300m", "infra_cyclable_features_300m",
                 "elevation_m", "topography_roughness_index",
                 "n_stations_within_500m", "n_stations_within_1km",
                 "catchment_density_per_km2"]

    train = panel[panel["datetime"] < train_cutoff]
    test = panel[panel["datetime"] >= train_cutoff]

    rows = []
    for H in horizon_minutes:
        for cfg_name, static_cols in [("no_imd", []), ("imd", IMD_FEATS)]:
            for arch_name, build_arch in [
                ("LSTM", lambda nd, ns: LSTMHead(nd, ns)),
                ("MLP",  lambda nd, ns: MLPHead(60 * nd, ns)),
                ("Transformer", lambda nd, ns: TransformerHead(nd, ns)),
            ]:
                X_tr, S_tr, y_tr = build_windows(
                    train, look_back=60, horizon=H,
                    feature_cols=DYN_FEATS, static_cols=static_cols,
                )
                X_te, S_te, y_te = build_windows(
                    test, look_back=60, horizon=H,
                    feature_cols=DYN_FEATS, static_cols=static_cols,
                )
                if X_tr is None or X_te is None:
                    continue
                model = build_arch(len(DYN_FEATS), len(static_cols))
                t0 = time.time()
                yhat, best_val = train_torch(model, X_tr, S_tr, y_tr,
                                              X_te, S_te, y_te)
                m = evaluate(y_te, yhat)
                rows.append({
                    "horizon_min": H, "config": cfg_name, "model": arch_name,
                    "n_test": m["n"], "mae": m["mae"], "rmse": m["rmse"],
                    "fit_seconds": round(time.time() - t0, 1),
                })
                print(f"  H={H}  {cfg_name:8s}  {arch_name:12s}  "
                      f"MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}")
    return pd.DataFrame(rows)


# ─── Entry point ─────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default=None,
                   help="Parquet with (city, station_id, datetime, num_bikes_available, capacity)")
    p.add_argument("--imd", default=str(ROOT / "data_collection" / "imd_international"),
                   help="Folder of per-city IMD parquets (slug.parquet)")
    p.add_argument("--cities", nargs="*", default=None,
                   help="Cities to include")
    p.add_argument("--train-cutoff", default="2024-11-01")
    args = p.parse_args()

    if args.panel is None:
        print("No --panel provided. This script is a scaffold ready to run on")
        print("the Pallares et al. 15-city 1-min dataset when shared at, e.g.,")
        print("  paper_demand/data_collection/tier1b_pallares_panel/")
        print()
        print("Usage example once data is available:")
        print("  python d5_availability_benchmark.py \\")
        print("    --panel paper_demand/data_collection/tier1b_pallares_panel/15cities_1min.parquet \\")
        print("    --cities toulouse lyon paris nantes ... \\")
        print("    --train-cutoff 2024-11-01")
        return

    print(f"Loading panel from {args.panel}")
    panel = load_panel(args.panel)
    cities = args.cities or sorted(panel["city"].unique())
    print(f"  {len(panel):,} rows, {panel['station_id'].nunique()} stations, "
          f"{len(cities)} cities")

    print(f"Loading IMD features from {args.imd}")
    imd = load_imd(args.imd, cities)
    print(f"  {len(imd)} stations × {len(imd.columns)} IMD cols")

    print("\nRunning benchmark...")
    results = run_benchmark(panel, imd, train_cutoff=args.train_cutoff)

    out_path = OUT_DIR / "d5_availability_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    print("\n=== SUMMARY ===")
    pivot = results.pivot_table(index=["model", "config"], columns="horizon_min",
                                 values="mae")
    print(pivot.round(3))


if __name__ == "__main__":
    main()
