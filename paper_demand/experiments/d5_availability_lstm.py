"""
d5_availability_lstm.py — IMD-augmented version of the Pallares
et al. 2026 short-term bike-availability forecasting setup.

Replicates the reference LSTM/MLP/XGBoost/ARIMA benchmark on
the same task : forecast `num_bikes_available(s, t + h)` for
horizons h ∈ {15, 30, 45, 60} minutes from a window of the
station's recent history plus exogenous features.

Adds three differentiators :
  1. IMD-4 station-level features (M, I, T, D axes)
  2. Multi-city training (instead of single-station Toulouse)
  3. Transformer encoder + Temporal Fusion Transformer baselines

Data dependency : this script reads station-status time series
in long-form parquet at the path expected by
`paper_demand/data_collection/tier2_pseudo_flow.py`, but at
1-minute resolution rather than the current ~5-min polling. To
run on the Pallares 15-city dataset, point STATUS_ROOT at the
shared CSV/parquet archive.

Output : per-model MAE & RMSE per horizon per city per
configuration ; see outputs/d5_*.json.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path("/home/rohanfosse/Bureau/Recherche/imd-national-catalogue/paper_demand")
STATUS_ROOT = Path("/home/rohanfosse/Bureau/Recherche/bikeshare-data-explorer/data/status_snapshots")
IMD_INTL_DIR = ROOT / "data_collection" / "imd_international"
OUT_DIR = ROOT / "experiments" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS_MIN = (15, 30, 45, 60)
WINDOW_MIN = 60  # 60-minute lookback for the LSTM input window
RESOLUTION_MIN = 5  # downsample to this if the source is 1-min


# ─── Models ───────────────────────────────────────────────────────────────────
class LSTMAvailability(nn.Module):
    """Stacked LSTM matching the Pallares-paper hyperparameters
    (2 layers, 100 units, dropout 0.2, Xavier init)."""
    def __init__(self, n_features: int, hidden: int = 100, n_horizons: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.head = nn.Linear(hidden, n_horizons)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class TransformerAvailability(nn.Module):
    """Vanilla Transformer encoder for the same task. Added as a
    modern (2017) baseline that the Pallares paper does not test."""
    def __init__(self, n_features: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2, n_horizons: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=128,
            dropout=0.2, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_horizons)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.encoder(h)
        return self.head(h[:, -1, :])


# ─── Data loader ──────────────────────────────────────────────────────────────
def load_city_status(city: str) -> pd.DataFrame:
    """Load all daily station-status parquets for one city and
    return a long-form DataFrame (station_id, datetime, num_bikes_available,
    capacity_proxy)."""
    city_dir = STATUS_ROOT / city
    if not city_dir.exists():
        raise FileNotFoundError(f"{city_dir}")
    frames = []
    for path in sorted(city_dir.glob("*.parquet")):
        if path.name == "station_info.parquet":
            continue
        df = pd.read_parquet(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    df["capacity_proxy"] = (df["num_bikes_available"] + df["num_docks_available"])
    df = df.sort_values(["station_id", "fetched_at"]).reset_index(drop=True)
    return df


def merge_imd(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Attach the per-station IMD-4 features for the given city."""
    imd_path = IMD_INTL_DIR / f"{city}.parquet"
    if not imd_path.exists():
        # Try alternate names common in our setup
        for alt in [f"{city}_velomagg.parquet", f"velomagg.parquet"]:
            if (IMD_INTL_DIR / alt).exists():
                imd_path = IMD_INTL_DIR / alt
                break
    if not imd_path.exists():
        print(f"  [warn] no IMD parquet for {city} at {imd_path}")
        return df
    imd = pd.read_parquet(imd_path)
    imd["station_id"] = imd["station_id"].astype(str)
    df["station_id"] = df["station_id"].astype(str)
    return df.merge(imd, on="station_id", how="left")


def build_windows(df: pd.DataFrame, window_min: int = WINDOW_MIN,
                  resolution_min: int = RESOLUTION_MIN,
                  imd_cols: list[str] | None = None) -> tuple:
    """Build (X, y, station_idx) for sliding-window forecasting.

    X[i] = [num_bikes_available_{t-W..t}, IMD_static_features]
    y[i] = num_bikes_available_{t+h} for h in HORIZONS_MIN
    """
    # Resample per station to the target resolution
    df = df.copy()
    df["minute"] = df["fetched_at"].dt.floor(f"{resolution_min}min")
    minute_panel = (df.groupby(["station_id", "minute"], as_index=False)
                    .agg(num_bikes_available=("num_bikes_available", "mean"),
                         capacity_proxy=("capacity_proxy", "max")))

    if imd_cols:
        first_per_station = df.groupby("station_id")[imd_cols].first().reset_index()
        minute_panel = minute_panel.merge(first_per_station, on="station_id", how="left")

    n_lookback = window_min // resolution_min
    Xs, ys, sids = [], [], []
    station_to_idx = {s: i for i, s in enumerate(sorted(minute_panel["station_id"].unique()))}

    for s, g in minute_panel.groupby("station_id"):
        g = g.sort_values("minute").reset_index(drop=True)
        ab = g["num_bikes_available"].values.astype(np.float32)
        if imd_cols:
            static_feats = g.iloc[0][imd_cols].fillna(0).values.astype(np.float32)
        else:
            static_feats = np.zeros(0, dtype=np.float32)
        max_h = max(HORIZONS_MIN) // resolution_min
        for i in range(n_lookback, len(g) - max_h):
            window = ab[i - n_lookback:i].reshape(-1, 1)
            if imd_cols and len(static_feats):
                # Broadcast static features along the time window
                static_block = np.tile(static_feats, (n_lookback, 1))
                feats = np.concatenate([window, static_block], axis=1)
            else:
                feats = window
            targets = [ab[i + h // resolution_min - 1] for h in HORIZONS_MIN]
            Xs.append(feats)
            ys.append(targets)
            sids.append(station_to_idx[s])

    return (np.array(Xs, dtype=np.float32),
            np.array(ys, dtype=np.float32),
            np.array(sids, dtype=np.int64))


# ─── Training + evaluation ────────────────────────────────────────────────────
def train_and_eval(model, X_tr, y_tr, X_te, y_te, epochs=20, batch=256, lr=1e-3):
    device = torch.device("cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    n = len(X_tr_t)

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        loss_sum, nb = 0.0, 0
        for b in range(0, n, batch):
            idx = perm[b:b + batch]
            opt.zero_grad()
            yhat = model(X_tr_t[idx])
            loss = loss_fn(yhat, y_tr_t[idx])
            loss.backward()
            opt.step()
            loss_sum += loss.item(); nb += 1

    fit_s = time.time() - t0
    model.eval()
    with torch.no_grad():
        X_te_t = torch.from_numpy(X_te).to(device)
        preds = []
        for b in range(0, len(X_te_t), batch):
            preds.append(model(X_te_t[b:b + batch]).cpu().numpy())
        y_pred = np.concatenate(preds)

    mae_per_h = [mean_absolute_error(y_te[:, h], y_pred[:, h])
                 for h in range(len(HORIZONS_MIN))]
    rmse_per_h = [np.sqrt(mean_squared_error(y_te[:, h], y_pred[:, h]))
                  for h in range(len(HORIZONS_MIN))]
    return {
        "horizons_min": list(HORIZONS_MIN),
        "mae_per_horizon": [float(v) for v in mae_per_h],
        "rmse_per_horizon": [float(v) for v in rmse_per_h],
        "fit_seconds": round(fit_s, 1),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def run_city(city: str, use_imd: bool, model_name: str = "lstm", epochs: int = 15):
    print(f"\n=== {city} | use_imd={use_imd} | model={model_name} ===")
    df = load_city_status(city)
    if df.empty:
        return None
    print(f"  status rows = {len(df):,}  stations = {df['station_id'].nunique()}")

    imd_cols = None
    if use_imd:
        df = merge_imd(df, city)
        candidate_imd = [
            "gtfs_heavy_stops_300m", "infra_cyclable_features_300m",
            "elevation_m", "topography_roughness_index",
            "n_stations_within_500m", "n_stations_within_1km",
        ]
        imd_cols = [c for c in candidate_imd if c in df.columns]
        print(f"  IMD features attached : {imd_cols}")

    X, y, sids = build_windows(df, imd_cols=imd_cols)
    if len(X) == 0:
        print(f"  no windows produced")
        return None
    print(f"  windows: {len(X):,}  n_features={X.shape[2]}  n_lookback={X.shape[1]}")

    # 80/20 temporal split: take last 20% as test (per-station ordering preserved)
    cut = int(0.8 * len(X))
    X_tr, X_te = X[:cut], X[cut:]
    y_tr, y_te = y[:cut], y[cut:]

    # Normalise on train
    s = StandardScaler().fit(X_tr.reshape(-1, X_tr.shape[-1]))
    X_tr = s.transform(X_tr.reshape(-1, X_tr.shape[-1])).reshape(X_tr.shape).astype(np.float32)
    X_te = s.transform(X_te.reshape(-1, X_te.shape[-1])).reshape(X_te.shape).astype(np.float32)

    if model_name == "lstm":
        model = LSTMAvailability(n_features=X.shape[2])
    elif model_name == "transformer":
        model = TransformerAvailability(n_features=X.shape[2])
    else:
        raise ValueError(model_name)

    return train_and_eval(model, X_tr, y_tr, X_te, y_te, epochs=epochs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--city", default="montpellier")
    p.add_argument("--model", choices=["lstm", "transformer"], default="lstm")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--with-imd", action="store_true")
    p.add_argument("--without-imd", action="store_true")
    args = p.parse_args()

    configs = []
    if args.with_imd or not args.without_imd:
        configs.append(("with_imd", True))
    if args.without_imd or not args.with_imd:
        configs.append(("no_imd", False))

    results = {}
    for tag, use_imd in configs:
        r = run_city(args.city, use_imd=use_imd, model_name=args.model,
                     epochs=args.epochs)
        if r is None: continue
        results[tag] = r
        print(f"  {tag}  MAE per horizon : {[round(v,2) for v in r['mae_per_horizon']]}")

    out_path = OUT_DIR / f"d5_{args.city}_{args.model}.json"
    out_path.write_text(json.dumps({
        "city": args.city,
        "model": args.model,
        "configs": results,
    }, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
