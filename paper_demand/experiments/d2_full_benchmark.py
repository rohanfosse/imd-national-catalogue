"""
d2_full_benchmark.py — Full demand-prediction benchmark on Vélomagg.

Compares six model families on hourly station-level trips :
 (P) Persistence            — y_pred(s,t) = y(s, t - 7d)
 (AR) ARIMA(1,1,1) per stn  — classical time-series
 (L) Linear (Ridge)         — same features as LightGBM
 (G) LightGBM               — gradient-boosted trees (tabular SOTA)
 (N) LSTM                   — sequence model, station-conditioned
 (G+) LightGBM + IMD        — explicit IMD-augmented variant of (G)

Train: 2021-09 → 2024-08    Test: 2024-09 → 2025-08
Target: log1p(demande)      Metric: R^2 / MAE / RMSE on trips scale
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from pandas.api.types import is_numeric_dtype

warnings.filterwarnings("ignore")

DATA_ROOT = Path("/home/rohanfosse/Bureau/Recherche/bikeshare-data-explorer/data")
OUT_DIR = Path("/home/rohanfosse/Bureau/Recherche/imd-national-catalogue/paper_demand/experiments/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Shared data loader ───────────────────────────────────────────────────────
def load_data():
    trips = pd.read_csv(DATA_ROOT / "processed" / "dataset_prediction_complet.csv")
    trips["datetime"] = pd.to_datetime(trips["datetime"])
    weather = pd.read_csv(DATA_ROOT / "processed" / "weather_data_enriched.csv")
    weather["datetime"] = pd.to_datetime(weather["datetime"])

    gs = pd.read_parquet(DATA_ROOT / "stations_gold_standard_final.parquet")
    mtp = gs[gs["city"].str.contains("Montpellier", case=False, na=False)].copy()
    imd_cols = [
        "station_name",
        "gtfs_heavy_stops_300m", "gtfs_stops_within_300m_pct",
        "infra_cyclable_km", "infra_cyclable_pct",
        "elevation_m", "topography_roughness_index",
        "n_stations_within_500m", "n_stations_within_1km",
        "catchment_density_per_km2",
        "revenu_median_uc", "revenu_d1",
        "part_menages_voit0", "part_velo_travail",
    ]
    mtp = mtp[imd_cols].rename(columns={"station_name": "station"})

    df = trips.merge(weather, on="datetime", how="inner")
    df = df.merge(mtp, on="station", how="left")
    df = df.dropna(subset=["gtfs_heavy_stops_300m"]).copy()
    df["log_demand"] = np.log1p(df["demande"])
    return df


def make_splits(df, test_cutoff="2024-09-01"):
    return df[df["datetime"] < test_cutoff].copy(), df[df["datetime"] >= test_cutoff].copy()


def evaluate(name, y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(np.clip(y_pred_log, 0, None))
    return {
        "model": name,
        "n_test": int(len(y_true)),
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
        "r2_trips": float(r2_score(y_true, y_pred)),
        "mae_trips": float(mean_absolute_error(y_true, y_pred)),
        "rmse_trips": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


FEATS_T = ["hour", "day_of_week", "month", "season", "temperature",
           "humidity", "precipitation", "wind_speed", "cloud_cover",
           "is_raining", "feels_like", "bad_weather_score"]
FEATS_NET = ["network_volume", "network_entropy", "network_gini"]
FEATS_IMD = ["gtfs_heavy_stops_300m", "gtfs_stops_within_300m_pct",
             "infra_cyclable_km", "infra_cyclable_pct",
             "elevation_m", "topography_roughness_index",
             "n_stations_within_500m", "n_stations_within_1km",
             "catchment_density_per_km2",
             "revenu_median_uc", "revenu_d1",
             "part_menages_voit0", "part_velo_travail"]


# ─── (P) Persistence ──────────────────────────────────────────────────────────
def model_persistence(df, train, test):
    """y_pred(s,t) = y(s, t - 7d)"""
    df_lag = df[["station", "datetime", "log_demand"]].copy()
    df_lag["datetime_pred"] = df_lag["datetime"] + pd.Timedelta(days=7)
    df_lag = df_lag.rename(columns={"log_demand": "y_lag7"})
    pred_test = test.merge(df_lag[["station", "datetime_pred", "y_lag7"]],
                           left_on=["station", "datetime"],
                           right_on=["station", "datetime_pred"], how="left")
    mask = pred_test["y_lag7"].notna()
    return evaluate("P_persistence_lag7d",
                    pred_test.loc[mask, "log_demand"].values,
                    pred_test.loc[mask, "y_lag7"].values)


# ─── (AR) ARIMA per station ───────────────────────────────────────────────────
def model_arima(train, test, max_stations=53):
    """Fit ARIMA(1,1,1) per station on hourly series. Slow but classical."""
    results_y_true, results_y_pred = [], []
    stations = train["station"].unique()[:max_stations]
    for i, s in enumerate(stations):
        tr = train[train["station"] == s].sort_values("datetime")["log_demand"].values
        te = test[test["station"] == s].sort_values("datetime")["log_demand"].values
        if len(tr) < 200 or len(te) == 0:
            continue
        try:
            model = ARIMA(tr, order=(1, 1, 1)).fit(method_kwargs={"warn_convergence": False})
            forecast = model.forecast(steps=len(te))
            results_y_true.extend(te)
            results_y_pred.extend(forecast)
        except Exception:
            # Fallback: use the train mean
            results_y_true.extend(te)
            results_y_pred.extend(np.full(len(te), tr.mean()))
        if (i + 1) % 10 == 0:
            print(f"        ARIMA fitted {i+1}/{len(stations)} stations")
    return evaluate("AR_arima_111", np.array(results_y_true), np.array(results_y_pred))


# ─── (L) Linear Ridge ─────────────────────────────────────────────────────────
def _encode(X_train, X_test, features):
    Xt, Xv = X_train[features].copy(), X_test[features].copy()
    for c in features:
        if not is_numeric_dtype(Xt[c]):
            combined = pd.concat([Xt[c].astype(str), Xv[c].astype(str)])
            cats = combined.astype("category").cat.categories
            Xt[c] = pd.Categorical(Xt[c].astype(str), categories=cats).codes
            Xv[c] = pd.Categorical(Xv[c].astype(str), categories=cats).codes
    return Xt.astype("float64"), Xv.astype("float64")


def model_ridge(train, test, features, name):
    Xt, Xv = _encode(train, test, features)
    scaler = StandardScaler()
    Xt_s = scaler.fit_transform(Xt.fillna(0))
    Xv_s = scaler.transform(Xv.fillna(0))
    model = Ridge(alpha=1.0)
    t0 = time.time()
    model.fit(Xt_s, train["log_demand"].values)
    fit_s = time.time() - t0
    y_pred = model.predict(Xv_s)
    m = evaluate(name, test["log_demand"].values, y_pred)
    m["fit_seconds"] = round(fit_s, 1)
    return m


# ─── (G) LightGBM ─────────────────────────────────────────────────────────────
def model_lgb(train, test, features, name):
    Xt, Xv = _encode(train, test, features)
    cat_cols = [c for c in features if not is_numeric_dtype(train[c])]
    model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        min_child_samples=30, reg_lambda=0.5, random_state=42,
        n_jobs=-1, verbose=-1,
    )
    t0 = time.time()
    model.fit(Xt, train["log_demand"].values,
              categorical_feature=cat_cols if cat_cols else "auto")
    fit_s = time.time() - t0
    y_pred = model.predict(Xv)
    m = evaluate(name, test["log_demand"].values, y_pred)
    m["fit_seconds"] = round(fit_s, 1)
    fi = pd.DataFrame({"feature": features,
                       "importance": model.booster_.feature_importance(importance_type="gain")})
    fi = fi.sort_values("importance", ascending=False)
    return m, fi


# ─── (N) LSTM ─────────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, n_features, n_stations, hidden=64, n_layers=1):
        super().__init__()
        self.station_emb = nn.Embedding(n_stations, 16)
        self.lstm = nn.LSTM(input_size=n_features + 16, hidden_size=hidden,
                            num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x_seq, station_ids):
        emb = self.station_emb(station_ids).unsqueeze(1).expand(-1, x_seq.size(1), -1)
        x = torch.cat([x_seq, emb], dim=-1)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def make_sequences(df, station_to_idx, features, seq_len=24, horizon=1):
    """Build (X_seq, station_id, y) for one-step-ahead prediction.
    For each (station, t), use [t-seq_len, t-1] to predict t.
    """
    df = df.sort_values(["station", "datetime"]).reset_index(drop=True)
    Xs, sids, ys = [], [], []
    for s, g in df.groupby("station"):
        if s not in station_to_idx:
            continue
        arr = g[features].fillna(0).values
        y = g["log_demand"].values
        for i in range(seq_len, len(g) - horizon + 1):
            Xs.append(arr[i - seq_len:i])
            sids.append(station_to_idx[s])
            ys.append(y[i + horizon - 1])
    if not Xs:
        return None, None, None
    return np.array(Xs, dtype=np.float32), np.array(sids, dtype=np.int64), np.array(ys, dtype=np.float32)


def model_lstm(train, test, features, name, epochs=3, seq_len=24, batch=512):
    print(f"        Building LSTM sequences (seq_len={seq_len})...")
    stations = sorted(train["station"].unique())
    s2i = {s: i for i, s in enumerate(stations)}

    # Scale numeric features on train
    train_feat = train[features].fillna(0).astype("float64")
    test_feat = test[features].fillna(0).astype("float64")
    scaler = StandardScaler().fit(train_feat)
    train2 = train.copy()
    test2 = test.copy()
    train2[features] = scaler.transform(train_feat)
    test2[features] = scaler.transform(test_feat)

    X_tr, s_tr, y_tr = make_sequences(train2, s2i, features, seq_len)
    X_te, s_te, y_te = make_sequences(test2, s2i, features, seq_len)
    if X_tr is None or X_te is None:
        return {"model": name, "error": "no sequences"}

    print(f"        Train sequences: {len(X_tr):,}  Test sequences: {len(X_te):,}")

    device = torch.device("cpu")
    model = LSTMModel(n_features=len(features), n_stations=len(s2i)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    X_tr_t = torch.from_numpy(X_tr).to(device)
    s_tr_t = torch.from_numpy(s_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    n = len(X_tr_t)

    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n)
        running = 0.0
        nb_batches = 0
        for b in range(0, n, batch):
            idx = perm[b:b + batch]
            opt.zero_grad()
            yhat = model(X_tr_t[idx], s_tr_t[idx])
            loss = loss_fn(yhat, y_tr_t[idx])
            loss.backward()
            opt.step()
            running += loss.item()
            nb_batches += 1
        print(f"        Epoch {ep+1}/{epochs}  train MSE = {running/nb_batches:.4f}")

    fit_s = time.time() - t0
    model.eval()
    with torch.no_grad():
        X_te_t = torch.from_numpy(X_te).to(device)
        s_te_t = torch.from_numpy(s_te).to(device)
        preds = []
        for b in range(0, len(X_te_t), batch):
            preds.append(model(X_te_t[b:b + batch], s_te_t[b:b + batch]).numpy())
        y_pred = np.concatenate(preds)

    m = evaluate(name, y_te, y_pred)
    m["fit_seconds"] = round(fit_s, 1)
    return m


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("[1/7] Loading data...")
    df = load_data()
    train, test = make_splits(df)
    print(f"      total={len(df):,}  train={len(train):,}  test={len(test):,}")
    print(f"      stations={df['station'].nunique()}  span={df['datetime'].min()} to {df['datetime'].max()}")

    results = []

    print("[2/7] (P) Persistence...")
    results.append(model_persistence(df, train, test))
    print(f"      R²_trips = {results[-1]['r2_trips']:.4f}  MAE = {results[-1]['mae_trips']:.3f}")

    print("[3/7] (AR) ARIMA(1,1,1) per station (53 fits)...")
    results.append(model_arima(train, test))
    print(f"      R²_trips = {results[-1]['r2_trips']:.4f}  MAE = {results[-1]['mae_trips']:.3f}")

    print("[4/7] (L) Linear Ridge (full features)...")
    results.append(model_ridge(train, test, FEATS_T + FEATS_NET + FEATS_IMD, "L_ridge_full"))
    print(f"      R²_trips = {results[-1]['r2_trips']:.4f}  MAE = {results[-1]['mae_trips']:.3f}")

    print("[5/7] (G) LightGBM (full features)...")
    m, fi = model_lgb(train, test, FEATS_T + FEATS_NET + FEATS_IMD, "G_lgbm_full")
    results.append(m)
    fi.to_csv(OUT_DIR / "d2_feature_importance_lgbm.csv", index=False)
    print(f"      R²_trips = {results[-1]['r2_trips']:.4f}  MAE = {results[-1]['mae_trips']:.3f}")

    print("[6/7] (G-) LightGBM ablation (NO IMD features)...")
    m2, _ = model_lgb(train, test, FEATS_T + FEATS_NET, "G_lgbm_no_imd")
    results.append(m2)
    print(f"      R²_trips = {results[-1]['r2_trips']:.4f}  MAE = {results[-1]['mae_trips']:.3f}")

    print("[7/7] (N) LSTM (seq_len=24, 3 epochs)...")
    m_lstm = model_lstm(train, test, FEATS_T + FEATS_NET + FEATS_IMD, "N_lstm_full")
    results.append(m_lstm)
    print(f"      R²_trips = {results[-1].get('r2_trips', 'err'):.4f}" if 'r2_trips' in results[-1] else f"      error")

    print("\n" + "=" * 70)
    print("FINAL BENCHMARK SUMMARY (test set, trips units)")
    print("=" * 70)
    print(f"  {'model':30s}  {'R²_trips':>10s}  {'MAE':>8s}  {'RMSE':>8s}")
    print("  " + "-" * 60)
    for r in results:
        print(f"  {r['model']:30s}  {r.get('r2_trips', 0):10.4f}  {r.get('mae_trips', 0):8.3f}  {r.get('rmse_trips', 0):8.3f}")

    # Save
    summary = {
        "experiment": "d2_full_benchmark",
        "data": {
            "n_total": int(len(df)),
            "n_stations": int(df["station"].nunique()),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
        },
        "models": results,
    }
    with open(OUT_DIR / "d2_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] Wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
