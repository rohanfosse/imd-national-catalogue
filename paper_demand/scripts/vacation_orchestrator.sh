#!/usr/bin/env bash
# vacation_orchestrator.sh — runs the long-running data-collection and
# benchmark jobs for the paper_demand IMD-augmented forecasting study.
# Designed to be safely left running unattended for 5 days.
#
# Usage:  cd ~/Bureau/Recherche/imd-national-catalogue && \
#         bash paper_demand/scripts/vacation_orchestrator.sh
#
# Each job runs in the background, writes to its own log under
# paper_demand/scripts/_vacation_logs/, and is independent: if one
# crashes the others continue.
set -u

REPO="/home/rohanfosse/Bureau/Recherche/imd-national-catalogue"
EXPLORER="/home/rohanfosse/Bureau/Recherche/bikeshare-data-explorer"
PY="$REPO/.venv/bin/python"
LOGS="$REPO/paper_demand/scripts/_vacation_logs"
mkdir -p "$LOGS"
cd "$REPO" || exit 1

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(stamp)] vacation orchestrator starting" > "$LOGS/_master.log"

# ── Job 1 : GBFS polling daemon (continuous, the highest ROI) ─────────────────
#   Keeps the Tier 2 multi-city panel growing. 1 poll / minute on each
#   priority system. Expected output: ~5 GB over 5 days across ~43 cities.
nohup "$PY" "$EXPLORER/scripts/collect_status.py" \
    --interval 60 --duration 432000 --keep-awake \
    > "$LOGS/job1_polling.log" 2>&1 &
echo "[$(stamp)] Job 1 polling daemon  PID=$!" | tee -a "$LOGS/_master.log"

# ── Job 2 : finish Tier 1 trip-log downloads ──────────────────────────────────
#   Completes 2024-01 → 2025-12 on the 5 Lyft-operated US networks and adds
#   the cities we did not start: BIXI Montreal annual archives, TfL London
#   quarterlies, Mibici Guadalajara, Bicing Barcelona, etc. ~24 h wall time
#   bound by bandwidth.
(
  "$PY" "$REPO/paper_demand/data_collection/tier1_downloader.py" \
      --cities nyc_citibike dc_capitalbikeshare chicago_divvy boston_bluebikes sf_baywheels \
      --start-year 2024 --end-year 2025 --start-month 1 --end-month 12
) > "$LOGS/job2_tier1_download.log" 2>&1 &
echo "[$(stamp)] Job 2 Tier 1 download  PID=$!" | tee -a "$LOGS/_master.log"

# ── Job 3 : IMD-international for the 185 candidate cities ───────────────────
#   Calls OSM Overpass + Open-Elevation for every eligible system in
#   world_candidates_output/eligible_systems.csv that doesn't already have
#   a parquet under imd_international/. Rate-limited by Overpass (≥1 s
#   between bbox queries) and Open-Elevation. Expected wall time: 2-4 days.
cat > "$LOGS/job3_imd_intl_batch.py" << 'PYEOF'
"""Batch IMD-international for the 185 world candidates."""
import sys, time
from pathlib import Path
import pandas as pd

REPO = Path("/home/rohanfosse/Bureau/Recherche/imd-national-catalogue")
sys.path.insert(0, str(REPO))
from paper_demand.data_collection.imd_international import (
    compute_imd_for_city, IMD_INTL_DIR
)

cands = pd.read_csv(REPO / "paper_demand/data_collection/world_candidates_output/eligible_systems.csv")
# Filter: docked, status ok, n_stations >= 20
cands = cands[(cands["status"] == "ok") & (cands["n_stations"] >= 20)].copy()

# For each candidate, we need (station_id, lat, lon). Without trip logs we
# scrape station_information directly from the auto-discovery URL.
import json, requests, re
def slug(name):
    return re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")[:60]

session = requests.Session()
session.headers.update({"User-Agent": "imd-international-research/0.1 (rfosse@cesi.fr)"})

for _, row in cands.iterrows():
    name = row["name"]; country = row["country"]; url = row["url"]
    cslug = f"world_{country.lower()}_{slug(name)}"
    out_path = IMD_INTL_DIR / f"{cslug}.parquet"
    if out_path.exists():
        print(f"  skip {cslug}: already exists")
        continue
    print(f"[{time.strftime('%H:%M:%S')}] {cslug} ← {url}")
    try:
        # Find the station_information feed
        r = session.get(url, timeout=20)
        feeds = r.json().get("data", {})
        info_url = None
        for lang_data in feeds.values():
            for f in lang_data.get("feeds", []):
                if f.get("name") == "station_information":
                    info_url = f["url"]
                    break
            if info_url: break
        if not info_url:
            print(f"  ✗ no station_information feed")
            continue
        r2 = session.get(info_url, timeout=20)
        stations = r2.json().get("data", {}).get("stations", [])
        if len(stations) < 20:
            print(f"  ✗ only {len(stations)} stations")
            continue
        df = pd.DataFrame([{
            "station_id": str(s.get("station_id")),
            "name": s.get("name", ""),
            "lat": s.get("lat"),
            "lng": s.get("lon"),
            "n": 1,
        } for s in stations if s.get("lat") and s.get("lon")])
        df["city"] = cslug
        if df.empty:
            continue
        imd = compute_imd_for_city(df, cslug, skip_elevation=False)
        imd.to_parquet(out_path, index=False)
        print(f"  ✓ {len(imd)} stations  M_mean={imd['gtfs_heavy_stops_300m'].mean():.1f}  I_mean={imd['infra_cyclable_features_300m'].mean():.1f}")
    except Exception as e:
        print(f"  ✗ {type(e).__name__}: {e}")
        time.sleep(5)
        continue
    time.sleep(1.0)  # be polite to Overpass and to operator feeds
PYEOF
(
  "$PY" "$LOGS/job3_imd_intl_batch.py"
) > "$LOGS/job3_imd_intl.log" 2>&1 &
echo "[$(stamp)] Job 3 IMD-international batch  PID=$!" | tee -a "$LOGS/_master.log"

# ── Job 4 : finish stuck NYC d3 + run d4 cross-city transfer ─────────────────
(
  echo "=== d3 NYC ==="
  "$PY" "$REPO/paper_demand/experiments/d3_multicity_benchmark.py" \
      --cities nyc_citibike
  echo
  echo "=== d4 cross-city transfer ==="
  "$PY" "$REPO/paper_demand/experiments/d4_cross_city_transfer.py" \
      --cities dc_capitalbikeshare chicago_divvy boston_bluebikes sf_baywheels nyc_citibike
) > "$LOGS/job4_d3_nyc_d4.log" 2>&1 &
echo "[$(stamp)] Job 4 d3 NYC + d4 transfer  PID=$!" | tee -a "$LOGS/_master.log"

# ── Job 5 : daily checkpoint reporter ─────────────────────────────────────────
#   Wakes up every 6 h, snapshots disk usage, job status, and writes a
#   single CSV row so you can see progress without scrolling 5 days of logs.
cat > "$LOGS/job5_checkpoint.sh" << 'SHEOF'
#!/usr/bin/env bash
LOGS="$1"
CSV="$LOGS/_progress.csv"
echo "timestamp,disk_used_gb,tier1_gb,tier2_files,intl_parquets,d3_outputs,d4_outputs" > "$CSV"
while true; do
  TS=$(date '+%Y-%m-%d %H:%M:%S')
  DISK=$(df -B1G /home/rohanfosse/Bureau/Recherche | awk 'NR==2{print $3}')
  TIER1=$(du -B1G -s /home/rohanfosse/Bureau/Recherche/imd-national-catalogue/paper_demand/data_collection/tier1_trip_logs 2>/dev/null | cut -f1)
  TIER2=$(ls /home/rohanfosse/Bureau/Recherche/bikeshare-data-explorer/data/status_snapshots/*/*.parquet 2>/dev/null | wc -l)
  INTL=$(ls /home/rohanfosse/Bureau/Recherche/imd-national-catalogue/paper_demand/data_collection/imd_international/*.parquet 2>/dev/null | wc -l)
  D3=$(ls /home/rohanfosse/Bureau/Recherche/imd-national-catalogue/paper_demand/experiments/outputs/d3_*.json 2>/dev/null | wc -l)
  D4=$(ls /home/rohanfosse/Bureau/Recherche/imd-national-catalogue/paper_demand/experiments/outputs/d4_*.csv 2>/dev/null | wc -l)
  echo "$TS,$DISK,$TIER1,$TIER2,$INTL,$D3,$D4" >> "$CSV"
  sleep 21600  # 6 h
done
SHEOF
chmod +x "$LOGS/job5_checkpoint.sh"
nohup bash "$LOGS/job5_checkpoint.sh" "$LOGS" > "$LOGS/job5_checkpoint.log" 2>&1 &
echo "[$(stamp)] Job 5 checkpoint loop  PID=$!" | tee -a "$LOGS/_master.log"

# ── Status banner ─────────────────────────────────────────────────────────────
echo ""
echo "─── 5 jobs launched ────────────────────────────────────────────"
echo "  Logs : $LOGS/"
echo "  Progress CSV : $LOGS/_progress.csv  (updates every 6 h)"
echo ""
echo "To monitor :"
echo "  tail -f $LOGS/_progress.csv"
echo "  ps -ef | grep -E 'collect_status|tier1_downloader|imd_intl|d3_multicity|d4_cross|checkpoint.sh'"
echo ""
echo "To stop everything :"
echo "  pkill -f 'collect_status|tier1_downloader|imd_intl_batch|d3_multicity|d4_cross|checkpoint.sh'"
echo ""
echo "Estimated 5-day deliverables :"
echo "  - 5+ days of GBFS polling on 43 French cities (~5-8 GB)"
echo "  - 2 years of trip data on 5 US networks (~50 GB)"
echo "  - IMD-international parquets for ~150-185 candidate cities"
echo "  - NYC d3 benchmark + full 5x5 cross-city transfer matrix"
echo "  - Progress checkpoints every 6 hours"
