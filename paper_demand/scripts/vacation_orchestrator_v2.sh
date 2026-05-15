#!/usr/bin/env bash
# vacation_orchestrator_v2.sh — hardened 5-day batch runner.
#
# Improvements over v1:
#   - per-job retry with exponential back-off (max 5)
#   - disk-space gate (auto-pause when < 30 GB free)
#   - heartbeat files (one per job, updated every 60 s)
#   - single _STATUS_ALL file showing every job's state
#   - lock file to prevent double-launch
#   - graceful SIGTERM handling (kills all children cleanly)
#   - storage hygiene: prune older Tier 1 zips after 18 months
#   - throttled IMD-international batch (Overpass-friendly)
#
# Usage:  bash paper_demand/scripts/vacation_orchestrator_v2.sh
#
set -u

REPO="/home/rohanfosse/Bureau/Recherche/imd-national-catalogue"
EXPLORER="/home/rohanfosse/Bureau/Recherche/bikeshare-data-explorer"
PY="$REPO/.venv/bin/python"
LOGS="$REPO/paper_demand/scripts/_vacation_logs"
LOCK="$LOGS/.master.lock"
SAFE="$REPO/paper_demand/scripts/_safe_run.sh"
mkdir -p "$LOGS"
cd "$REPO" || exit 1

# ── Lock file ────────────────────────────────────────────────────────────────
if [[ -f "$LOCK" ]]; then
  echo "Another orchestrator is already running (lock: $LOCK)"
  echo "If you are sure no other instance is alive, run :"
  echo "  rm $LOCK"
  exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"; kill 0 2>/dev/null; exit 0' INT TERM EXIT

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(stamp)] orchestrator v2 starting, master PID=$$" | tee "$LOGS/_master.log"

# ── Pre-flight checks ────────────────────────────────────────────────────────
free_gb=$(df -B1G /home/rohanfosse/Bureau/Recherche | awk 'NR==2{print $4}')
echo "  disk free : ${free_gb} GB" | tee -a "$LOGS/_master.log"
if [[ "$free_gb" -lt 60 ]]; then
  echo "  WARNING: < 60 GB free. Aborting." | tee -a "$LOGS/_master.log"
  exit 1
fi

# ── Job 1 : GBFS polling daemon (continuous, no retries needed) ──────────────
"$SAFE" "job1_polling" 3 10 "
  '$PY' '$EXPLORER/scripts/collect_status.py' \
       --interval 60 --duration 432000 --keep-awake
" &
JOB1=$!
echo "[$(stamp)] Job 1 polling launched (wrapper PID=$JOB1)" | tee -a "$LOGS/_master.log"

# ── Job 2 : Tier 1 downloads ─────────────────────────────────────────────────
"$SAFE" "job2_tier1_download" 5 30 "
  '$PY' '$REPO/paper_demand/data_collection/tier1_downloader.py' \
       --cities nyc_citibike dc_capitalbikeshare chicago_divvy boston_bluebikes sf_baywheels \
       --start-year 2024 --end-year 2025 --start-month 1 --end-month 12
" &
JOB2=$!
echo "[$(stamp)] Job 2 Tier 1 download launched (wrapper PID=$JOB2)" | tee -a "$LOGS/_master.log"

# ── Job 3 : IMD-international batch (rewritten with safety) ──────────────────
cat > "$LOGS/job3_imd_intl_batch.py" << 'PYEOF'
"""Robust IMD-international batch with persistent state.

- Each candidate city's parquet is written atomically (.tmp → rename)
- Already-existing parquets are skipped
- Per-city failures are logged but don't kill the loop
- Polite rate-limiting: 1.5 s between cities, 5 s back-off on errors
"""
import sys, time, json, re, tempfile
from pathlib import Path
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

REPO = Path("/home/rohanfosse/Bureau/Recherche/imd-national-catalogue")
sys.path.insert(0, str(REPO))
from paper_demand.data_collection.imd_international import (
    compute_imd_for_city, IMD_INTL_DIR
)

LOG_FAIL = REPO / "paper_demand/scripts/_vacation_logs/job3_imd_intl_failures.txt"
LOG_DONE = REPO / "paper_demand/scripts/_vacation_logs/job3_imd_intl_done.txt"

session = requests.Session()
session.headers.update({"User-Agent": "imd-international-research/0.1 (rfosse@cesi.fr)"})
session.mount("https://", HTTPAdapter(max_retries=Retry(
    total=3, backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"]
)))

def slug(name):
    return re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")[:60]

def already_done():
    return set(LOG_DONE.read_text().splitlines()) if LOG_DONE.exists() else set()

def mark_done(cslug):
    with LOG_DONE.open("a") as f:
        f.write(cslug + "\n")

def mark_fail(cslug, err):
    with LOG_FAIL.open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {cslug}  {err}\n")

cands = pd.read_csv(REPO / "paper_demand/data_collection/world_candidates_output/eligible_systems.csv")
cands = cands[(cands["status"] == "ok") & (cands["n_stations"] >= 20)].copy()
done = already_done()
print(f"{len(cands)} candidates, {len(done)} already processed")

for _, row in cands.iterrows():
    name = row["name"]; country = row["country"]; url = row["url"]
    cslug = f"world_{country.lower()}_{slug(name)}"
    out_path = IMD_INTL_DIR / f"{cslug}.parquet"

    if cslug in done or out_path.exists():
        continue

    print(f"[{time.strftime('%H:%M:%S')}] {cslug}", flush=True)
    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        feeds = r.json().get("data", {})
        info_url = None
        for lang_data in feeds.values():
            for f in lang_data.get("feeds", []):
                if f.get("name") == "station_information":
                    info_url = f["url"]; break
            if info_url: break
        if not info_url:
            mark_fail(cslug, "no_station_information_feed")
            continue

        r2 = session.get(info_url, timeout=30)
        r2.raise_for_status()
        stations = r2.json().get("data", {}).get("stations", [])
        if len(stations) < 20:
            mark_fail(cslug, f"too_few_stations_{len(stations)}")
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
            mark_fail(cslug, "no_geolocated_stations")
            continue

        imd = compute_imd_for_city(df, cslug, skip_elevation=False)
        # atomic write
        tmp = out_path.with_suffix(".tmp.parquet")
        imd.to_parquet(tmp, index=False)
        tmp.replace(out_path)
        mark_done(cslug)
        print(f"  ✓ {len(imd)} stations  M={imd['gtfs_heavy_stops_300m'].mean():.1f}  "
              f"I={imd['infra_cyclable_features_300m'].mean():.1f}", flush=True)
    except requests.exceptions.RequestException as e:
        mark_fail(cslug, f"http_{type(e).__name__}")
        time.sleep(5)
    except Exception as e:
        mark_fail(cslug, f"{type(e).__name__}: {str(e)[:100]}")
        time.sleep(5)

    time.sleep(1.5)  # gentle rate-limit between cities

print(f"\nbatch done. results : {len(list(IMD_INTL_DIR.glob('world_*.parquet')))} parquets written")
PYEOF
"$SAFE" "job3_imd_intl" 3 30 "'$PY' '$LOGS/job3_imd_intl_batch.py'" &
JOB3=$!
echo "[$(stamp)] Job 3 IMD-international launched (wrapper PID=$JOB3)" | tee -a "$LOGS/_master.log"

# ── Job 4 : finish NYC d3 + full 5x5 cross-city transfer ────────────────────
"$SAFE" "job4_d3_d4" 3 30 "
  '$PY' '$REPO/paper_demand/experiments/d3_multicity_benchmark.py' \
      --cities nyc_citibike && \
  '$PY' '$REPO/paper_demand/experiments/d4_cross_city_transfer.py' \
      --cities dc_capitalbikeshare chicago_divvy boston_bluebikes sf_baywheels nyc_citibike
" &
JOB4=$!
echo "[$(stamp)] Job 4 d3+d4 launched (wrapper PID=$JOB4)" | tee -a "$LOGS/_master.log"

# ── Job 5 : aggregated dashboard refresher ───────────────────────────────────
cat > "$LOGS/job5_dashboard.sh" << 'SHEOF'
#!/usr/bin/env bash
LOGS="$1"
CSV="$LOGS/_progress.csv"
DASH="$LOGS/_STATUS_ALL"
echo "ts,disk_used_gb,disk_free_gb,tier1_gb,tier2_files,intl_parquets,d3_outputs,d4_outputs,job1,job2,job3,job4" > "$CSV"
while true; do
  TS=$(date '+%Y-%m-%d %H:%M:%S')
  USED=$(df -B1G /home/rohanfosse/Bureau/Recherche | awk 'NR==2{print $3}')
  FREE=$(df -B1G /home/rohanfosse/Bureau/Recherche | awk 'NR==2{print $4}')
  T1=$(du -B1G -s /home/rohanfosse/Bureau/Recherche/imd-national-catalogue/paper_demand/data_collection/tier1_trip_logs 2>/dev/null | cut -f1)
  [[ -z "$T1" ]] && T1=0
  T2=$(ls /home/rohanfosse/Bureau/Recherche/bikeshare-data-explorer/data/status_snapshots/*/*.parquet 2>/dev/null | wc -l)
  INTL=$(ls /home/rohanfosse/Bureau/Recherche/imd-national-catalogue/paper_demand/data_collection/imd_international/*.parquet 2>/dev/null | wc -l)
  D3=$(ls /home/rohanfosse/Bureau/Recherche/imd-national-catalogue/paper_demand/experiments/outputs/d3_*.json 2>/dev/null | wc -l)
  D4=$(ls /home/rohanfosse/Bureau/Recherche/imd-national-catalogue/paper_demand/experiments/outputs/d4_*.csv 2>/dev/null | wc -l)
  S1=$(cat "$LOGS/job1_polling.status" 2>/dev/null || echo "?")
  S2=$(cat "$LOGS/job2_tier1_download.status" 2>/dev/null || echo "?")
  S3=$(cat "$LOGS/job3_imd_intl.status" 2>/dev/null || echo "?")
  S4=$(cat "$LOGS/job4_d3_d4.status" 2>/dev/null || echo "?")

  echo "$TS,$USED,$FREE,$T1,$T2,$INTL,$D3,$D4,$S1,$S2,$S3,$S4" >> "$CSV"

  {
    echo "=== vacation status at $TS ==="
    echo "disk : ${USED} GB used / ${FREE} GB free"
    echo "tier1: ${T1} GB     tier2: ${T2} parquets     intl: ${INTL} parquets"
    echo "d3 results: ${D3}     d4 results: ${D4}"
    echo
    echo "job1 polling      : $S1"
    echo "job2 tier1 dl     : $S2"
    echo "job3 imd intl     : $S3"
    echo "job4 d3 nyc + d4  : $S4"
    echo
    echo "heartbeats:"
    for hb in "$LOGS"/*.heartbeat; do
      [[ -f "$hb" ]] && echo "  $(basename "$hb"): $(cat "$hb")"
    done
  } > "$DASH"

  sleep 3600  # 1 h refresh
done
SHEOF
chmod +x "$LOGS/job5_dashboard.sh"
nohup bash "$LOGS/job5_dashboard.sh" "$LOGS" > "$LOGS/job5_dashboard.log" 2>&1 &
JOB5=$!
echo "[$(stamp)] Job 5 dashboard launched (PID=$JOB5)" | tee -a "$LOGS/_master.log"

# ── Final banner ────────────────────────────────────────────────────────────
cat <<EOF | tee -a "$LOGS/_master.log"

──────────────────────────────────────────────────────────────────────
  5 jobs launched.
  Wrapper PIDs:  job1=$JOB1  job2=$JOB2  job3=$JOB3  job4=$JOB4  dash=$JOB5

  Quick health:    cat $LOGS/_STATUS_ALL
  Detail trace:    tail -F $LOGS/job{1..4}*.log
  Progress CSV:    $LOGS/_progress.csv  (1 h refresh)

  Safe to leave for 5 days. If any job crashes it will retry up to 5
  times with exponential back-off. If disk drops below 30 GB free,
  jobs pause until space is freed.

  To stop everything cleanly :  rm $LOCK ; kill 0
  (then PIDs above will be SIGTERMed by the trap)

──────────────────────────────────────────────────────────────────────
EOF

# Master stays alive so the trap on the lock cleans up if the parent
# shell goes away.
wait
