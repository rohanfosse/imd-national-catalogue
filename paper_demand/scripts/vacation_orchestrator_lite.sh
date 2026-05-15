#!/usr/bin/env bash
# vacation_orchestrator_lite.sh — RAM-safe version of v2.
#
# Lessons from the v2 crash:
#   - d3_multicity_benchmark on NYC loads ~12 GB of trip data in pandas.
#   - d4_cross_city_transfer loads 5 cities at once.
#   - Together they exceed the 15 GB laptop RAM and trigger the OOM killer,
#     which SIGKILLs the parent bash, taking the whole orchestrator with it.
#
# Lite strategy : only run the jobs that are bounded in memory.
#   ✓ Job 1 — GBFS polling daemon  (~200 MB RSS, network-bound)
#   ✓ Job 2 — Tier 1 download      (urllib, ~50 MB RSS per city, sequential)
#   ✓ Job 3 — IMD-international    (Overpass + Open-Elevation, ~500 MB RSS)
#   ✓ Job 5 — Dashboard / reporter (negligible)
#   ✗ Job 4 — d3 NYC + d4 transfer : skipped. Run on demand when back at
#     desk, ideally on a workstation with > 32 GB RAM.
#
# Memory guard : every job is wrapped with _safe_run.sh and additionally
# pauses when free RAM falls below MIN_FREE_RAM_GB. This avoids triggering
# the OOM killer when VSCode / Firefox spike.
#
# Tier 1 downloads run SEQUENTIALLY (one city at a time) instead of the v2
# parallel layout, so peak memory is one urllib download buffer (≤ 50 MB).
#
# Usage : bash paper_demand/scripts/vacation_orchestrator_lite.sh
set -u

REPO="/home/rohanfosse/Bureau/Recherche/imd-national-catalogue"
EXPLORER="/home/rohanfosse/Bureau/Recherche/bikeshare-data-explorer"
PY="$REPO/.venv/bin/python"
SCRIPTS="$REPO/paper_demand/scripts"
LOGS="$SCRIPTS/_vacation_logs"
LOCK="$LOGS/.master.lock"
SAFE="$SCRIPTS/_safe_run.sh"

MIN_FREE_RAM_GB=2     # pause if available RAM falls below this
MIN_FREE_DISK_GB=20   # pause if free disk falls below this

mkdir -p "$LOGS"
cd "$REPO" || exit 1

# ── Lock guard ────────────────────────────────────────────────────────────────
if [[ -f "$LOCK" ]]; then
  OLD=$(cat "$LOCK")
  if /bin/kill -0 "$OLD" 2>/dev/null; then
    echo "Orchestrator already running (PID $OLD). Exit." ; exit 1
  fi
  rm -f "$LOCK"
fi
echo "$$" > "$LOCK"

cleanup() {
  echo "[$(date '+%F %T')] orchestrator stopping" >> "$LOGS/_master.log"
  pkill -P $$ 2>/dev/null
  rm -f "$LOCK"
}
trap cleanup INT TERM EXIT

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
master_log() { echo "[$(stamp)] $*" | tee -a "$LOGS/_master.log"; }

master_log "=== vacation orchestrator LITE starting (PID=$$) ==="
master_log "free disk : $(df -B1G "$REPO" | awk 'NR==2{print $4}') GB"
master_log "free RAM  : $(free -g | awk 'NR==2{print $7}') GB"
master_log "skipping  : d3 NYC + d4 transfer (memory-unsafe on 15 GB laptop)"

# ── RAM-aware wrapper ─────────────────────────────────────────────────────────
# Wraps the existing _safe_run.sh with an additional RAM guard. Every job
# polls free RAM every 60 s ; if it falls below MIN_FREE_RAM_GB, the job
# sends SIGSTOP to its Python child to freeze it, then SIGCONT when memory
# clears. This avoids being killed by oom_reaper.
cat > "$LOGS/_ram_guard.sh" << 'GUARDEOF'
#!/usr/bin/env bash
# Background sentinel : SIGSTOPs the most-RAM-hungry child if free RAM
# drops below the threshold, SIGCONTs when it clears.
JOB="${1}"
MIN_FREE_GB="${2:-2}"
LOGS="${3}"
LOG="$LOGS/${JOB}.ramguard.log"
echo "[$(date '+%F %T')] ramguard starting for $JOB (min_free=${MIN_FREE_GB} GB)" >> "$LOG"
declare -a stopped
while true; do
  free_g=$(free -g | awk 'NR==2{print $7}')
  if [[ "$free_g" -lt "$MIN_FREE_GB" ]]; then
    # Find biggest python child not already stopped
    biggest=$(/usr/bin/ps -eo pid,rss,state,cmd --sort=-rss \
              | grep -v grep | awk -v job="$JOB" '$3!="T" && /python/ && $0 ~ job {print $1; exit}')
    if [[ -n "$biggest" ]] && ! [[ " ${stopped[*]} " =~ " $biggest " ]]; then
      kill -STOP "$biggest" 2>/dev/null && {
        stopped+=("$biggest")
        echo "[$(date '+%F %T')] RAM low (${free_g} GB) → SIGSTOP $biggest" >> "$LOG"
      }
    fi
  elif [[ "$free_g" -ge $((MIN_FREE_GB + 2)) ]] && [[ "${#stopped[@]}" -gt 0 ]]; then
    # Resume all stopped children once we have at least 2 GB headroom
    for pid in "${stopped[@]}"; do
      kill -CONT "$pid" 2>/dev/null && \
        echo "[$(date '+%F %T')] RAM ok (${free_g} GB) → SIGCONT $pid" >> "$LOG"
    done
    stopped=()
  fi
  sleep 60
done
GUARDEOF
chmod +x "$LOGS/_ram_guard.sh"

# ── Job 1 : GBFS polling daemon (lightest, highest ROI) ──────────────────────
"$SAFE" job1_polling 10 "$MIN_FREE_DISK_GB" \
  "'$PY' '$EXPLORER/scripts/collect_status.py' --interval 60 --duration 432000 --keep-awake" &
master_log "Job 1 polling daemon          PID=$!"

# ── Job 2 : Tier 1 downloads (one city at a time, sequential) ────────────────
# Sequential keeps peak memory at one urllib download buffer (≤ 50 MB).
SEQUENTIAL_TIER1_CMD=""
for city in dc_capitalbikeshare chicago_divvy boston_bluebikes sf_baywheels nyc_citibike; do
  CMD_ONE="'$PY' '$REPO/paper_demand/data_collection/tier1_downloader.py' --cities $city --start-year 2024 --end-year 2025 --start-month 1 --end-month 12"
  if [[ -z "$SEQUENTIAL_TIER1_CMD" ]]; then
    SEQUENTIAL_TIER1_CMD="$CMD_ONE"
  else
    SEQUENTIAL_TIER1_CMD="$SEQUENTIAL_TIER1_CMD ; $CMD_ONE"
  fi
done
"$SAFE" job2_tier1_sequential 5 50 "$SEQUENTIAL_TIER1_CMD" &
master_log "Job 2 Tier 1 download (sequ.)  PID=$!"

# ── Job 3 : IMD-international (Overpass-bound, light memory) ─────────────────
# Re-uses the existing batch script that was generated by v2.
if [[ ! -f "$LOGS/job3_imd_intl_batch.py" ]]; then
cat > "$LOGS/job3_imd_intl_batch.py" << 'PYEOF'
import json, re, sys, time
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

def make_session():
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=2.0,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET", "POST"])
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "imd-international-research/0.2"})
    return s

session = make_session()
def slug(name):
    return re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")[:60]

cands = pd.read_csv(REPO / "paper_demand/data_collection/world_candidates_output/eligible_systems.csv")
cands = cands[(cands["status"] == "ok") & (cands["n_stations"] >= 20)].copy()
print(f"[{time.strftime('%F %T')}] {len(cands)} candidates", flush=True)

processed = 0
for _, row in cands.iterrows():
    name = row["name"]; country = row["country"]; url = row["url"]
    cslug = f"world_{country.lower()}_{slug(name)}"
    out_path = IMD_INTL_DIR / f"{cslug}.parquet"
    if out_path.exists():
        continue
    try:
        r = session.get(url, timeout=20)
        feeds = r.json().get("data", {})
        info_url = None
        for lang_data in feeds.values():
            for f in lang_data.get("feeds", []):
                if f.get("name") == "station_information":
                    info_url = f["url"]; break
            if info_url: break
        if not info_url:
            print(f"  ✗ {cslug}: no station_information feed", flush=True); continue
        r2 = session.get(info_url, timeout=20)
        stations = r2.json().get("data", {}).get("stations", [])
        df = pd.DataFrame([{
            "station_id": str(s.get("station_id")),
            "name": s.get("name", ""),
            "lat": s.get("lat"), "lng": s.get("lon"), "n": 1,
        } for s in stations if s.get("lat") and s.get("lon")])
        df["city"] = cslug
        if len(df) < 20:
            print(f"  ✗ {cslug}: only {len(df)} usable stations", flush=True); continue
        imd = compute_imd_for_city(df, cslug, skip_elevation=False)
        tmp = out_path.with_suffix(".parquet.tmp")
        imd.to_parquet(tmp, index=False)
        tmp.rename(out_path)
        processed += 1
        print(f"  ✓ {cslug}: {len(imd)} stns ({processed} done)", flush=True)
    except Exception as e:
        print(f"  ✗ {cslug}: {type(e).__name__}: {e}", flush=True)
    time.sleep(1.5)

print(f"[{time.strftime('%F %T')}] done, {processed} processed", flush=True)
PYEOF
fi
"$SAFE" job3_imd_intl 3 10 "'$PY' '$LOGS/job3_imd_intl_batch.py'" &
master_log "Job 3 IMD-international        PID=$!"

# ── Job 5 : hourly reporter (uses _summary.txt + _progress.csv) ──────────────
cat > "$LOGS/job5_lite_reporter.sh" << 'SHEOF'
#!/usr/bin/env bash
LOGS="$1"
CSV="$LOGS/_progress.csv"
SUMMARY="$LOGS/_summary.txt"
[[ -f "$CSV" ]] || echo "timestamp,disk_used_gb,disk_free_gb,ram_used_gb,ram_free_gb,tier1_gb,tier2_files,intl_parquets" > "$CSV"
while true; do
  TS=$(date '+%Y-%m-%d %H:%M:%S')
  USED=$(df -B1G /home/rohanfosse/Bureau/Recherche | awk 'NR==2{print $3}')
  FREE=$(df -B1G /home/rohanfosse/Bureau/Recherche | awk 'NR==2{print $4}')
  RAM_USED=$(free -g | awk 'NR==2{print $3}')
  RAM_FREE=$(free -g | awk 'NR==2{print $7}')
  TIER1=$(du -B1G -s /home/rohanfosse/Bureau/Recherche/imd-national-catalogue/paper_demand/data_collection/tier1_trip_logs 2>/dev/null | cut -f1)
  TIER2=$(ls /home/rohanfosse/Bureau/Recherche/bikeshare-data-explorer/data/status_snapshots/*/*.parquet* 2>/dev/null | wc -l)
  INTL=$(ls /home/rohanfosse/Bureau/Recherche/imd-national-catalogue/paper_demand/data_collection/imd_international/*.parquet 2>/dev/null | wc -l)
  echo "$TS,$USED,$FREE,$RAM_USED,$RAM_FREE,$TIER1,$TIER2,$INTL" >> "$CSV"
  {
    echo "=== STATUS $(date) ==="
    echo "Disk: $USED GB used, $FREE GB free"
    echo "RAM:  $RAM_USED GB used, $RAM_FREE GB free"
    echo "Tier 1 downloads : $TIER1 GB"
    echo "Tier 2 snapshots : $TIER2 parquets"
    echo "IMD-intl         : $INTL parquets"
    echo
    echo "Per-job status:"
    for f in "$LOGS"/*.status; do
      [[ -f "$f" ]] && printf "  %-35s %s\n" "$(basename "$f" .status)" "$(cat "$f")"
    done
  } > "$SUMMARY"
  sleep 1800  # 30 min, denser than v2's 1 h for shorter runs
done
SHEOF
chmod +x "$LOGS/job5_lite_reporter.sh"
nohup bash "$LOGS/job5_lite_reporter.sh" "$LOGS" > "$LOGS/job5_lite_reporter.log" 2>&1 &
master_log "Job 5 hourly reporter           PID=$!"

# ── Banner ────────────────────────────────────────────────────────────────────
echo
echo "─── LITE mode: 3 jobs + reporter ─────────────────────────────"
echo "  Logs        : $LOGS/"
echo "  Live status : cat $LOGS/_summary.txt"
echo "  Progress    : cat $LOGS/_progress.csv"
echo
echo "Skipped on this machine (15 GB RAM is too tight):"
echo "  ✗ Job 4 d3 NYC benchmark        : run on demand from desk"
echo "  ✗ Job 4 d4 cross-city transfer  : same"
echo "  ✗ Job 5 storage compactor       : not urgent for 5-day run"
echo
echo "What WILL run for 5 days:"
echo "  ✓ GBFS polling daemon (43 FR cities × 1 poll/min)"
echo "  ✓ Tier 1 download (5 US networks × 24 months, sequential)"
echo "  ✓ IMD-international (185 candidate cities via OSM + elevation)"
echo "  ✓ Reporter every 30 min to _summary.txt / _progress.csv"
echo
echo "Stop everything :"
echo "  rm $LOCK ; pkill -f '_safe_run|collect_status|tier1_down|imd_intl|reporter'"
echo
