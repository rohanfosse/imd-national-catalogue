#!/usr/bin/env bash
# _safe_run.sh — wraps a single long-running command with retry, disk
# guard, and structured logging.  Used by vacation_orchestrator_v2.sh.
#
# Usage:  _safe_run.sh <job_name> <max_retries> <min_free_gb> "<command>"
#
# Behaviour:
#   - Pauses if free disk falls below min_free_gb.
#   - Retries with exponential back-off when the wrapped command exits non-zero.
#   - Heartbeat: writes timestamp + PID to <LOGS>/<job_name>.heartbeat
#     every 60 s while the wrapped command is alive.
#   - Status: writes <LOGS>/<job_name>.status with one of
#     RUNNING / PAUSED / FAILED / DONE.

set -u

JOB="${1:?usage: _safe_run.sh <job> <max_retries> <min_free_gb> '<command>'}"
MAX_RETRIES="${2:-5}"
MIN_FREE_GB="${3:-30}"
CMD="${4:?missing command}"

LOGS="/home/rohanfosse/Bureau/Recherche/imd-national-catalogue/paper_demand/scripts/_vacation_logs"
mkdir -p "$LOGS"
LOG="$LOGS/${JOB}.log"
STATUS="$LOGS/${JOB}.status"
HEARTBEAT="$LOGS/${JOB}.heartbeat"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
write_status() { echo "$1" > "$STATUS"; }

write_status "STARTING"
echo "[$(stamp)] === $JOB starting (max_retries=$MAX_RETRIES, min_free=$MIN_FREE_GB GB) ===" >> "$LOG"

# disk-space guard: blocks until enough free space (or hard cap of 30 min)
wait_for_disk() {
  local waited=0
  while :; do
    local free
    free=$(df -B1G /home/rohanfosse/Bureau/Recherche | awk 'NR==2{print $4}')
    if [[ "$free" -ge "$MIN_FREE_GB" ]]; then
      return 0
    fi
    write_status "PAUSED_DISK_${free}GB"
    echo "[$(stamp)] paused, free=${free} GB < ${MIN_FREE_GB} GB, sleep 60s" >> "$LOG"
    sleep 60
    waited=$((waited + 60))
    if [[ "$waited" -ge 1800 ]]; then
      echo "[$(stamp)] disk pressure persistent (>30 min). aborting." >> "$LOG"
      write_status "FAILED_DISK_FULL"
      return 1
    fi
  done
}

# heartbeat loop in the background
(
  while [[ -f "$STATUS" ]] && [[ "$(cat "$STATUS" 2>/dev/null)" != "DONE" ]] && \
        [[ "$(cat "$STATUS" 2>/dev/null)" != "FAILED" ]] && \
        [[ "$(cat "$STATUS" 2>/dev/null)" != "FAILED_DISK_FULL" ]]; do
    echo "$(stamp) $$" > "$HEARTBEAT"
    sleep 60
  done
) &
HB_PID=$!

trap 'kill $HB_PID 2>/dev/null; write_status "INTERRUPTED"; exit 130' INT TERM

attempt=1
while [[ "$attempt" -le "$MAX_RETRIES" ]]; do
  wait_for_disk || { kill $HB_PID 2>/dev/null; exit 1; }
  write_status "RUNNING_ATTEMPT_${attempt}"
  echo "[$(stamp)] attempt $attempt / $MAX_RETRIES :: $CMD" >> "$LOG"

  set +e
  eval "$CMD" >> "$LOG" 2>&1
  rc=$?
  set -e

  if [[ "$rc" -eq 0 ]]; then
    write_status "DONE"
    echo "[$(stamp)] === $JOB done (rc=0) ===" >> "$LOG"
    kill $HB_PID 2>/dev/null
    exit 0
  fi

  echo "[$(stamp)] attempt $attempt failed rc=$rc, sleep $((30 * attempt)) s" >> "$LOG"
  write_status "RETRYING_AFTER_RC_${rc}_ATTEMPT_${attempt}"
  sleep $((30 * attempt))  # 30, 60, 90, 120, 150 s back-off
  attempt=$((attempt + 1))
done

write_status "FAILED"
echo "[$(stamp)] === $JOB FAILED after $MAX_RETRIES attempts ===" >> "$LOG"
kill $HB_PID 2>/dev/null
exit 1
