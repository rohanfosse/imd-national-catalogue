#!/usr/bin/env bash
# vacation_status.sh — one-shot summary of the vacation orchestrator.
# Safe to run as often as you like.  Reads files only.

LOGS="/home/rohanfosse/Bureau/Recherche/imd-national-catalogue/paper_demand/scripts/_vacation_logs"

if [[ -f "$LOGS/_STATUS_ALL" ]]; then
  cat "$LOGS/_STATUS_ALL"
else
  echo "no status file yet ($LOGS/_STATUS_ALL not found)"
  echo "is the orchestrator running?  ps -ef | grep vacation_orchestrator"
fi

echo
echo "── recent IMD-intl failures (last 10) ───────────────────────"
tail -10 "$LOGS/job3_imd_intl_failures.txt" 2>/dev/null || echo "  none yet"
echo
echo "── recent IMD-intl successes (last 5) ───────────────────────"
tail -5 "$LOGS/job3_imd_intl_done.txt" 2>/dev/null || echo "  none yet"
