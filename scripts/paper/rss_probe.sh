#!/bin/sh
# Measures the resident-set size of a driver run from outside the process.
# Usage: rss_probe.sh <probe-name> <out-dir> -- <command...>
set -eu

PROBE="$1"; shift
OUT="$1"; shift
if [ "${1:-}" = "--" ]; then
  shift
fi
if [ $# -eq 0 ]; then
  echo "usage: rss_probe.sh <probe-name> <out-dir> -- <command...>" >&2
  exit 2
fi

mkdir -p "$OUT"
LOG="$OUT/poll_log.csv"
echo "timestamp_iso,elapsed_s,workingset_bytes,workingset_mb" > "$LOG"

START=$(date +%s)
"$@" &
PID=$!
PEAK=0
SAMPLES=0

while kill -0 "$PID" 2>/dev/null; do
  KB=$(ps -o rss= -p "$PID" 2>/dev/null | tr -d ' ' || echo 0)
  [ -z "$KB" ] && KB=0
  BYTES=$((KB * 1024))
  [ "$BYTES" -gt "$PEAK" ] && PEAK=$BYTES
  NOW=$(date +%s)
  ELAPSED=$((NOW - START))
  printf '%s,%s,%s,%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$ELAPSED" "$BYTES" \
    "$(awk -v b="$BYTES" 'BEGIN { printf "%.2f", b / 1048576 }')" >> "$LOG"
  SAMPLES=$((SAMPLES + 1))
  sleep 2
done

wait "$PID" || EXIT=$?
EXIT=${EXIT:-0}
DURATION=$(( $(date +%s) - START ))
ESCAPED_CLI=$(printf '%s' "$*" | sed 's/\\/\\\\/g; s/"/\\"/g')

cat > "$OUT/summary.json" <<JSON
{
  "probe": "$PROBE",
  "cli": "$ESCAPED_CLI",
  "peak_rss_bytes": $PEAK,
  "peak_rss_gb": $(awk -v b="$PEAK" 'BEGIN { printf "%.4f", b / 1073741824 }'),
  "n_samples": $SAMPLES,
  "duration_s": $DURATION,
  "exit_code": $EXIT
}
JSON
exit "$EXIT"
