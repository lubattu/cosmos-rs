#!/bin/sh
# Runs the topology cell, then the companion cell. Pass --smoke
# for the shortened form, which checks itself against the documented
# reference.
set -eu

OUT="runs"
DATA_DIR="data"
THREADS=1
EXTRA=""

while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) EXTRA="$EXTRA --smoke" ;;
    --no-download) EXTRA="$EXTRA --no-download" ;;
    --out) OUT="$2"; shift ;;
    --data-dir) DATA_DIR="$2"; shift ;;
    --threads) THREADS="$2"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

set -- run -p cosmos-paper --release -- e4 --out "$OUT" --data-dir "$DATA_DIR" --threads "$THREADS" $EXTRA
echo "cargo $*"
exec cargo "$@"
