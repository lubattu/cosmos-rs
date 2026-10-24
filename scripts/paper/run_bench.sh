#!/bin/sh
# Runs the benchmark cell. Pass --quick or --smoke for the shortened
# measurement set.
set -eu

OUT="runs/bench"
EXTRA=""

while [ $# -gt 0 ]; do
  case "$1" in
    --quick) EXTRA="$EXTRA --quick" ;;
    --smoke) EXTRA="$EXTRA --smoke" ;;
    --out) OUT="$2"; shift ;;
    --config) EXTRA="$EXTRA --config $2"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

set -- run -p cosmos-paper --release -- bench --out "$OUT" $EXTRA
echo "cargo $*"
exec cargo "$@"
