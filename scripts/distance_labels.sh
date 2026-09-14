#!/usr/bin/env bash
# Distance-based labels: positive when the bumper hits within 0.15 m of travel
# instead of within 0.5 s (the same thing at the default 0.3 m/s). Relabels the
# datasets from the overnight run, so no new collection, and compares against
# the time-labelled results.
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=.venv/bin
OUT=results/distance
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*"; }
COMMON=(--seeds 0 1 2 3 4 --rounds 10 --local-epochs 5 --evaluate-every 2 --finetune-epochs 5 --jobs 2
        --proximal-mus --label distance --horizon-m 0.15)

sweep() {  # name, data root, extra args...
  local name=$1 data=$2; shift 2
  log "start $name"
  if "$BIN/ramms-fleet-sweep" "${COMMON[@]}" --data-root "$data" --out "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1
  then log "done $name"; else log "FAILED $name (see $OUT/$name.log)"; fi
}

sweep clutter data/sweep
sweep speed data/hetero-speed
sweep full data/hetero-full
sweep noise data/hetero-noise
"$BIN/ramms-fleet-compare" time-clutter=results/sweep-e5 distance-clutter=$OUT/clutter time-speed=results/hetero-speed \
  distance-speed=$OUT/speed time-noise=results/hetero-noise distance-noise=$OUT/noise time-full=results/hetero-full \
  distance-full=$OUT/full --out "$OUT/comparison.md" > /dev/null 2>&1 && log "wrote $OUT/comparison.md"
log "distance label experiment finished"
