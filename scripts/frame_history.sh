#!/usr/bin/env bash
# Does a camera model need to see motion? Retrains the RAMMS camera runs with the
# whole window of frames stacked as channels ("-history" inputs) instead of the
# latest frame, on the data experiments 7 and 8 already collected: arenas with
# pedestrians, where one frame did not help, and arenas without them.
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=.venv/bin
OUT=results/frame-history
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*"; }
COMMON=(--seeds 0 1 2 --rounds 10 --local-epochs 5 --proximal-mus --finetune-epochs 5 --evaluate-every 2 --jobs 2)

sweep() {  # name, data root, inputs
  log "start $1"
  if "$BIN/ramms-fleet-sweep" "${COMMON[@]}" --inputs "$3" --data-root "$2" --out "$OUT/$1" > "$OUT/$1.log" 2>&1
  then log "done $1"; else log "FAILED $1 (see $OUT/$1.log)"; fi
}

sweep crowds-camera-history data/crowds-ramms camera-history
sweep crowds-both-history data/crowds-ramms both-history
sweep quiet-camera-history data/ramms-camera camera-history
sweep quiet-both-history data/ramms-camera both-history

"$BIN/ramms-fleet-compare" camera=results/crowds/ramms-camera both=results/crowds/ramms-both \
  camera-history=$OUT/crowds-camera-history both-history=$OUT/crowds-both-history \
  --out "$OUT/crowds.md" > /dev/null 2>&1 && log "wrote $OUT/crowds.md"
"$BIN/ramms-fleet-compare" camera=results/ramms-camera/camera both=results/ramms-camera/both \
  camera-history=$OUT/quiet-camera-history both-history=$OUT/quiet-both-history \
  --out "$OUT/quiet.md" > /dev/null 2>&1 && log "wrote $OUT/quiet.md"
log "frame history experiment finished"
