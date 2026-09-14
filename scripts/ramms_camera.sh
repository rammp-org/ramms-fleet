#!/usr/bin/env bash
# Does the front camera help? Collect RAMMS runs with frame-synchronized camera
# images, then train the same methods on rangefinders and IMU only, camera only,
# and both, on identical samples.
#
# Needs the RAMMS editor running with the URLab bridge, 'Use Less CPU when in
# Background' off, and nothing else loading the CPU during collection.
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=.venv/bin
DATA=data/ramms-camera
OUT=results/ramms-camera
mkdir -p "$OUT/logs"
log() { echo "[$(date '+%F %T')] $*"; }
SEEDS=(0 1 2 3 4)
SECONDS_PER_RUN=300

for seed in "${SEEDS[@]}"; do
  if [ -f "$DATA/seed_$seed/meta.json" ]; then log "skip collect seed $seed (done)"; continue; fi
  log "start collect seed $seed"
  if "$BIN/ramms-fleet-collect" --backend ramms --camera --rovers 8 --seconds "$SECONDS_PER_RUN" --seed "$seed" \
      --out "$DATA/seed_$seed" > "$OUT/logs/collect_$seed.log" 2>&1; then log "done collect seed $seed"
  else log "FAILED collect seed $seed"; exit 1; fi
done

for inputs in features camera both; do
  log "start sweep $inputs"
  if "$BIN/ramms-fleet-sweep" --seeds "${SEEDS[@]}" --rounds 10 --local-epochs 5 --proximal-mus --finetune-epochs 5 \
      --evaluate-every 2 --jobs 2 --inputs "$inputs" --data-root "$DATA" --out "$OUT/$inputs" > "$OUT/$inputs.log" 2>&1
  then log "done sweep $inputs"; else log "FAILED sweep $inputs"; fi
done
"$BIN/ramms-fleet-compare" features=$OUT/features camera=$OUT/camera both=$OUT/both --out "$OUT/comparison.md" \
  > /dev/null 2>&1 && log "wrote $OUT/comparison.md"
log "camera experiment finished"
