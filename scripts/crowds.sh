#!/usr/bin/env bash
# Do pedestrians change what federation buys? Rovers share their arenas with 0 to
# 7 scripted pedestrians (a different count per rover, shuffled per seed).
#
#   ramms    RAMMS with frame-synchronized front cameras, 3 seeds x 300 s; trained
#            on features, camera, and both, as in ramms_camera.sh
#   mujoco   MuJoCo, 5 seeds x 600 s, features only. Arena layouts and seeds match
#            data/sweep, so results/sweep-e5 is the pedestrian-free comparison.
#
# RAMMS collection runs first, on a quiet CPU (see ramms_camera.sh); all training
# follows. Needs the RAMMS editor running with the URLab bridge.
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=.venv/bin
OUT=results/crowds
mkdir -p "$OUT/logs"
log() { echo "[$(date '+%F %T')] $*"; }
PEDS=(--pedestrians 0 7)
RAMMS_SEEDS=(0 1 2)
COMMON=(--rounds 10 --local-epochs 5 --proximal-mus --finetune-epochs 5 --evaluate-every 2 --jobs 2)

for seed in "${RAMMS_SEEDS[@]}"; do
  if [ -f "data/crowds-ramms/seed_$seed/meta.json" ]; then log "skip collect seed $seed (done)"; continue; fi
  log "start RAMMS collect seed $seed"
  if "$BIN/ramms-fleet-collect" --backend ramms --camera --rovers 8 --seconds 300 --seed "$seed" "${PEDS[@]}" \
      --out "data/crowds-ramms/seed_$seed" > "$OUT/logs/collect_ramms_$seed.log" 2>&1; then log "done collect seed $seed"
  else log "FAILED collect seed $seed"; exit 1; fi
done

sweep() {  # name, extra args...
  local name=$1; shift
  log "start $name"
  if "$BIN/ramms-fleet-sweep" "$@" > "$OUT/$name.log" 2>&1; then log "done $name"; else log "FAILED $name (see $OUT/$name.log)"; fi
}

sweep mujoco --seeds 0 1 2 3 4 "${COMMON[@]}" "${PEDS[@]}" --data-root data/crowds-mujoco --out "$OUT/mujoco"
for inputs in features camera both; do
  sweep "ramms-$inputs" --seeds "${RAMMS_SEEDS[@]}" "${COMMON[@]}" --inputs "$inputs" \
    --data-root data/crowds-ramms --out "$OUT/ramms-$inputs"
done

"$BIN/ramms-fleet-compare" no-pedestrians=results/sweep-e5 pedestrians=$OUT/mujoco --out "$OUT/mujoco.md" \
  > /dev/null 2>&1 && log "wrote $OUT/mujoco.md"
"$BIN/ramms-fleet-compare" features=$OUT/ramms-features camera=$OUT/ramms-camera both=$OUT/ramms-both \
  --out "$OUT/ramms.md" > /dev/null 2>&1 && log "wrote $OUT/ramms.md"
log "crowd experiment finished"
