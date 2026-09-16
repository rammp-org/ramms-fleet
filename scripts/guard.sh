#!/usr/bin/env bash
# Does the prediction make the fleet safer? Each rover drives with its own model
# guarding the exploration policy: above a risk threshold it brakes and turns away.
# Compares no guard, each rover's local-only model, the FedAvg model, and the
# centralized model, in the arenas those models were trained on.
#
# Safety alone is trivial (stand still), so runs report distance covered too.
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=.venv/bin
OUT=results/guard
SEEDS=(0 1 2 3 4)
SECONDS_PER_RUN=600
mkdir -p "$OUT/logs"
log() { echo "[$(date '+%F %T')] $*"; }

run() {  # condition, method, seed, model (may be empty), extra args...
  local condition=$1 method=$2 seed=$3 model=$4; shift 4
  local out="$OUT/$condition/$method/seed_$seed.json"
  [ -f "$out" ] && return 0
  local args=(--rovers 8 --seconds "$SECONDS_PER_RUN" --seed "$seed" --out "$out" "$@")
  [ -n "$model" ] && args+=(--model "$model")
  OMP_NUM_THREADS=2 "$BIN/ramms-fleet-guard" "${args[@]}" > "$OUT/logs/${condition}_${method}_$seed.log" 2>&1
}

for seed in "${SEEDS[@]}"; do
  # Clutter-only arenas, models from the 5 x 10 sweep (experiment 3).
  CLUTTER=results/sweep-e5/seed_$seed
  run clutter none "$seed" "" &
  run clutter local "$seed" "$CLUTTER/local/rover_{rover}.pt" &
  run clutter fedavg "$seed" "$CLUTTER/fedavg/model.pt" &
  run clutter centralized "$seed" "$CLUTTER/centralized/model.pt" &
  wait
  # Arenas with pedestrians, models from experiment 8.
  CROWD=results/crowds/mujoco/seed_$seed
  PEDS=(--pedestrians 0 7)
  run crowd none "$seed" "" "${PEDS[@]}" &
  run crowd local "$seed" "$CROWD/local/rover_{rover}.pt" "${PEDS[@]}" &
  run crowd fedavg "$seed" "$CROWD/fedavg/model.pt" "${PEDS[@]}" &
  run crowd centralized "$seed" "$CROWD/centralized/model.pt" "${PEDS[@]}" &
  wait
  log "seed $seed done"
done

# How cautious should the guard be? FedAvg in arenas with pedestrians.
for threshold in 0.3 0.7; do
  for seed in "${SEEDS[@]}"; do
    run "crowd-threshold-$threshold" fedavg "$seed" "results/crowds/mujoco/seed_$seed/fedavg/model.pt" \
      --pedestrians 0 7 --threshold "$threshold" &
  done
  wait
  log "threshold $threshold done"
done

"$BIN/ramms-fleet-guard-summary" "$OUT" --out "$OUT/summary.md" > /dev/null && log "wrote $OUT/summary.md"
log "guard experiment finished"
