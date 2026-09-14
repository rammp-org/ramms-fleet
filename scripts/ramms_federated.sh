#!/usr/bin/env bash
# Federated training on data collected inside RAMMS, and cross-simulator scoring.
#
# 1. Collect seeds 0-4 in RAMMS (same arenas as the MuJoCo sweep in data/sweep).
#    Needs the RAMMS editor running with the URLab bridge.
# 2. Sweep: local-only, centralized, FedAvg, FedAvg + fine-tune on the RAMMS data.
# 3. Score MuJoCo-trained models on RAMMS data and RAMMS-trained models on
#    MuJoCo data, per seed.
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=.venv/bin
OUT=results/ramms
mkdir -p "$OUT/logs"
log() { echo "[$(date '+%F %T')] $*"; }
SEEDS=(0 1 2 3 4)

for seed in "${SEEDS[@]}"; do
  if [ -f "data/ramms/seed_$seed/meta.json" ]; then log "skip collect seed $seed (done)"; continue; fi
  log "start collect seed $seed"
  if "$BIN/ramms-fleet-collect" --backend ramms --rovers 8 --seconds 600 --seed "$seed" --out "data/ramms/seed_$seed" \
      > "$OUT/logs/collect_$seed.log" 2>&1; then log "done collect seed $seed"; else log "FAILED collect seed $seed"; exit 1; fi
done

log "start sweep"
if "$BIN/ramms-fleet-sweep" --seeds "${SEEDS[@]}" --rounds 10 --local-epochs 5 --proximal-mus --finetune-epochs 5 \
    --evaluate-every 2 --jobs 2 --data-root data/ramms --out "$OUT" > "$OUT/sweep.log" 2>&1
then log "done sweep"; else log "FAILED sweep"; exit 1; fi

log "start transfer scoring"
for pair in "mujoco-to-ramms:results/sweep-e5:data/ramms" "ramms-to-mujoco:$OUT:data/sweep"; do
  IFS=: read -r name models data <<< "$pair"
  for seed in "${SEEDS[@]}"; do
    dir="results/transfer/$name/seed_$seed"
    mkdir -p "$dir"
    for method in local centralized fedavg; do ln -sfn "$PWD/$models/seed_$seed/$method" "$dir/$method"; done
    "$BIN/ramms-fleet-eval" --data "$data/seed_$seed" --results "$dir" > "$dir/eval.log" 2>&1 || log "FAILED transfer $name seed $seed"
  done
  "$BIN/ramms-fleet-sweep" --seeds "${SEEDS[@]}" --summarize-only --out "results/transfer/$name" > "results/transfer/$name.log" 2>&1
done
"$BIN/ramms-fleet-compare" mujoco=results/sweep-e5 ramms=$OUT mujoco-to-ramms=results/transfer/mujoco-to-ramms \
  ramms-to-mujoco=results/transfer/ramms-to-mujoco --out "$OUT/comparison.md" > /dev/null 2>&1 && log "wrote $OUT/comparison.md"
log "ramms federated experiment finished"
