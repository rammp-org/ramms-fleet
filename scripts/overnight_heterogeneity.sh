#!/usr/bin/env bash
# Overnight heterogeneity experiment: which rover differences make federated
# learning struggle, and does per-rover fine-tuning recover it?
#
# Conditions (5 seeds, 8 rovers, 10 rounds x 5 local epochs, clutter varies in all):
#   clutter      speed and noise identical        (re-evaluates results/sweep-e5)
#   full         + speed 0.15-0.6 m/s + sensor noise
#   speed        + speed only
#   noise        + sensor noise only
#   full-16      full, with 16 rovers (3 seeds)
# Then renders videos from the full condition and writes comparison tables.
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=.venv/bin
OUT=results/overnight
mkdir -p "$OUT/videos"
log() { echo "[$(date '+%F %T')] $*"; }

COMMON=(--rounds 10 --local-epochs 5 --evaluate-every 2 --finetune-epochs 5 --jobs 2)
SPEED=(--speed 0.15 0.6)
NOISE=(--range-noise 0 0.2 --accel-noise 0 2 --gyro-noise 0 0.3)

sweep() {  # name, extra args...
  local name=$1; shift
  log "start $name"
  if "$BIN/ramms-fleet-sweep" "$@" > "$OUT/$name.log" 2>&1; then log "done $name"; else log "FAILED $name (see $OUT/$name.log)"; fi
}

sweep clutter --seeds 0 1 2 3 4 "${COMMON[@]}" --proximal-mus 0.01 0.1 1.0 --data-root data/sweep --out results/sweep-e5
sweep full --seeds 0 1 2 3 4 "${COMMON[@]}" --proximal-mus 0.01 0.1 "${SPEED[@]}" "${NOISE[@]}" \
  --data-root data/hetero-full --out results/hetero-full

log "start videos"
RUN=data/hetero-full/seed_0
RES=results/hetero-full/seed_0
"$BIN/ramms-fleet-render" overview --data "$RUN" --model "FedAvg=$RES/fedavg/model.pt" \
  --title "FedAvg global model" --seconds 30 --out "$OUT/videos/fleet_fedavg.mp4" >> "$OUT/videos.log" 2>&1
# Close-ups: the rover that gained most from federation, and the slowest and noisiest rovers.
read -r GAIN SLOW NOISY < <("$BIN/python" - "$RES/evaluation.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))["rovers"]
gain = max(rows, key=lambda r: r["fedavg"]["auprc"] - r["local"]["auprc"])["rover"]
slow = min(rows, key=lambda r: r["cruise_speed"])["rover"]
noisy = max(rows, key=lambda r: r["range_noise"])["rover"]
print(gain, slow, noisy)
PY
)
for pair in "gain:$GAIN" "slowest:$SLOW" "noisiest:$NOISY"; do
  label=${pair%%:*}; rover=${pair##*:}
  "$BIN/ramms-fleet-render" compare --data "$RUN" --rover "$rover" \
    --models "Local only=$RES/local/rover_{rover}.pt" "FedAvg=$RES/fedavg/model.pt" \
    --seconds 20 --out "$OUT/videos/compare_${label}_rover${rover}.mp4" >> "$OUT/videos.log" 2>&1
done
log "done videos"

sweep speed --seeds 0 1 2 3 4 "${COMMON[@]}" --proximal-mus 0.01 0.1 "${SPEED[@]}" \
  --data-root data/hetero-speed --out results/hetero-speed
sweep noise --seeds 0 1 2 3 4 "${COMMON[@]}" --proximal-mus 0.01 0.1 "${NOISE[@]}" \
  --data-root data/hetero-noise --out results/hetero-noise

compare() {
  "$BIN/ramms-fleet-compare" clutter=results/sweep-e5 speed=results/hetero-speed noise=results/hetero-noise \
    full=results/hetero-full "$@" --out "$OUT/comparison.md" > /dev/null 2>&1
}
compare && log "wrote $OUT/comparison.md"

sweep full-16 --seeds 0 1 2 "${COMMON[@]}" --jobs 1 --rovers 16 --proximal-mus 0.01 0.1 "${SPEED[@]}" "${NOISE[@]}" \
  --data-root data/hetero-full-16 --out results/hetero-full-16
compare full-16=results/hetero-full-16 && log "wrote $OUT/comparison.md (with 16 rovers)"
log "overnight experiment finished"
