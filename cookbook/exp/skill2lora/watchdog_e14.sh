#!/bin/bash
# One-shot watchdog (2026-07-28): the running ablate12 launcher holds the OLD plan
# (E7 -> E8). E14 (reward SNR ablation) was inserted after E7 in config.py, so when E7
# finishes (DONE.json) OR the launcher dies (E7 crash), swap to a fresh launcher that
# reads the new plan: completed arms skip via DONE.json, so it starts E14 directly.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DONE="$HERE/output.ablate12/E7_rl_ab_on_pitfall/DONE.json"
PIDF=/tmp/ablate12.pid
LOG="$HERE/watchdog_e14.log"

echo "[watchdog $(date +%H:%M:%S)] waiting for E7 DONE.json or launcher exit" >> "$LOG"
while true; do
    [ -f "$DONE" ] && { echo "[watchdog $(date +%H:%M:%S)] E7 DONE.json found" >> "$LOG"; break; }
    OLD=$(cat "$PIDF" 2>/dev/null || echo "")
    if [ -n "$OLD" ] && ! kill -0 "$OLD" 2>/dev/null; then
        echo "[watchdog $(date +%H:%M:%S)] launcher $OLD died without E7 DONE (crash?); restarting anyway" >> "$LOG"
        break
    fi
    sleep 60
done

sleep 10
OLD=$(cat "$PIDF" 2>/dev/null || echo "")
[ -n "$OLD" ] && kill "$OLD" 2>/dev/null && echo "[watchdog] killed old launcher $OLD" >> "$LOG"
# kill any experiment python the old launcher may have just started (E8 race window)
pkill -f "skill_ablate.main" 2>/dev/null && echo "[watchdog] killed stray skill_ablate.main" >> "$LOG"

# wait for the 8 GPUs to drain (engine teardown), max 15 min
for i in $(seq 1 90); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    [ "${USED:-1}" -lt 1000 ] && break
    sleep 10
done
echo "[watchdog $(date +%H:%M:%S)] GPUs drained (used=${USED:-?}MiB); relaunching" >> "$LOG"

cd "$HERE"
nohup bash run_ablate12.sh > run_ablate12.nohup.log 2>&1 &
echo $! > "$PIDF"
echo "[watchdog $(date +%H:%M:%S)] new launcher pid=$(cat $PIDF) (plan includes E14 after E7)" >> "$LOG"
