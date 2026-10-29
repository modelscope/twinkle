#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python_bin=${PYTHON_BIN:-python}
model=${MODEL_PATH:?Set MODEL_PATH to the local Qwen3.5-4B base checkpoint}
data=${DATA_ROOT:?Set DATA_ROOT to the prepared joint JSONL directory}
source_root=$(cd ../../.. && pwd)
export PYTHONPATH="$PWD:$source_root/src:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:?Assign four NPU devices}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HCCL_CONNECT_TIMEOUT=300 HCCL_EXEC_TIMEOUT=600
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
stage=${1:-smoke}
output=${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new experiment directory}
case "$stage" in
  smoke) extra=(--steps 2 --save-every 2 --output "$output/smoke") ;;
  resume) extra=(--steps 3 --save-every 3 --resume "$output/smoke/checkpoint-2" --output "$output/resume") ;;
  train) extra=(--steps 120 --save-every 120 --model-only --validation "$data/validation.jsonl" --output "$output/train") ;;
  *) exit 2 ;;
esac
"$python_bin" -m torch.distributed.run --nproc-per-node 4 \
  --master-addr "${MASTER_ADDR:-127.0.0.1}" --master-port "${MASTER_PORT:-29661}" -m twinkle_adapter.train \
  --model "$model" --data "$data/train.jsonl" \
  --schedule-steps 120 --global-batch 16 "${extra[@]}"
