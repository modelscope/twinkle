#!/bin/bash
# DPO Training Script for Ray Mode
#
# This script launches DPO (Direct Preference Optimization) training using Ray
# for distributed training across multiple GPUs.
#
# Usage:
#   ./dpo.sh                    # Default settings (8 GPUs: 4 policy + 4 ref)
#   ./dpo.sh simpo              # Use SimPO (no reference model needed)
#   ./dpo.sh orpo               # Use ORPO (no reference model needed)
#
# Environment variables can be set to customize training:
#   MODEL_ID          - Model to train (default: ms://Qwen/Qwen3.5-4B)
#   DATASET_ID        - Preference dataset (default: UltraFeedback)
#   MODEL_GPUS        - GPUs for policy model (default: 4)
#   REF_MODEL_GPUS    - GPUs for reference model (default: 4)
#   USE_REFERENCE_MODEL - Use reference model (default: 1)
#   BATCH_SIZE        - Global batch size (default: 8)
#   MAX_STEPS         - Training steps (default: 1000)
#   LR                - Learning rate (default: 5e-6)
#   DPO_BETA          - DPO beta parameter (default: 0.1)
#   LOSS_TYPE         - Loss variant: sigmoid/hinge/ipo/simpo/orpo/cpo (default: sigmoid)

set -e

# Parse command line argument for loss type
LOSS_TYPE_ARG=${1:-sigmoid}

# Set default environment variables if not already set
export MODEL_ID=${MODEL_ID:-"ms://Qwen/Qwen3.5-4B"}
export DATASET_ID=${DATASET_ID:-"ms://argilla/ultrafeedback-binarized-preferences-cleaned"}
export MODEL_GPUS=${MODEL_GPUS:-4}
export BATCH_SIZE=${BATCH_SIZE:-8}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
export MAX_STEPS=${MAX_STEPS:-1000}
export LR=${LR:-5e-6}
export DPO_BETA=${DPO_BETA:-0.1}
export SAVE_STEPS=${SAVE_STEPS:-100}
export MAX_LENGTH=${MAX_LENGTH:-2048}

# Set loss type from argument or environment
export LOSS_TYPE=${LOSS_TYPE:-$LOSS_TYPE_ARG}

# Reference-free losses don't need reference model
if [[ "$LOSS_TYPE" == "simpo" || "$LOSS_TYPE" == "orpo" || "$LOSS_TYPE" == "cpo" ]]; then
    export USE_REFERENCE_MODEL=${USE_REFERENCE_MODEL:-0}
    export REF_MODEL_GPUS=${REF_MODEL_GPUS:-0}
    echo "Using $LOSS_TYPE loss (reference-free)"
else
    export USE_REFERENCE_MODEL=${USE_REFERENCE_MODEL:-1}
    export REF_MODEL_GPUS=${REF_MODEL_GPUS:-4}
    echo "Using $LOSS_TYPE loss with reference model"
fi

# Calculate total GPUs
if [[ "$USE_REFERENCE_MODEL" == "1" && "$REF_MODEL_GPUS" -gt 0 ]]; then
    TOTAL_GPUS=$((MODEL_GPUS + REF_MODEL_GPUS))
else
    TOTAL_GPUS=$MODEL_GPUS
fi

echo "=========================================="
echo "DPO Training Configuration"
echo "=========================================="
echo "Model:              $MODEL_ID"
echo "Dataset:            $DATASET_ID"
echo "Loss Type:          $LOSS_TYPE"
echo "DPO Beta:           $DPO_BETA"
echo "Policy GPUs:        $MODEL_GPUS"
echo "Reference GPUs:     $REF_MODEL_GPUS"
echo "Total GPUs:         $TOTAL_GPUS"
echo "Batch Size:         $BATCH_SIZE"
echo "Micro Batch Size:   $MICRO_BATCH_SIZE"
echo "Max Steps:          $MAX_STEPS"
echo "Learning Rate:      $LR"
echo "Max Length:         $MAX_LENGTH"
echo "Save Steps:         $SAVE_STEPS"
echo "=========================================="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run training
python "$SCRIPT_DIR/dpo.py"
