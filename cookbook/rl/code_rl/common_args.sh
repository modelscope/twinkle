# Shared training hyper-parameters for both backends. Sourced by run_*.sh.
#
# Kept in one place so the two backends stay a controlled comparison: if these
# differed, a change in reward could not be attributed to the execution backend.
# Backend-specific settings live in the run_*.sh scripts and in backends/.
#
# Override anything at invocation, e.g.:
#   sh run_openenv.sh --max-steps 500 --batch-size 8

# Capacity note: the backend must host BATCH_SIZE x NUM_GENERATIONS concurrent
# envs (32 with the defaults below).
#   - OpenEnv:  WORKERS x MAX_CONCURRENT_ENVS in serve.sh (256 by default)
#   - AgentENV: 32 x sandbox memory + ~8GB for the host
#
# Floor: trajectories (batch-size x num-generations) must stay >= --model-gpus,
# or every batch is dropped by the length filter with only a warning.
TRAIN_ARGS="
    --model-id ms://Qwen/Qwen3.5-4B
    --model-gpus 4
    --sampler-gpus 4
    --num-generations 8
    --max-tokens 2048
    --batch-size 4
    --mini-batch-size 8
    --micro-batch-size 2
    --max-steps 1000
    --lr 1e-5
    --lora-r 16
    --save-steps 500
    --adapter-name default
"

# Max tool-calling turns per episode.
export MAX_TURNS="${MAX_TURNS:-6}"
# Concurrent reset/score calls issued from the driver.
export ENV_CONCURRENCY="${ENV_CONCURRENCY:-16}"
