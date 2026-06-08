
# `deepseek-ai/DeepSeek-V4-Flash` uses mixed FP4/FP8 weights.
# Convert the checkpoint before training by following:
# https://gitcode.com/cann/cann-recipes-train/blob/master/llm_pretrain/deepseekv4/README.md#%E6%A8%A1%E5%9E%8B%E6%9D%83%E9%87%8D%E5%87%86%E5%A4%87
# Install `transformers==5.8.0` before running this cookbook.

export DSV4_MODEL_ID="ms://deepseek-ai/DeepSeek-V4-Flash-bf16"
export DATASET_ID="ms://swift/self-cognition"
# The following environment variables are required for multi-node training. Adjust the values according to your cluster setup.
export GLOO_SOCKET_IFNAME="eth0" # Use ifconfig to check the network interface name
export HCCL_SOCKET_IFNAME="eth0"
export HCCL_EXEC_TIMEOUT=1200
export HCCL_CONNECT_TIMEOUT=1200
export NNODES=4
export NUM_GPUS=64
export MASTER_ADDR="node0" # Replace with the IP address or hostname of the master node
export MASTER_PORT=29500 # Replace with an open port on the master node
export HCCL_IF_BASE_PORT=20000

torchrun --nnodes=$NNODES --node_rank=$NODE_RANK --nproc_per_node=16 \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT ep_fsdp2_lora_deepseek_v4.py

#  NODE_RANK=0  OUTPUT_DIR=./output sh ep_fsdp2_lora_deepseek_v4_multinode.sh
#  NODE_RANK=1  OUTPUT_DIR=./output sh ep_fsdp2_lora_deepseek_v4_multinode.sh
#  NODE_RANK=2  OUTPUT_DIR=./output sh ep_fsdp2_lora_deepseek_v4_multinode.sh
#  NODE_RANK=3 OUTPUT_DIR=./output sh ep_fsdp2_lora_deepseek_v4_multinode.sh
