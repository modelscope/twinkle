#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 ray start --head --port=6379 --num-gpus=4 --disable-usage-stats --include-dashboard=false
CUDA_VISIBLE_DEVICES=4,5,6,7 ray start --address=127.0.0.1:6379 --num-gpus=4
CUDA_VISIBLE_DEVICES="" ray start --address=127.0.0.1:6379 --num-gpus=0
exec twinkle-server launch -c "$(dirname "${BASH_SOURCE[0]}")/server_config.yaml"
