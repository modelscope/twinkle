export RAY_ROTATION_MAX_BYTES=1024
export RAY_ROTATION_BACKUP_COUNT=1
CUDA_VISIBLE_DEVICES=0,1,2,3 ray start --head --port=6379 --num-gpus=4 --disable-usage-stats --temp-dir=/dashscope/caches/application/ray_logs
CUDA_VISIBLE_DEVICES="" ray start --address=127.0.0.1:6379 --num-gpus=0
python server.py
