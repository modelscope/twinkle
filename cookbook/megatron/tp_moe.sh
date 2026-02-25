CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 tp_moe.py
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup torchrun --nproc_per_node=4 /mnt/nas2/hujinghan.hjh/twinkle/cookbook/megatron/tp_moe.py > tp_moe.log 2>&1 &

