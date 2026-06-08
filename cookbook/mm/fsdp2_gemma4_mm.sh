export CUDA_VISIBLE_DEVICES=0,1

torchrun --nnodes=1 --nproc_per_node=2 fsdp2_gemma4_mm.py
