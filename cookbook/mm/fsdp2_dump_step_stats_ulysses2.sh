CUDA_VISIBLE_DEVICES=0,1 \
ULYSSES_SIZE=2 \
TWINKLE_COMPARE_SEED=1234 \
TWINKLE_COMPARE_STEP=0 \
TWINKLE_COMPARE_OUTPUT=compare_step_stats_ulysses2.pt \
torchrun --nproc_per_node=2 fsdp2_dump_step_stats.py
