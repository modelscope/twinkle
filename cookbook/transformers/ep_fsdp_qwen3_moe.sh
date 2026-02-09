# EP + FSDP2 (Transformers MoE) example.
# With expert_parallel enabled, expert parameters are sharded across the EP dimension.
# Non-expert parameters are sharded by FSDP (across world_size).
# Officially validated scope: qwen3_moe_like models (for example, Qwen3-30B-A3B).
# Other MoE models may work if their MoE blocks expose: `experts` + `gate/router` + `top_k` (or `num_experts_per_tok`).
# EP runtime constraints: `num_experts % ep_world_size == 0`.
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ep_fsdp_qwen3_moe.py
