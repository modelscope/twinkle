# Data-parallel workload balancing

`TrainableModel` owns the common DP-balance configuration, planning and local sample exchange.
`dp_balance=True` is the default on TransformersModel, MegatronModel and their multi-LoRA variants.
Subclasses that override a `slice_dp` forward_only/forward_backward entry inherit the dispatch hook.
The constructor calls `_init_dp_balance`; a new backend declares `_dp_balance_backend` and supplies a
matching cost estimator before enabling a different micro-batch layout.

A call is planned before DP dispatch: keep equal per-replica sample counts, reorder every aligned list,
tensor and nested field together, then restore collected per-sample outputs to the caller's order.
TP/PP/CP peers in the same DP replica receive the same shard. Local mode exchanges CPU-serialized
samples and outputs only when reordering is necessary; its communication overhead must be measured.
Nested model entry points are balanced once.

Transformers estimates cost with its fixed/dynamic micro-batch planner. Megatron uses consecutive
fixed micro-batches, per-micro-batch or per-rank padding according to variable_seq_lengths, and CP/SP
length alignment. It keeps the existing micro-batch count/schedule; dynamic batching and incomplete
micro-batches skip balancing. Grouped losses also require a micro-batch layout that keeps groups intact.

Losses opt in only when reordering preserves gradient weights. GRPO and PPO sequence-mean (including
per-token/per-sequence controls such as dual-clip, overlong filtering, rollout importance sampling and
off-policy sequence masking), CE sum, BNPO global and selected pair/group objectives participate.
Entropy-quantile filtering, FIPO and SDAR keep their order. Rank-local token means, batch-dependent
objectives and undeclared custom losses retain their order. A new loss must declare its invariant
`dp_balance_group_size` contract. This is a cost heuristic, not expert-routing balancing or an exact
multimodal cost model. DP=1, uneven rank sizes, raw/group-shaped inputs without a safe length contract,
raw-logit returns and opaque whole-batch reference arguments also skip.

For remote features, pass `input_lengths=[...]` alongside a list of per-sample ObjectRefs. Planning
uses metadata without calling ray.get on the driver; dispatch slices the reference list and each worker
resolves only its shard. Missing metadata leaves opaque inputs unchanged. A whole tensor behind one
ObjectRef cannot be safely reordered and is skipped. This metadata is consumed at the dispatch boundary.
Training metrics are reported through model.calculate_metric as dp_balance/*.

After a reordered Transformers forward_only without micro-batching, supply per-sample loss arguments
on that forward_only call and use the same loss_key in calculate_loss. The model reuses the aligned
arguments and rejects fresh batch arguments. The same holds when micro-batching runs the rank batch as
one micro-batch. Separate calculate_loss after a reordered forward_only split into several
micro-batches is rejected because its cache contains only the last micro-batch; use forward_backward.

Swift exposes the same default-on option as `--dp_balance`; `--dp_balance false` disables it.
Recipes pass only features, training signals and optional lengths; they do not plan or permute batches.
