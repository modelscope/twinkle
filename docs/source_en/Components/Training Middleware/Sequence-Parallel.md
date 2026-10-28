# Sequence Parallel (SP)

Transformers SP shards token sequences and runs attention through Ulysses and optional Ring communication. Each model owns an independent `SequenceParallel` context. Process-wide wrappers resolve that context at call time.

## Configuration

```python
from twinkle.model.transformers.strategy.sequence_parallel import SequenceParallelConfig

config = SequenceParallelConfig(enabled=True, ulysses_size=None)
```

A `None` degree is derived from DeviceMesh. The old `gather_logits` option has been removed: output rules and the current loss determine collection.

```python
from twinkle.utils import DeviceMesh

# Eight devices forming two SP=4 groups.
mesh = DeviceMesh.from_sizes(dp_size=8, fsdp_size=1, ulysses_size=4)
```

## Processing responsibilities

1. Processor converts tensors, collates samples, applies ordinary padding and exposes `is_padding_free()` as a layout query. It neither receives nor calls an SP strategy.
2. The model wrapper partitions model inputs and loss inputs. The strategy shards labels and explicitly local auxiliary fields at the prepare boundary.
3. Forward hooks pad/split model token inputs at their declared boundaries. Attention masks retain global layout, including padding masks for single-sample SDPA batches.
4. Attention wrappers perform communication and invoke the configured backend.
5. Model-side `postprocess_task_outputs()` handles task semantics: embedding pooling/normalization, classification/reranker pooling, or token-loss SP collection.
6. Processor's `unpack_packed_sequences()` restores already collected sequences to per-sample layouts before loss calculation.

Model postprocessing interprets task outputs and performs SP collectives; processor postprocessing restores layouts.

## Field ownership

- `data_format/sequence_fields.py` defines `DataKeySpec` and provider contracts, without a global list of all fields.
- `processor/fields.py` owns supported encoded text/media/replay/control inputs and collation/restoration rules. Templates encode raw media first. Internal `length` metadata is not a forward argument.
- `model/transformers/strategy/sequence_parallel/data_keys.py` derives SP execution defaults from the common processor layouts, adding processing boundaries and autograd settings without repeating axes or padding values. It also owns batch-local collections. Media pixels and control arguments are not blindly sharded as tokens.
- Model adapters and losses declare their own extensions.

Processor defaults do not schedule SP operations. Custom loss fields remain available with SP disabled.

Declare extensions on the model, adapter or loss; adding a custom key does not require editing either built-in list. Model-side schema resolution is cached for the current model/strategy/loss instances. Direct input/output registrations and `set_loss()` invalidate the cache. Configure provider declarations before forwarding; if reconfiguring an existing loss instance, call `set_loss()` again. Field rules only pad, split and gather; they do not shift token targets.

```python
from twinkle.data_format.sequence_fields import DataKeySpec

# Global loss input, restored per sample after packing.
input_key_specs = (DataKeySpec('token_targets', pad_value=-100),)

# Explicitly scheduled model-forward input.
model_input_key_specs = (
    DataKeySpec('token_features', split_dim=1, stage='forward',
                pad_at='forward', split_at='forward'),
)
```

`pad_at/split_at` accept `prepare`, `forward`, `model_internal`, `router_replay`, or `None`. `need_pad/need_split` are derived properties. `stage` classifies fields (including `auxiliary` and `output`), rather than naming the execution boundary.

## Loss declarations

CE uses the default collected `logps`. Other objectives may need values or custom sequence outputs; SP cannot infer their requirements from arbitrary Python code.

- `input_key_specs/output_key_specs` describe layout, routing and optional parallel operations. Processor consumes layouts even without SP.
- `sp_gather_keys` lists full-sequence outputs required by the current loss call. Missing required outputs raise an error. A field's `gather_at='loss'` instead requests default collection when that output is present.
- `require_values` requests per-token `values` from the Transformers wrapper; PPOValueLoss then collects them.
- Earlier `sp_input_key_specs/sp_output_key_specs` names are still read for compatibility. New code uses the common names.

PPOValueLoss declares:

```python
require_logps = False
require_values = True
sp_gather_keys = ('values',)
output_key_specs = (
    DataKeySpec('values', split_dim=1, side='output', stage='output', autograd=True),
)
```

Schema defines tensor handling; required-output declarations specify this call's needs. Full vocabulary logits are not gathered unconditionally. Fused CE handles its hidden-state slot separately. Global completion masks, loss weights and default custom targets do not incur an unnecessary split/gather cycle.

## Attention backends

| Backend | Implementation scope |
|---|---|
| FlashAttention2 / FlashAttention3 | Shared SP wrapper and packed/varlen preparation; concrete kernels follow model configuration and require their own dependencies/device capabilities |
| SDPA | Ordinary non-packed batches; padding-free/packed inputs are rejected |
| Derived Ring | FA2/FA3 dispatch is implemented and requires batch_size=1; the FA3 path rejects dropout and ALiBi |

Implementation support does not establish runtime acceptance on every device. Record FA3 validation separately from FA2.

## Model adapters and Megatron

Adapters configure Qwen3-VL DeepStack/mRoPE, Qwen3.5 GatedDeltaNet and MoE router auxiliary loss using the owning model's context.

This page describes Transformers SP. Megatron has its own CP/TP-SP path: `MegatronEmbeddingPatch` handles embeddings, Megatron forward_step handles classification/reranking, and `postprocess_tensor_cp()` reconstructs CP shards. Both paths share processor collation and packed restoration. Shared-path regression does not replace full Megatron training validation.

Initialize `torch.distributed` before distributed execution. Attention-head count and the configured SP degree determine the Ulysses/Ring topology.
