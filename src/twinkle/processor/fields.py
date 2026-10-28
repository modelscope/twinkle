"""InputProcessor field handling: collation, filtering and packed restoration.

These rules cover the processor's supported encoded inputs. Raw media is encoded
by templates first; model/loss providers declare any additional sequence keys.
SP's execution defaults live separately in its data_keys module.
"""

from twinkle.data_format.sequence_fields import DataKeySpec, declared_key_specs

VLM_CONCAT_FIELDS = {
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
    "input_features",
    "input_features_mask",
    "feature_attention_mask",
    "grid_thws",
}

PROCESSOR_PASSTHROUGH_KEYS = ("cache_position", "padding_free")

# Single source for built-in sequence layouts. SP derives execution defaults
# from these declarations; the processor never schedules SP transforms.
COMMON_INPUT_SPECS = (
    DataKeySpec("input_ids", split_dim=1, pad_value="tokenizer", stage="forward"),
    DataKeySpec("inputs_embeds", split_dim=1, pad_value=0.0, stage="forward"),
    DataKeySpec("position_ids", pad_value=-1, stage="forward"),
    DataKeySpec("attention_mask", split_dim=1, stage="forward"),
    DataKeySpec("labels", pad_value=-100),
    DataKeySpec("completion_mask"),
    DataKeySpec("loss_scale", pad_value=0.0),
)
_FORWARD_KEYS = tuple(spec.name for spec in COMMON_INPUT_SPECS if spec.stage == "forward")
TRANSFORMERS_INPUT_KEYS = frozenset(spec.name for spec in COMMON_INPUT_SPECS) | frozenset(
    {
        "channel",
        "cu_seq_lens_q",
        "cu_seq_lens_k",
        "cu_seqlens_q",
        "cu_seqlens_kv",
        "max_length_q",
        "max_length_k",
        "packed_seq_params",
        "routed_experts",
        "replay_mask",
        "mm_token_type_ids",
        "second_per_grid_ts",
        *VLM_CONCAT_FIELDS,
        *PROCESSOR_PASSTHROUGH_KEYS,
    }
)

# Numeric padding for existing processor/CP paths. Processor has no tokenizer:
# preserve its input_ids fallback of 0; SP resolves "tokenizer" in its context.
PADDING_VALUES = {
    spec.name: 0 if spec.pad_value == "tokenizer" else spec.pad_value
    for spec in COMMON_INPUT_SPECS
}
PADDING_VALUES.update({
    "mm_token_type_ids": 0,
    "length": -1,
    "pixel_values": 0.0,
    "image_grid_thw": 0,
    "pixel_values_videos": 0.0,
    "video_grid_thw": 0,
    "input_features": 0.0,
    "feature_attention_mask": 0,
    "routed_experts": 0,
    "replay_mask": 0,
})

# Only default sequence-aligned outputs need unpacking here. Task-specific
# outputs (e.g. PPO values) are declared by their model/loss provider.
COMMON_OUTPUT_SPECS = tuple(
    DataKeySpec(name, split_dim=1, side="output", stage="output") for name in ("logps", "entropies", "logits")
)


def validate_input_key_spec(spec):
    """Validate provider and direct-registration inputs with the same rules."""
    if not isinstance(spec, DataKeySpec):
        raise TypeError("Expected a DataKeySpec")
    if spec.side != "input" or spec.stage not in ("auxiliary", "forward"):
        raise ValueError("Register only auxiliary or model-forward input keys here")
    if spec.name in _FORWARD_KEYS:
        raise ValueError(f"{spec.name!r} is owned by the model forward hook")
    if spec.name in PROCESSOR_PASSTHROUGH_KEYS:
        raise ValueError(f"{spec.name!r} is a special processor/model argument, not an SP data key")
    if not spec.model_input and "forward" in (spec.pad_at, spec.split_at):
        raise ValueError(
            f"Loss-only input {spec.name!r} cannot use the forward boundary; "
            "use prepare, or set model_input=True for a model argument"
        )


def resolve_key_specs(*providers):
    """Merge provider schemas independently of any SP strategy or execution registry."""
    inputs = {spec.name: spec for spec in COMMON_INPUT_SPECS}
    outputs = {spec.name: spec for spec in COMMON_OUTPUT_SPECS}
    declared = {"input": set(), "output": set()}
    layout_fields = ("name", "split_dim", "pad_value", "side", "stage", "model_input")
    for provider in providers:
        for side, registry in (("input", inputs), ("output", outputs)):
            for spec in declared_key_specs(provider, side):
                if side == "input":
                    validate_input_key_spec(spec)
                else:
                    if not isinstance(spec, DataKeySpec):
                        raise TypeError("Expected a DataKeySpec")
                    if spec.side != side:
                        raise ValueError(f"Expected an {side} key rule")
                previous = registry.get(spec.name)
                if previous is not None and previous != spec:
                    # Defaults describe layouts only. A provider may add its SP
                    # operations without changing a built-in field's layout.
                    if spec.name in declared[side] or any(
                        getattr(previous, field) != getattr(spec, field) for field in layout_fields
                    ):
                        raise ValueError(f"Conflicting {side} field rule for {spec.name!r}")
                registry[spec.name] = spec
                declared[side].add(spec.name)
    return tuple(inputs.values()), tuple(outputs.values())
