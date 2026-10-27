"""Shared sequence-field schemas consumed by processors, model wrappers and SP."""

from dataclasses import dataclass

_BOUNDARIES = ("forward", "auxiliary", "model_internal", "router_replay", "output")
_PROCESSING_BOUNDARIES = ("prepare", "forward", "model_internal", "router_replay")
_FORWARD_KEYS = ("input_ids", "inputs_embeds", "position_ids", "attention_mask")


@dataclass(frozen=True)
class DataKeySpec:
    """Describe collation, routing and explicitly scheduled parallel operations.

    Loss-only fields stay global unless pad_at/split_at opt into local processing.
    Packed restoration is independent of sequence-parallel gather and split.
    """

    name: str
    split_dim: int = -1
    pad_value: int | float | str = 0
    roll: bool = False
    side: str = "input"
    stage: str = "auxiliary"
    autograd: bool = False
    pad_at: str | None = None
    split_at: str | None = None
    gather_at: str | None = None
    unpack_packed: bool | None = None
    model_input: bool | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("A sequence field must have a non-empty name")
        if not isinstance(self.split_dim, int):
            raise TypeError("split_dim must be an integer")
        if self.side not in ("input", "output"):
            raise ValueError("side must be input or output")
        if self.stage not in _BOUNDARIES:
            raise ValueError("Invalid SP key stage")
        if (self.side == "output") != (self.stage == "output"):
            raise ValueError("Output keys must use stage=output")
        if isinstance(self.pad_value, str) and (self.name != "input_ids" or self.pad_value != "tokenizer"):
            raise ValueError("Only input_ids can use the tokenizer padding sentinel")
        for operation, boundary in (("padding", self.pad_at), ("split", self.split_at)):
            if boundary is not None and boundary not in _PROCESSING_BOUNDARIES:
                raise ValueError(f"Invalid SP {operation} boundary: {boundary!r}")
        if self.gather_at not in (None, "loss"):
            raise ValueError(f"Invalid SP gather boundary: {self.gather_at!r}")
        if self.side == "output" and (self.need_pad or self.need_split):
            raise ValueError("Output keys cannot request input padding or splitting")
        if self.side == "input" and self.gather_at is not None and not self.need_split:
            raise ValueError("Only sharded input keys can request gathering")
        if self.stage == "forward" and self.gather_at is not None:
            raise ValueError("Model-forward keys cannot request loss gathering")
        if self.roll and not self.need_split:
            raise ValueError("Rolling requires a split boundary")
        if self.model_input is None:
            object.__setattr__(self, "model_input", self.stage == "forward")
        if self.unpack_packed is None:
            object.__setattr__(
                self, "unpack_packed", self.side == "output" or self.gather_at is not None or not self.need_split
            )

    @property
    def need_pad(self):
        return self.pad_at is not None

    @property
    def need_split(self):
        return self.split_at is not None

    def sequence_dim(self, ndim):
        if not -ndim <= self.split_dim < ndim:
            raise ValueError(f"Invalid sequence axis {self.split_dim} for SP key {self.name!r}")
        dim = self.split_dim % ndim
        if dim == 0:
            raise ValueError(f"SP key {self.name!r} cannot split or gather the batch axis")
        return dim


BUILTIN_INPUT_KEYS = (
    DataKeySpec("input_ids", split_dim=1, pad_value="tokenizer", stage="forward", pad_at="forward", split_at="forward"),
    DataKeySpec("inputs_embeds", split_dim=1, stage="forward", pad_at="forward", split_at="forward"),
    DataKeySpec("position_ids", pad_value=-1, stage="forward", pad_at="forward", split_at="forward"),
    DataKeySpec("attention_mask", split_dim=1, stage="forward", pad_at="forward"),
    DataKeySpec("labels", pad_value=-100, pad_at="prepare", split_at="prepare", gather_at="loss", unpack_packed=True),
    # Loss-only global fields still need per-sample restoration after packing.
    DataKeySpec("completion_mask", unpack_packed=True),
    DataKeySpec("loss_scale", unpack_packed=True),
)

# Control arguments are preserved without sequence transforms.
PROCESSOR_PASSTHROUGH_KEYS = ("cache_position", "padding_free")

BUILTIN_OUTPUT_KEYS = (
    DataKeySpec(
        "logps", split_dim=1, side="output", stage="output", gather_at="loss", autograd=True, unpack_packed=True
    ),
    DataKeySpec(
        "entropies", split_dim=1, side="output", stage="output", gather_at="loss", autograd=True, unpack_packed=True
    ),
    # Real logits are opt-in; fused CE explicitly gathers this hidden-state slot.
    DataKeySpec("logits", split_dim=1, side="output", stage="output", autograd=True, unpack_packed=True),
)


class InputKeyRegistry:
    """Collect provider declarations; reject conflicting definitions.

    Auxiliary keys are processed at their declared boundary (usually
    preprocessing). Custom model-forward keys remain global until the forward
    hook, so every key is sharded at most once.
    """

    _SPECIAL_KEYS = frozenset(PROCESSOR_PASSTHROUGH_KEYS)

    def __init__(self):
        self._specs = {spec.name: spec for spec in BUILTIN_INPUT_KEYS}

    def register(self, spec: DataKeySpec):
        if not isinstance(spec, DataKeySpec):
            raise TypeError("Expected a DataKeySpec")
        if spec.side != "input" or spec.stage not in ("auxiliary", "forward"):
            raise ValueError("Register only auxiliary or model-forward input keys here")
        if spec.name in _FORWARD_KEYS:
            raise ValueError(f"{spec.name!r} is owned by the model forward hook")
        if spec.name in self._SPECIAL_KEYS:
            raise ValueError(f"{spec.name!r} is a special processor/model argument, not an SP data key")
        if not spec.model_input and "forward" in (spec.pad_at, spec.split_at):
            raise ValueError(
                f"Loss-only input {spec.name!r} cannot use the forward boundary; "
                "use prepare, or set model_input=True for a model argument"
            )
        existing = self._specs.get(spec.name)
        if existing is not None and existing != spec:
            raise ValueError(f"Conflicting SP input rule for {spec.name!r}")
        self._specs[spec.name] = spec

    def __iter__(self):
        return iter(self._specs.values())

    def get(self, name):
        return self._specs.get(name)


class OutputKeyRegistry:
    def __init__(self):
        self._specs = {spec.name: spec for spec in BUILTIN_OUTPUT_KEYS}

    def register(self, spec):
        if not isinstance(spec, DataKeySpec):
            raise TypeError("Expected a DataKeySpec")
        if spec.side != "output":
            raise ValueError("Expected an output key rule")
        existing = self._specs.get(spec.name)
        if existing is not None and existing != spec:
            raise ValueError(f"Conflicting SP output rule for {spec.name!r}")
        self._specs[spec.name] = spec

    def __iter__(self):
        return iter(self._specs.values())

    def get(self, name):
        return self._specs.get(name)


def declared_key_specs(provider, side):
    """Read common field declarations, retaining legacy loss declaration names."""
    specs = getattr(provider, f"{side}_key_specs", ())
    if callable(specs):
        specs = specs()
    legacy = getattr(provider, f"sp_{side}_key_specs", ())
    for entries in (specs, legacy):
        if isinstance(entries, DataKeySpec):
            entries = (entries,)
        yield from entries or ()


def resolve_key_specs(*providers):
    """Build a call-local schema independently of any parallel strategy."""
    inputs, outputs = InputKeyRegistry(), OutputKeyRegistry()
    for provider in providers:
        for side, registry in (("input", inputs), ("output", outputs)):
            for spec in declared_key_specs(provider, side):
                registry.register(spec)
    return tuple(inputs), tuple(outputs)


def partition_model_inputs(inputs, input_key_specs):
    """Separate forward arguments from loss-only tensors without mutating the batch."""
    loss_names = {spec.name for spec in input_key_specs if not spec.model_input}
    loss_names.add("channel")  # sample-level loss metadata, not a sequence tensor
    model_inputs = {key: value for key, value in inputs.items() if key not in loss_names}
    loss_inputs = {key: value for key, value in inputs.items() if key in loss_names}
    return model_inputs, loss_inputs
