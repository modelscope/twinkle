"""Sequence-field declarations shared by data processors and model/loss providers."""

from dataclasses import dataclass

_BOUNDARIES = ("forward", "auxiliary", "model_internal", "router_replay", "output")
_PROCESSING_BOUNDARIES = ("prepare", "forward", "model_internal", "router_replay")


@dataclass(frozen=True)
class DataKeySpec:
    """Describe collation, routing and explicitly scheduled parallel operations.

    Loss-only fields stay global unless pad_at/split_at opt into local processing.
    Packed restoration is independent of sequence-parallel gather and split.
    """

    name: str
    split_dim: int = -1
    pad_value: int | float | str = 0
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


def partition_model_inputs(inputs, input_key_specs):
    """Separate forward arguments from loss-only tensors without mutating the batch."""
    loss_names = {spec.name for spec in input_key_specs if not spec.model_input}
    loss_names.add("channel")  # sample-level loss metadata, not a sequence tensor
    model_inputs = {key: value for key, value in inputs.items() if key not in loss_names}
    loss_inputs = {key: value for key, value in inputs.items() if key in loss_names}
    return model_inputs, loss_inputs
