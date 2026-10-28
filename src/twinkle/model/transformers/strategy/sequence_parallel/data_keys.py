"""SP-local execution defaults, registries and batch-local key collections."""
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional

from twinkle.data_format.sequence_fields import DataKeySpec
from twinkle.processor.fields import PROCESSOR_PASSTHROUGH_KEYS

_FORWARD_KEYS = ('input_ids', 'inputs_embeds', 'position_ids', 'attention_mask')

__all__ = ['DataKeySpec', 'InputKeyRegistry', 'OutputKeyRegistry', 'BUILTIN_INPUT_KEYS',
           'BUILTIN_OUTPUT_KEYS', 'PROCESSOR_PASSTHROUGH_KEYS', 'SPKeyValue', 'SPKeyCollection']


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


@dataclass
class SPKeyValue:
    """One tensor value paired with its immutable SP processing rule."""

    spec: DataKeySpec
    value: Any

    @property
    def name(self):
        return self.spec.name


@dataclass
class SPKeyCollection:
    """Batch-local common and registered SP tensors passed through one executor."""

    common: Dict[str, SPKeyValue] = field(default_factory=dict)
    extra: Dict[str, SPKeyValue] = field(default_factory=dict)

    def add(self, item: SPKeyValue, *, common: bool = False):
        if not isinstance(item, SPKeyValue):
            raise TypeError('SPKeyCollection accepts SPKeyValue items')
        if item.name in self.common or item.name in self.extra:
            raise ValueError(f'Duplicate SP key in collection: {item.name!r}')
        target = self.common if common else self.extra
        target[item.name] = item

    def values(self) -> Iterator[SPKeyValue]:
        yield from self.common.values()
        yield from self.extra.values()

    def get(self, name: str) -> Optional[SPKeyValue]:
        return self.common.get(name, self.extra.get(name))

    def get_value(self, name: str, default=None):
        """Return a key's payload without exposing its SP rule wrapper."""
        item = self.get(name)
        return default if item is None else item.value

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any], specs: Iterable[DataKeySpec], common_names=()):
        collection = cls()
        common_names = set(common_names)
        for spec in specs:
            if spec.name in values:
                collection.add(SPKeyValue(spec, values[spec.name]), common=spec.name in common_names)
        return collection
