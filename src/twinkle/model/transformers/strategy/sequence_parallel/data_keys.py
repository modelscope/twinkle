"""SP-local execution defaults, registries and batch-local key collections."""
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional

from twinkle.data_format.sequence_fields import DataKeySpec
from twinkle.processor.fields import (
    COMMON_INPUT_SPECS, COMMON_OUTPUT_SPECS, PROCESSOR_PASSTHROUGH_KEYS, validate_input_key_spec)

__all__ = ['DataKeySpec', 'InputKeyRegistry', 'OutputKeyRegistry', 'BUILTIN_INPUT_KEYS',
           'BUILTIN_OUTPUT_KEYS', 'PROCESSOR_PASSTHROUGH_KEYS', 'SPKeyValue', 'SPKeyCollection']


# Layouts have one source in processor.fields. SP adds only execution rules;
# loss-only completion_mask/loss_scale retain their global layouts unchanged.
_INPUT_OPERATIONS = {
    "input_ids": dict(pad_at="forward", split_at="forward"),
    "inputs_embeds": dict(pad_at="forward", split_at="forward"),
    "position_ids": dict(pad_at="forward", split_at="forward"),
    "attention_mask": dict(pad_at="forward"),
    "labels": dict(pad_at="prepare", split_at="prepare", gather_at="loss"),
}
BUILTIN_INPUT_KEYS = tuple(replace(spec, **_INPUT_OPERATIONS.get(spec.name, {})) for spec in COMMON_INPUT_SPECS)

# Real logits are opt-in; fused CE explicitly gathers this hidden-state slot.
BUILTIN_OUTPUT_KEYS = tuple(
    replace(spec, autograd=True, gather_at="loss" if spec.name in ("logps", "entropies") else None)
    for spec in COMMON_OUTPUT_SPECS
)


class InputKeyRegistry:
    """Collect provider declarations; reject conflicting definitions.

    Auxiliary keys are processed at their declared boundary (usually
    preprocessing). Custom model-forward keys remain global until the forward
    hook, so every key is sharded at most once.
    """

    def __init__(self):
        self._specs = {spec.name: spec for spec in BUILTIN_INPUT_KEYS}

    def register(self, spec: DataKeySpec):
        validate_input_key_spec(spec)
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


class SPKeyCollection:
    """Batch-local common and registered SP tensors passed through one executor."""

    def __init__(self):
        self._items: Dict[str, SPKeyValue] = {}

    def add(self, item: SPKeyValue):
        if not isinstance(item, SPKeyValue):
            raise TypeError('SPKeyCollection accepts SPKeyValue items')
        if item.name in self._items:
            raise ValueError(f'Duplicate SP key in collection: {item.name!r}')
        self._items[item.name] = item

    def values(self) -> Iterator[SPKeyValue]:
        return iter(self._items.values())

    def get(self, name: str) -> Optional[SPKeyValue]:
        return self._items.get(name)

    def get_value(self, name: str, default=None):
        """Return a key's payload without exposing its SP rule wrapper."""
        item = self.get(name)
        return default if item is None else item.value

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any], specs: Iterable[DataKeySpec]):
        collection = cls()
        for spec in specs:
            if spec.name in values:
                collection.add(SPKeyValue(spec, values[spec.name]))
        return collection
