"""SP execution collections; common field declarations live in twinkle.data_fields."""
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional

# Preserve the existing public imports for model adapters and external callers.
from twinkle.data_fields import (BUILTIN_INPUT_KEYS, BUILTIN_OUTPUT_KEYS, PROCESSOR_PASSTHROUGH_KEYS,
                                 DataKeySpec, InputKeyRegistry, OutputKeyRegistry)


__all__ = ['DataKeySpec', 'InputKeyRegistry', 'OutputKeyRegistry', 'BUILTIN_INPUT_KEYS',
           'BUILTIN_OUTPUT_KEYS', 'PROCESSOR_PASSTHROUGH_KEYS', 'SPKeyValue', 'SPKeyCollection']


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
