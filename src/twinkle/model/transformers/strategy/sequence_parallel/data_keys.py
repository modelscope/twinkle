"""Per-model contracts for sequence-aligned inputs and outputs.

Each key declares whether and where it is padded, split, gathered, or restored
after packed-sequence processing. Forward and auxiliary keys share the same
executor but run at different lifecycle boundaries. Loss-scale weights are
already aligned with Twinkle's shifted labels, so the production rule does not
roll them a second time.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional


_BOUNDARIES = frozenset({'prepare', 'forward', 'model_internal', 'router_replay'})


@dataclass(frozen=True)
class DataKeySpec:
    name: str
    split_dim: int = -1
    pad_value: int | float | str = 0
    need_roll: bool = False
    side: str = 'input'
    stage: str = 'auxiliary'
    need_split: bool = True
    need_gather: bool = False
    autograd: bool = False
    owner: str = 'generic'
    need_pad: Optional[bool] = None
    pad_at: Optional[str] = 'auto'
    split_at: Optional[str] = 'auto'
    need_unpack: Optional[bool] = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('An SP input key must have a non-empty name')
        if not isinstance(self.split_dim, int):
            raise TypeError('split_dim must be an integer')
        if self.side not in ('input', 'output'):
            raise ValueError('side must be input or output')
        if self.stage not in ('forward', 'auxiliary', 'model_internal', 'router_replay', 'output'):
            raise ValueError('Invalid SP key stage')
        if (self.side == 'output') != (self.stage == 'output'):
            raise ValueError('Output keys must use stage=output')
        if self.owner not in ('generic', 'task', 'model', 'loss'):
            raise ValueError('Invalid SP key owner')
        if isinstance(self.pad_value, str) and (self.name != 'input_ids' or self.pad_value != 'tokenizer'):
            raise ValueError('Only input_ids can use the tokenizer padding sentinel')
        if self.side == 'input' and self.need_gather and not self.need_split:
            raise ValueError('Only sharded input keys can request gathering')
        if self.stage == 'forward' and self.need_gather:
            raise ValueError('Model-forward keys cannot request loss gathering')
        default_boundary = {
            'auxiliary': 'prepare',
            'forward': 'forward',
            'model_internal': 'model_internal',
            'router_replay': 'router_replay',
            'output': None,
        }[self.stage]
        if self.need_pad is None:
            object.__setattr__(self, 'need_pad', self.need_split and self.side == 'input')
        if self.pad_at == 'auto':
            object.__setattr__(self, 'pad_at', default_boundary if self.need_pad else None)
        elif self.pad_at is not None and self.pad_at not in _BOUNDARIES:
            raise ValueError(f'Invalid SP padding boundary: {self.pad_at!r}')
        elif (self.pad_at is None) == bool(self.need_pad):
            raise ValueError('need_pad and pad_at specify conflicting SP padding behavior')
        if self.pad_at != 'auto':
            object.__setattr__(self, 'need_pad', self.pad_at is not None)
        if self.split_at == 'auto':
            object.__setattr__(self, 'split_at', default_boundary if self.need_split else None)
        elif self.split_at is not None and self.split_at not in _BOUNDARIES:
            raise ValueError(f'Invalid SP split boundary: {self.split_at!r}')
        elif (self.split_at is None) == bool(self.need_split):
            raise ValueError('need_split and split_at specify conflicting SP split behavior')
        if self.split_at != 'auto':
            object.__setattr__(self, 'need_split', self.split_at is not None)
        if self.side == 'output' and (self.pad_at is not None or self.split_at is not None):
            raise ValueError('Output keys cannot request input padding or splitting')
        if self.need_unpack is None:
            # Preserve the historical packed restoration behavior unless a
            # built-in rule explicitly opts out.
            object.__setattr__(self, 'need_unpack', self.side == 'output' or self.need_gather or not self.need_split)

    def sequence_dim(self, ndim):
        if not -ndim <= self.split_dim < ndim:
            raise ValueError(f'Invalid sequence axis {self.split_dim} for SP key {self.name!r}')
        dim = self.split_dim % ndim
        if dim == 0:
            raise ValueError(f'SP key {self.name!r} cannot split or gather the batch axis')
        return dim


BUILTIN_INPUT_KEYS = (
    DataKeySpec('input_ids', split_dim=1, pad_value='tokenizer', stage='forward'),
    DataKeySpec('inputs_embeds', split_dim=1, stage='forward'),
    DataKeySpec('position_ids', pad_value=-1, stage='forward'),
    DataKeySpec('attention_mask', split_dim=1, stage='forward', need_pad=True, need_split=False),
    DataKeySpec('labels', pad_value=-100, need_gather=True, need_unpack=True),
    # These are not needed by the model forward. Keep them global through SP;
    # packed sequence restoration is a separate action from sharding/gathering.
    DataKeySpec('completion_mask', need_pad=False, need_split=False, need_unpack=True),
    DataKeySpec('loss_scale', need_pad=False, need_split=False, need_unpack=True),
)

# These are processor/model control arguments, not sequence data contracts.
# Keep them out of DataKeySpec and preserve them through the Transformers input
# conversion path explicitly instead.
PROCESSOR_PASSTHROUGH_KEYS = ('cache_position', 'padding_free')

BUILTIN_OUTPUT_KEYS = (
    DataKeySpec('logps', split_dim=1, side='output', stage='output',
                need_split=False, need_gather=True, autograd=True, need_unpack=True),
    DataKeySpec('entropies', split_dim=1, side='output', stage='output',
                need_split=False, need_gather=True, autograd=True, need_unpack=True),
    # Real logits are opt-in; fused CE explicitly gathers this hidden-state slot.
    DataKeySpec('logits', split_dim=1, side='output', stage='output',
                need_split=False, autograd=True, need_unpack=True),
)


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


class InputKeyRegistry:
    """Own declarations locally; reject accidental replacement of existing rules.

    Auxiliary keys are processed at their declared boundary (usually
    preprocessing). Custom model-forward keys remain global until the forward
    hook, so every key is sharded at most once.
    """

    _FORWARD_KEYS = frozenset({'input_ids', 'inputs_embeds', 'position_ids', 'attention_mask'})
    _SPECIAL_KEYS = frozenset(PROCESSOR_PASSTHROUGH_KEYS)

    def __init__(self):
        self._specs = {spec.name: spec for spec in BUILTIN_INPUT_KEYS}

    def register(self, spec: DataKeySpec):
        if not isinstance(spec, DataKeySpec):
            raise TypeError('Expected a DataKeySpec')
        if spec.side != 'input' or spec.stage not in ('auxiliary', 'forward'):
            raise ValueError('Register only auxiliary or model-forward input keys here')
        if spec.name in self._FORWARD_KEYS:
            raise ValueError(f'{spec.name!r} is owned by the model forward hook')
        if spec.name in self._SPECIAL_KEYS:
            raise ValueError(f'{spec.name!r} is a special processor/model argument, not an SP data key')
        existing = self._specs.get(spec.name)
        if existing is not None and existing != spec:
            raise ValueError(f'Conflicting SP input rule for {spec.name!r}')
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
            raise TypeError('Expected a DataKeySpec')
        if spec.side != 'output':
            raise ValueError('Expected an output key rule')
        existing = self._specs.get(spec.name)
        if existing is not None and existing != spec:
            raise ValueError(f'Conflicting SP output rule for {spec.name!r}')
        self._specs[spec.name] = spec

    def __iter__(self):
        return iter(self._specs.values())

    def get(self, name):
        return self._specs.get(name)
