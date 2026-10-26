"""Shared sequence-field schemas consumed by processors, model wrappers and SP."""
from dataclasses import dataclass
from typing import Optional


_BOUNDARIES = frozenset({'prepare', 'forward', 'model_internal', 'router_replay'})


@dataclass(frozen=True)
class DataKeySpec:
    """Describe collation, model routing and optional parallel transforms.

    Auxiliary loss fields stay global by default. model_input controls routing;
    stage/pad_at/split_at control when an explicitly local tensor is transformed.
    """

    name: str
    split_dim: int = -1
    pad_value: int | float | str = 0
    need_roll: bool = False
    side: str = 'input'
    stage: str = 'auxiliary'
    need_split: Optional[bool] = None
    need_gather: bool = False
    autograd: bool = False
    owner: str = 'generic'
    need_pad: Optional[bool] = None
    pad_at: Optional[str] = 'auto'
    split_at: Optional[str] = 'auto'
    need_unpack: Optional[bool] = None
    model_input: Optional[bool] = None

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
        if self.need_split is None:
            local_stage = self.stage in ('forward', 'model_internal', 'router_replay')
            explicit_boundary = self.split_at not in (None, 'auto')
            object.__setattr__(self, 'need_split', self.side == 'input'
                               and (local_stage or explicit_boundary or self.need_gather or self.need_roll))
        if self.side == 'input' and self.need_gather and not self.need_split:
            raise ValueError('Only sharded input keys can request gathering')
        if self.stage == 'forward' and self.need_gather:
            raise ValueError('Model-forward keys cannot request loss gathering')
        if self.model_input is None:
            object.__setattr__(self, 'model_input', self.stage == 'forward')
        default_boundary = {
            'auxiliary': 'prepare',
            'forward': 'forward',
            'model_internal': 'model_internal',
            'router_replay': 'router_replay',
            'output': None,
        }[self.stage]
        if self.need_pad is None:
            object.__setattr__(self, 'need_pad', self.side == 'input'
                               and (self.need_split or self.pad_at not in (None, 'auto')))
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
    DataKeySpec('input_ids', split_dim=1, pad_value='tokenizer', stage='forward', need_split=True),
    DataKeySpec('inputs_embeds', split_dim=1, stage='forward', need_split=True),
    DataKeySpec('position_ids', pad_value=-1, stage='forward', need_split=True),
    DataKeySpec('attention_mask', split_dim=1, stage='forward', need_pad=True, need_split=False),
    DataKeySpec('labels', pad_value=-100, need_split=True, need_gather=True, need_unpack=True),
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


class InputKeyRegistry:
    """Collect provider declarations; reject conflicting definitions.

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
        if not spec.model_input and 'forward' in (spec.pad_at, spec.split_at):
            raise ValueError(f'Loss-only input {spec.name!r} cannot use the forward boundary; '
                             'use prepare, or set model_input=True for a model argument')
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


def declared_key_specs(provider, side):
    """Read common field declarations, retaining legacy loss declaration names."""
    specs = getattr(provider, f'{side}_key_specs', ())
    if callable(specs):
        specs = specs()
    legacy = getattr(provider, f'sp_{side}_key_specs', ())
    for entries in (specs, legacy):
        if isinstance(entries, DataKeySpec):
            entries = (entries,)
        yield from entries or ()


def resolve_key_specs(*providers):
    """Build a call-local schema independently of any parallel strategy."""
    inputs, outputs = InputKeyRegistry(), OutputKeyRegistry()
    for provider in providers:
        for side, registry in (('input', inputs), ('output', outputs)):
            for spec in declared_key_specs(provider, side):
                registry.register(spec)
    return tuple(inputs), tuple(outputs)


def partition_model_inputs(inputs, input_key_specs):
    """Separate forward arguments from loss-only tensors without mutating the batch."""
    loss_names = {spec.name for spec in input_key_specs if not spec.model_input}
    loss_names.add('channel')  # sample-level loss metadata, not a sequence tensor
    model_inputs = {key: value for key, value in inputs.items() if key not in loss_names}
    loss_inputs = {key: value for key, value in inputs.items() if key in loss_names}
    return model_inputs, loss_inputs
