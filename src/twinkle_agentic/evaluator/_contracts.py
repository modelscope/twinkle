"""Small public-facing contracts shared by evaluator internals."""

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from twinkle.data_format import SamplingParams, Trajectory


class EvaluatorConfigError(ValueError):
    """The evaluator constructor or its owned configuration is invalid."""


class UnsupportedCapabilityError(ValueError):
    """The selected backend cannot represent an explicit request exactly."""


class BackendContractError(RuntimeError):
    """A caller-owned API or sampler returned an invalid value."""


class SamplerBatchError(RuntimeError):
    """A single physical sampler batch failed."""


@runtime_checkable
class SamplerLike(Protocol):
    def sample(
        self,
        inputs: list[Trajectory],
        sampling_params: SamplingParams | Mapping[str, Any],
        **kwargs: Any,
    ) -> Sequence[Any]: ...


def read_value(value: Any, name: str, default: Any = None) -> Any:
    """Read an attribute or mapping key without treating falsy values as absent."""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)
