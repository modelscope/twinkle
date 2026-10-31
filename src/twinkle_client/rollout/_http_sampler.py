# Copyright (c) ModelScope Contributors. All rights reserved.
"""Adapt the existing HTTP sampler to the core multi-turn callback contract."""
from __future__ import annotations

from typing import Any

from twinkle.data_format.sampling import SampledSequence


class _HTTPSamplerAdapter:
    """Capture one call's policy and translate one wire response per turn."""

    def __init__(self, sampler, *, adapter_name: str = '', adapter_uri: str | None = None):
        self._sampler = sampler
        self._adapter_name = adapter_name
        self._adapter_uri = adapter_uri

    def __call__(
        self,
        sampler,
        api,
        sampling_params,
        *,
        input_feature,
        adapter_kwargs,
        rollout_state=None,
        trajectory_index=0,
        **kwargs,
    ) -> SampledSequence:
        del sampler, api, kwargs
        self._validate_state(rollout_state, input_feature, trajectory_index)
        adapter_uri = adapter_kwargs.get('adapter_path', self._adapter_uri)
        responses = self._sampler.sample(
            [input_feature],
            sampling_params=sampling_params,
            adapter_name=self._adapter_name,
            adapter_uri=adapter_uri,
            num_samples=1,
        )
        if not isinstance(responses, list) or len(responses) != 1:
            count = len(responses) if isinstance(responses, list) else type(responses).__name__
            raise RuntimeError(f'sampler returned {count} responses for trajectory {trajectory_index}; expected one')
        response = responses[0]
        if len(response.sequences) != 1:
            raise RuntimeError(
                f'sampler returned {len(response.sequences)} sequences for trajectory {trajectory_index}; expected one'
            )
        sequence = response.sequences[0]
        return SampledSequence(
            stop_reason=sequence.stop_reason,
            tokens=list(sequence.tokens),
            logprobs=sequence.logprobs,
            decoded=sequence.decoded,
            new_input_feature=sequence.new_input_feature,
        )

    @staticmethod
    def _validate_state(state, feature: dict[str, Any], trajectory_index: int) -> None:
        feature_ids = list(feature.get('input_ids') or [])
        if state is None or state.media:
            raise NotImplementedError(
                f'trajectory {trajectory_index} requires server-side RolloutState state_id for HTTP multi-turn media'
            )
        if feature_ids != list(state.engine_input_ids):
            raise NotImplementedError(
                f'trajectory {trajectory_index} requires server-side RolloutState state_id because feature and engine '
                'ids differ'
            )
