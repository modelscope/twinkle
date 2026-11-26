# Copyright (c) ModelScope Contributors. All rights reserved.
"""A served sampler and its weight sync, behind the interfaces the local rollout and driver use."""
from __future__ import annotations
import threading
from typing import Any, Dict, List, Optional

from twinkle.data_format.sampling import SampledSequence, SampleResponse, SamplingMask
from twinkle.data_format import copy_trajectory
from twinkle.protocol.json_utils import media_ref as _media_ref
from twinkle.template.contract import MediaBundle, engine_input_ids, feature_has_media


class HTTPSampler:
    """A served sampler behind the ``sample`` call :class:`MultiTurnRollout` makes.

    The server keeps no episode state. Text and multimodal prompts send the engine's token IDs,
    with media uploaded alongside. The client appends the returned tokens using the episode's
    template and preserves sampling metadata without retokenizing the reply.

    ``adapter_path`` is the policy samples come from (None: the server's base weights);
    :class:`ClientCheckpointManager` moves it to each newly saved adapter.
    """

    def __init__(self, sampler, template, *, adapter_name: str = ''):
        self.sampler = sampler
        self.template = template
        self.adapter_name = adapter_name
        self.adapter_path: Optional[str] = None

    def sample(self,
               inputs: Any,
               sampling_params: Any = None,
               *,
               adapter_path: Optional[str] = None,
               use_base_model: bool = False,
               session_id: Optional[str] = None) -> List[SampleResponse]:
        """One ``SampleResponse`` per input; ``session_id`` is accepted and unused (one server)."""
        del session_id
        if adapter_path and use_base_model:
            raise ValueError('adapter_path and use_base_model=True ask for opposite weights.')
        adapter_uri = None if use_base_model else (adapter_path or self.adapter_path)
        items = inputs if isinstance(inputs, list) else [inputs]
        return [self._sample_one(item, sampling_params, adapter_uri) for item in items]

    def _sample_one(self, item: Any, sampling_params: Any, adapter_uri: Optional[str]) -> SampleResponse:
        prompt_feature = copy_trajectory(item)
        scoring = (sampling_params.get('max_tokens') if isinstance(sampling_params, dict)
                   else getattr(sampling_params, 'max_tokens', None)) == 0
        if 'input_ids' not in prompt_feature:
            prompt_feature = self.template.encode_prompt(
                prompt_feature, backend='vllm', add_generation_prompt=not scoring)
        media = MediaBundle.from_sources(prompt_feature)
        if feature_has_media(prompt_feature) and not media:
            raise ValueError('Multimodal token-only sampling requires original media.')
        prefix_ids = list(prompt_feature['input_ids'])
        engine_ids = engine_input_ids(prompt_feature)
        if not media and engine_ids != prefix_ids:
            raise ValueError('A text trajectory must have identical training and engine input_ids.')
        entry = {'input_ids': engine_ids, **_wire_media(media)}
        if prompt_feature.get('mm_processor_kwargs'):
            entry['mm_processor_kwargs'] = prompt_feature['mm_processor_kwargs']
        responses = self.sampler.sample(
            [entry], sampling_params, adapter_name=self.adapter_name, adapter_uri=adapter_uri, token_only=True)
        if len(responses) != 1:
            raise RuntimeError(f'the server returned {len(responses)} responses for one prompt.')
        response = responses[0]
        # Engines may expand visual placeholders, so collapsed media IDs cannot be compared directly.
        if not media and response.prompt_token_ids != engine_ids:
            raise RuntimeError('Token-only sampling returned different or missing prompt_token_ids; '
                               'the training prefix must match the sampled text prompt exactly.')
        sequences = []
        for seq in response.sequences:
            tokens = list(seq.tokens)
            if seq.logprobs is not None and len(seq.logprobs) != len(tokens):
                raise RuntimeError('Token-only sampling must return one logprob entry per sampled token.')
            feature = (copy_trajectory(prompt_feature) if scoring or (seq.stop_reason == 'error' and not tokens)
                       else self.template.append_completion(prompt_feature, tokens))
            if list(feature['input_ids']) != prefix_ids + tokens:
                raise RuntimeError('Appending sampled tokens changed or truncated input_ids; '
                                   'increase max_length or use a template that preserves the sampled sequence.')
            # Routing belongs to this generation; a missing export must not reuse the prefix's old routing.
            feature.pop('routed_experts', None)
            if seq.routed_experts is not None:
                feature['routed_experts'] = seq.routed_experts
            if seq.sampling_masks is not None and len(seq.sampling_masks.offsets) != len(tokens) + 1:
                raise RuntimeError('Token-only sampling masks must contain one CSR segment per sampled token.')
            sampling_masks = (SamplingMask(token_ids=list(seq.sampling_masks.token_ids),
                                           offsets=list(seq.sampling_masks.offsets))
                              if seq.sampling_masks is not None else None)
            sequences.append(SampledSequence(
                stop_reason=seq.stop_reason, tokens=tokens, logprobs=seq.logprobs,
                decoded=self.template.decode(tokens), new_input_feature=feature,
                routed_experts=seq.routed_experts, sampling_masks=sampling_masks))
        return SampleResponse(sequences=sequences, prompt_token_ids=response.prompt_token_ids,
                              prompt_logprobs=response.prompt_logprobs,
                              topk_prompt_logprobs=response.topk_prompt_logprobs)


class ClientCheckpointManager:
    """The client's weight sync, with the ``sync_weights()`` / ``version()`` of CheckpointEngineManager.

    ``sync_weights`` saves the trained adapter on the server and points ``sampler`` at it, so every
    generation after it samples the new policy; ``version`` counts the syncs. Everything lives in
    this process, so the rollout and the driver never see an adapter path. Call ``sync_weights``
    once before training to publish the starting adapter.
    """

    def __init__(self, model, sampler: HTTPSampler, *, name_prefix: str = 'policy'):
        self.model = model
        self.sampler = sampler
        self.name_prefix = name_prefix
        self._version = 0
        self._lock = threading.Lock()

    def sync_weights(self) -> None:
        with self._lock:
            saved = self.model.save(f'{self.name_prefix}-{self._version}', save_optimizer=False)
            self.sampler.adapter_path = saved.twinkle_path
            self._version += 1

    def version(self) -> int:
        return self._version


def _wire_media(media: Any) -> Dict[str, List[str]]:
    """Original media references as URL or ``data:`` URI wire strings."""
    out = {}
    for key in ('images', 'videos', 'audios'):
        values = getattr(media, key, ()) or ()
        if values:
            out[key] = [_media_ref(value) for value in values]
    return out



