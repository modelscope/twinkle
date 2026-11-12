# Copyright (c) ModelScope Contributors. All rights reserved.
"""A served sampler and its weight sync, behind the interfaces the local rollout and driver use."""
from __future__ import annotations
import base64
import threading
from io import BytesIO
from typing import Any, Dict, List, Optional

from twinkle.data_format.sampling import SampledSequence, SampleResponse


class HTTPSampler:
    """A served sampler behind the ``sample`` call :class:`MultiTurnRollout` makes.

    The server keeps no episode state. A text prompt is sent as its encoded feature and the server
    returns the extended one. A prompt with media is sent as the ids the engine runs on plus the
    media itself, uploaded again on every turn; the server returns the sampled tokens only and the
    extended feature is built here with the episode's ``template``, the way an in-process sampler
    builds it.

    ``adapter_path`` is the policy samples come from (None: the server's base weights);
    :class:`ClientCheckpointManager` moves it to each newly saved adapter.
    """

    # Hand this sampler the episode's RolloutState: it carries the engine ids and the media.
    accepts_rollout_state = True

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
        media = getattr(item, 'media', None)
        if media:
            entry = {'input_ids': list(item.engine_input_ids), **_wire_media(media)}
        else:
            entry = dict(getattr(item, 'feature', item))
        responses = self.sampler.sample(
            [entry], sampling_params, adapter_name=self.adapter_name, adapter_uri=adapter_uri, token_only=bool(media))
        if len(responses) != 1:
            raise RuntimeError(f'the server returned {len(responses)} responses for one prompt.')
        response = responses[0]
        sequences = []
        for seq in response.sequences:
            tokens = list(seq.tokens)
            if media:
                completion = self.template.append_completion(item, tokens)
                sequences.append(SampledSequence(
                    stop_reason=seq.stop_reason, tokens=tokens, logprobs=seq.logprobs,
                    decoded=self.template.decode(tokens), new_input_feature=dict(completion.state.feature),
                    new_state=completion.state))
            else:
                sequences.append(SampledSequence(
                    stop_reason=seq.stop_reason, tokens=tokens, logprobs=seq.logprobs, decoded=seq.decoded,
                    new_input_feature=seq.new_input_feature))
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
    """The media of a RolloutState as the strings the wire carries (URL or ``data:`` URI)."""
    out = {}
    for key in ('images', 'videos', 'audios'):
        values = getattr(media, key, ()) or ()
        if values:
            out[key] = [_media_ref(value) for value in values]
    return out


def _media_ref(value: Any) -> str:
    """A URL passes as is; a local file, raw bytes or a PIL image is uploaded as a ``data:`` URI."""
    if isinstance(value, dict) and ('bytes' in value or 'path' in value):
        value = value.get('bytes') or value.get('path')
    if isinstance(value, str):
        if value.startswith(('http://', 'https://', 'data:')):
            return value
        with open(value, 'rb') as file:
            value = file.read()
    if isinstance(value, (bytes, bytearray)):
        return 'data:application/octet-stream;base64,' + base64.b64encode(bytes(value)).decode()
    if callable(getattr(value, 'save', None)):  # a PIL image: PNG keeps the pixels the template encoded
        buffer = BytesIO()
        value.save(buffer, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
    raise TypeError(f'cannot upload a {type(value).__name__} media entry; pass a path, URL, bytes or PIL image.')
