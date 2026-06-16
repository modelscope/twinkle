# Copyright (c) ModelScope Contributors. All rights reserved.
import httpx
import math
from copy import copy
from typing import Any, Dict, List, Literal, Optional, Union

from twinkle import get_logger
from twinkle.data_format import SampledSequence, SampleResponse, SamplingParams, Trajectory

logger = get_logger()


def _entropy_from_topk(logprobs_per_token: List[List[tuple]]) -> float:
    """Mean per-token entropy approximated from top-K logprobs (renormalized)."""
    if not logprobs_per_token:
        return float('inf')
    total = 0.0
    for candidates in logprobs_per_token:
        if not candidates:
            total += float('inf')
            continue
        lps = [lp for _, lp in candidates]
        max_lp = max(lps)
        # numerically stable softmax over top-K
        exps = [math.exp(lp - max_lp) for lp in lps]
        z = sum(exps)
        total += sum(-(e / z) * (lp - max_lp - math.log(z)) for e, lp in zip(exps, lps))
    return total / len(logprobs_per_token)


def _mean_logp(logprobs_per_token: List[List[tuple]], tokens: List[int]) -> float:
    """Mean log-probability of generated tokens (sequence-level confidence)."""
    if not logprobs_per_token or not tokens:
        return float('-inf')
    total = 0.0
    count = 0
    for t, candidates in enumerate(logprobs_per_token):
        if t >= len(tokens) or not candidates:
            continue
        tok = tokens[t]
        lp = next((v for tid, v in candidates if tid == tok), None)
        if lp is None:
            lp = candidates[0][1]
        total += lp
        count += 1
    return total / max(count, 1)


class RouterSampler:
    """Confidence-based routing sampler.

    Generates with a local sampler first; if confidence is low, falls back
    to an OpenAI-compatible endpoint (stronger model).
    """

    def __init__(
        self,
        sampler,
        fallback_endpoint: str,
        fallback_model: str = 'default',
        fallback_api_key: str = '',
        method: Literal['entropy', 'logp'] = 'entropy',
        threshold: float = 2.0,
        top_k_logprobs: int = 10,
        fallback_temperature: float = 0.7,
        fallback_max_tokens: int = 4096,
        timeout: float = 120.0,
    ):
        """
        Args:
            sampler: Inner sampler instance (e.g. vLLMSampler).
            fallback_endpoint: OpenAI-compatible API base URL.
            fallback_model: Model name for fallback requests.
            fallback_api_key: Bearer token for fallback API.
            method: Confidence metric — 'entropy' (route when H > threshold)
                    or 'logp' (route when mean logp < threshold).
            threshold: Routing threshold. For entropy: higher = more routing.
                       For logp: lower (more negative) = more routing.
            top_k_logprobs: Number of top logprobs to request from inner sampler.
            fallback_temperature: Temperature for fallback generation.
            fallback_max_tokens: Max tokens for fallback generation.
            timeout: HTTP timeout for fallback requests.
        """
        self.sampler = sampler
        self._method = method
        self._threshold = threshold
        self._top_k = top_k_logprobs
        self._fb_temperature = fallback_temperature
        self._fb_max_tokens = fallback_max_tokens
        self._fb_endpoint = f'{fallback_endpoint.rstrip("/")}/v1/chat/completions'
        self._fb_model = fallback_model
        headers = {'Content-Type': 'application/json'}
        if fallback_api_key:
            headers['Authorization'] = f'Bearer {fallback_api_key}'
        self._client = httpx.Client(timeout=timeout, headers=headers)

    @property
    def template(self):
        return self.sampler.template

    def set_template(self, *args, **kwargs):
        return self.sampler.set_template(*args, **kwargs)

    def _should_route(self, seq: SampledSequence) -> bool:
        if not seq.logprobs:
            return True
        if self._method == 'entropy':
            score = _entropy_from_topk(seq.logprobs)
            return score > self._threshold
        score = _mean_logp(seq.logprobs, seq.tokens)
        return score < self._threshold

    def _fallback_generate(self, trajectory: Trajectory) -> Optional[str]:
        messages = trajectory.get('messages', [])
        if not messages:
            return None
        api_messages = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            entry = {'role': m.get('role', 'user')}
            content = m.get('content', '')
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        parts.append(block.get('text', ''))
                content = '\n'.join(parts) if parts else ''
            entry['content'] = content or ''
            api_messages.append(entry)
        try:
            resp = self._client.post(
                self._fb_endpoint,
                json={
                    'model': self._fb_model,
                    'messages': api_messages,
                    'temperature': self._fb_temperature,
                    'max_tokens': self._fb_max_tokens,
                })
            resp.raise_for_status()
            choices = resp.json().get('choices', [])
            if choices:
                return (choices[0].get('message') or {}).get('content', '')
        except Exception as e:
            logger.warning(f'RouterSampler fallback failed: {e}')
        return None

    def sample(
        self,
        inputs: Union[Dict, List[Dict]],
        sampling_params: Optional[Union[SamplingParams, Dict[str, Any]]] = None,
        adapter_name: str = '',
        adapter_path: Optional[str] = None,
        **kwargs,
    ) -> List[SampleResponse]:
        """Sample with confidence-based routing to fallback model."""
        if sampling_params is None:
            sampling_params = SamplingParams()
        elif isinstance(sampling_params, dict):
            sampling_params = SamplingParams.from_dict(sampling_params)

        # Ensure logprobs are requested for confidence evaluation
        routed_params = copy(sampling_params)
        if routed_params.logprobs is None or routed_params.logprobs < self._top_k:
            routed_params.logprobs = self._top_k

        inputs_list = inputs if isinstance(inputs, list) else [inputs]
        is_trajectory = isinstance(inputs_list[0], dict) and 'input_ids' not in inputs_list[0]

        results = self.sampler.sample(inputs_list, routed_params, adapter_name, adapter_path=adapter_path, **kwargs)

        if not is_trajectory:
            return results

        for i, (resp, traj) in enumerate(zip(results, inputs_list)):
            new_sequences = []
            for seq in resp.sequences:
                if self._should_route(seq):
                    fallback_text = self._fallback_generate(traj)
                    if fallback_text is not None:
                        new_sequences.append(
                            SampledSequence(
                                stop_reason='stop',
                                tokens=[],
                                logprobs=None,
                                decoded=fallback_text,
                            ))
                        continue
                new_sequences.append(seq)
            results[i] = SampleResponse(
                sequences=new_sequences,
                prompt_token_ids=resp.prompt_token_ids,
                prompt_logprobs=resp.prompt_logprobs,
                topk_prompt_logprobs=resp.topk_prompt_logprobs,
            )

        return results
