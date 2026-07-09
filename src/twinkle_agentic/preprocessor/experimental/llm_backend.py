# Copyright (c) ModelScope Contributors. All rights reserved.
"""Abstract LLM backend for preprocessor pipeline.

Supports two modes:
  - OpenAIBackend: httpx-based calls to any OpenAI-compatible HTTP server
  - SamplerBackend: direct calls to Twinkle vLLMSampler Ray actor (no HTTP)
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from twinkle.utils import get_logger

logger = get_logger()


class LLMBackend(ABC):
    """Abstract base for LLM inference used by QualityPreprocessor stages."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 16,
        n: int = 1,
    ) -> List[Dict[str, str]]:
        """Chat completion.

        Returns:
            List of n choices, each a dict with keys 'content' and 'reasoning_content'.
        """

    def chat_batch(
        self,
        messages_list: List[List[Dict[str, Any]]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 16,
        n: int = 1,
    ) -> List[List[Dict[str, str]]]:
        """Batched chat completion. Returns one List[choice] per input messages list.

        Default impl loops over `chat`; backends should override to fan out concurrently
        (HTTP) or pass the full list to the underlying sampler in a single call (vLLM DP).
        """
        return [self.chat(m, temperature=temperature, max_tokens=max_tokens, n=n) for m in messages_list]

    @abstractmethod
    def prompt_logprobs(self, messages: List[Dict[str, Any]]) -> Optional[List]:
        """Evaluate prompt tokens without generation.

        Returns:
            List of per-token logprob entries (format varies by backend but
            is compatible with _extract_logprob helpers), or None on failure.
        """

    @abstractmethod
    def prompt_logprobs_ids(self, input_ids_list: List[List[int]]) -> List[List]:
        """Batched: evaluate raw token-id prompts without chat template wrapping.

        Used for unconditional perplexity (e.g. IFD denominator). Caller MUST
        supply a list of token-id sequences; for distributed backends the list
        length must satisfy backend-specific batching constraints (e.g.
        ``len >= dp_world_size`` for SamplerBackend).
        """

    def embeddings(self, texts: List[str]) -> Any:
        """Compute text embeddings. Override in backends that support it."""
        raise NotImplementedError(f'{type(self).__name__} does not support embeddings')


class OpenAIBackend(LLMBackend):
    """Backend wrapping any OpenAI-compatible HTTP endpoint."""

    def __init__(
        self,
        endpoint: str,
        model: str = 'default',
        api_key: str = '',
        timeout: float = 120.0,
    ):
        import httpx
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        self._client = httpx.Client(timeout=timeout, headers=headers)
        base = endpoint.rstrip('/')
        self._chat_endpoint = f'{base}/v1/chat/completions'
        self._embed_endpoint = f'{base}/v1/embeddings'
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 16,
        n: int = 1,
    ) -> List[Dict[str, str]]:
        try:
            resp = self._client.post(
                self._chat_endpoint,
                json={
                    'model': self._model,
                    'messages': messages,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'n': n,
                })
            resp.raise_for_status()
            choices = resp.json().get('choices', [])
            results = []
            for c in choices:
                msg = c.get('message') or {}
                results.append({
                    'content': msg.get('content') or '',
                    'reasoning_content': msg.get('reasoning_content') or '',
                })
            return results
        except Exception as e:
            logger.warning(f'[OpenAIBackend] chat failed: {e}')
            return []

    def chat_batch(
        self,
        messages_list: List[List[Dict[str, Any]]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 16,
        n: int = 1,
        max_workers: int = 16,
    ) -> List[List[Dict[str, str]]]:
        """Concurrent chat: vLLM HTTP server multiplexes requests; httpx.Client is thread-safe."""
        from concurrent.futures import ThreadPoolExecutor
        if not messages_list:
            return []
        workers = max(1, min(max_workers, len(messages_list)))
        results: List[List[Dict[str, str]]] = [[] for _ in messages_list]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(self.chat, m, temperature=temperature, max_tokens=max_tokens, n=n): i
                for i, m in enumerate(messages_list)
            }
            for fut in futs:
                results[futs[fut]] = fut.result()
        return results

    def prompt_logprobs(self, messages: List[Dict[str, Any]]) -> Optional[List]:
        try:
            resp = self._client.post(
                self._chat_endpoint,
                json={
                    'model': self._model,
                    'messages': messages,
                    'max_tokens': 0,
                    'prompt_logprobs': 1,
                })
            resp.raise_for_status()
            return resp.json().get('prompt_logprobs')
        except Exception:
            return None

    def prompt_logprobs_ids(self, input_ids_list: List[List[int]]) -> List[List]:
        endpoint = self._chat_endpoint.rsplit('/', 2)[0] + '/v1/completions'
        results: List[List] = []
        for input_ids in input_ids_list:
            resp = self._client.post(
                endpoint,
                json={
                    'model': self._model,
                    'prompt': list(input_ids),
                    'max_tokens': 0,
                    'echo': True,
                    'prompt_logprobs': 1,
                })
            resp.raise_for_status()
            data = resp.json()
            choices = data.get('choices') or []
            if choices and 'prompt_logprobs' in choices[0]:
                results.append(choices[0]['prompt_logprobs'])
            else:
                results.append(data['prompt_logprobs'])
        return results

    def embeddings(self, texts: List[str]):
        import numpy as np
        resp = self._client.post(
            self._embed_endpoint, json={
                'model': self._model,
                'input': texts,
            })
        resp.raise_for_status()
        data = resp.json().get('data', [])
        data_sorted = sorted(data, key=lambda x: x.get('index', 0))
        return np.array([d['embedding'] for d in data_sorted], dtype=np.float32)


class SamplerBackend(LLMBackend):
    """Backend wrapping a Twinkle vLLMSampler (Ray actor, no HTTP overhead)."""

    def __init__(
        self,
        sampler,
        embed_endpoint: str = '',
        embed_model: str = 'bge-m3',
    ):
        """
        Args:
            sampler: A vLLMSampler instance (with template already set).
            embed_endpoint: Optional OpenAI-compatible endpoint for embeddings.
            embed_model: Model name for embeddings.
        """
        self._sampler = sampler
        self._embed_endpoint = embed_endpoint
        self._embed_model = embed_model
        self._embed_client = None
        if embed_endpoint:
            import httpx
            self._embed_client = httpx.Client(timeout=120.0)
            self._embed_url = f'{embed_endpoint.rstrip("/")}/v1/embeddings'

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 16,
        n: int = 1,
    ) -> List[Dict[str, str]]:
        from twinkle.data_format import SamplingParams
        trajectory = {'messages': messages}
        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            num_samples=n,
        )
        try:
            responses = self._sampler.sample(trajectory, params)
            results = []
            for resp in responses:
                for seq in resp.sequences:
                    text = seq.decoded or ''
                    reasoning = ''
                    if '</think>' in text:
                        parts = text.split('</think>', 1)
                        reasoning = parts[0].split('<think>')[-1].strip()
                        text = parts[1].strip()
                    results.append({'content': text, 'reasoning_content': reasoning})
            return results
        except Exception as e:
            logger.warning(f'[SamplerBackend] chat failed: {e}')
            return []

    @staticmethod
    def _split_think(text: str) -> Tuple[str, str]:
        if '</think>' in text:
            parts = text.split('</think>', 1)
            return parts[1].strip(), parts[0].split('<think>')[-1].strip()
        return text, ''

    def chat_batch(
        self,
        messages_list: List[List[Dict[str, Any]]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 16,
        n: int = 1,
    ) -> List[List[Dict[str, str]]]:
        """One sampler dispatch over the full list; lets vLLM DP workers stay saturated."""
        from twinkle.data_format import SamplingParams
        if not messages_list:
            return []
        device_mesh = getattr(self._sampler, 'device_mesh', None)
        dp_world_size = getattr(device_mesh, 'dp_world_size', 1) or 1
        n_inputs = len(messages_list)
        feats = [{'messages': m} for m in messages_list]
        # Pad the dispatch so every DP worker has at least one item; trim duplicates after.
        if n_inputs < dp_world_size:
            feats = feats + [feats[-1]] * (dp_world_size - n_inputs)
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens, num_samples=n)
        try:
            responses = self._sampler.sample(feats, params)
        except Exception as e:
            logger.warning(f'[SamplerBackend] chat_batch failed: {e}')
            return [[] for _ in range(n_inputs)]
        responses = list(responses)[:n_inputs]
        out: List[List[Dict[str, str]]] = []
        for resp in responses:
            choices: List[Dict[str, str]] = []
            for seq in (getattr(resp, 'sequences', None) or []):
                text, reasoning = self._split_think(seq.decoded or '')
                choices.append({'content': text, 'reasoning_content': reasoning})
            out.append(choices)
        while len(out) < n_inputs:
            out.append([])
        return out

    def prompt_logprobs(self, messages: List[Dict[str, Any]]) -> Optional[List]:
        from twinkle.data_format import SamplingParams
        trajectory = {'messages': messages}
        params = SamplingParams(max_tokens=0, prompt_logprobs=1)
        try:
            responses = self._sampler.sample(trajectory, params)
            if responses and responses[0].prompt_logprobs is not None:
                return responses[0].prompt_logprobs
            return None
        except Exception as e:
            logger.warning(f'[SamplerBackend] prompt_logprobs failed: {e}')
            return None

    def prompt_logprobs_ids(self, input_ids_list: List[List[int]]) -> List[List]:
        from twinkle.data_format import SamplingParams
        if not isinstance(input_ids_list, list) or not input_ids_list:
            raise ValueError('prompt_logprobs_ids requires a non-empty List[List[int]].')
        device_mesh = getattr(self._sampler, 'device_mesh', None)
        dp_world_size = getattr(device_mesh, 'dp_world_size', 1) or 1
        if len(input_ids_list) < dp_world_size:
            raise ValueError(f'SamplerBackend.prompt_logprobs_ids requires at least '
                             f'dp_world_size={dp_world_size} inputs to keep all DP workers busy, '
                             f'got {len(input_ids_list)}. Batch upstream before calling.')
        feats = [{'input_ids': list(ids)} for ids in input_ids_list]
        params = SamplingParams(max_tokens=0, prompt_logprobs=1)
        responses = self._sampler.sample(feats, params)
        return [r.prompt_logprobs for r in responses]

    def embeddings(self, texts: List[str]):
        if self._embed_client is None:
            raise NotImplementedError('SamplerBackend requires embed_endpoint for embeddings. '
                                      'Pass embed_endpoint when constructing SamplerBackend.')
        import numpy as np
        resp = self._embed_client.post(
            self._embed_url, json={
                'model': self._embed_model,
                'input': texts,
            })
        resp.raise_for_status()
        data = resp.json().get('data', [])
        data_sorted = sorted(data, key=lambda x: x.get('index', 0))
        return np.array([d['embedding'] for d in data_sorted], dtype=np.float32)
