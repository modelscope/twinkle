# Copyright (c) ModelScope Contributors. All rights reserved.
"""Numpy-only mock model backend (R1, R3, R4).

Provides ``TwinkleCompatMockModel``, a stand-in for the real
``TwinkleCompatTransformersModel`` whose only purpose is to exercise the
server's HTTP and dispatch paths on a CPU-only host with no torch /
transformers / vllm / megatron installed. Determinism is keyed by
``(model_id, adapter_name, seed, input_shape)`` so repeated requests with the
same payload produce identical numpy-derived results.

This module deliberately avoids importing ``torch``, ``transformers``,
``vllm``, ``megatron`` or any module whose own imports would pull them
in transitively (e.g. ``twinkle.server.model.backends.common``). The class is
duck-typed against ``TwinkleCompatModelBase`` rather than subclassing it —
the base class lives in a torch-importing module and would defeat the
import-isolation requirement (R1.2).
"""
from __future__ import annotations

import hashlib
import numpy as np
from typing import Any


def _seed_for(model_id: str, adapter_name: str | None, seed: int, *extra: Any) -> int:
    """Deterministic per-request RNG seed derived from string/int components.

    Uses SHA-256 over a canonical string form rather than Python's built-in
    ``hash()``: the latter is salted per process (PYTHONHASHSEED) for tuples
    containing strings, which would make identical requests on different
    replicas / restarts produce different outputs and break R2.5 / R4.4 / R4.5
    across processes.
    """
    parts = (str(model_id), str(adapter_name), str(int(seed)), *(repr(x) for x in extra))
    digest = hashlib.sha256('\x1f'.join(parts).encode('utf-8')).digest()
    # numpy seeds must fit in uint32; take the first 4 bytes of the digest.
    return int.from_bytes(digest[:4], 'big')


class TwinkleCompatMockModel:
    """Numpy-only mock model.

    Public API mirrors the methods that the model FastAPI handlers call on
    ``self.model``. Every method either returns a deterministic numpy-derived
    payload or completes as a no-op without raising.
    """

    def __init__(
        self,
        model_id: str,
        *,
        hidden_size: int = 8,
        vocab_size: int = 32,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        self.model_id = model_id
        self._hidden_size = int(hidden_size)
        self._vocab_size = int(vocab_size)
        self._rng_seed = int(seed)
        # adapter_name -> arbitrary config payload
        self._adapters: dict[str, dict[str, Any]] = {}

    # ----- Forward family (R1.3) ----------------------------------------- #

    def _build_forward_result(
        self,
        inputs: Any,
        adapter_name: str | None,
        *,
        loss_value: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Return one deterministic synthetic per-input record.

        Shapes are derived from the input so ``_tinker_build_output``-style
        callers see correctly-sized arrays.
        """
        seq_lens = _input_seq_lengths(inputs)
        out: list[dict[str, Any]] = []
        for idx, seq_len in enumerate(seq_lens):
            rng = np.random.default_rng(_seed_for(self.model_id, adapter_name, self._rng_seed, idx, seq_len))
            logprobs = rng.uniform(-2.0, 0.0, size=seq_len).astype(np.float32)
            elementwise_loss = rng.uniform(0.0, 1.0, size=seq_len).astype(np.float32)
            out.append({
                'logprobs': logprobs.tolist(),
                'elementwise_loss': elementwise_loss.tolist(),
                'loss': float(loss_value),
            })
        return out

    def tinker_forward_only(self, *, inputs: Any, adapter_name: str | None = None, **kwargs: Any) -> list[Any]:
        return [self._build_forward_result(inputs, adapter_name), 0.0]

    def tinker_forward_backward(self, *, inputs: Any, adapter_name: str, loss_fn: str, **kwargs: Any) -> list[Any]:
        loss_seed = _seed_for(self.model_id, adapter_name, self._rng_seed, 'loss', loss_fn)
        loss = float(np.random.default_rng(loss_seed).uniform(0.0, 1.0))
        return [self._build_forward_result(inputs, adapter_name, loss_value=loss), loss]

    def forward(self, *, inputs: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return self._build_forward_result(inputs, kwargs.get('adapter_name'))

    def forward_only(self, *, inputs: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return self._build_forward_result(inputs, kwargs.get('adapter_name'))

    def forward_backward(self, *, inputs: Any, **kwargs: Any) -> list[Any]:
        loss = float(np.random.default_rng(self._rng_seed).uniform(0.0, 1.0))
        return [self._build_forward_result(inputs, kwargs.get('adapter_name'), loss_value=loss), loss]

    def calculate_loss(self, *, inputs: Any, **kwargs: Any) -> float:
        return float(np.random.default_rng(self._rng_seed).uniform(0.0, 1.0))

    # ----- Backward / optimizer (R1.4) ----------------------------------- #

    def backward(self, *args: Any, **kwargs: Any) -> None:
        return None

    def step(self, *args: Any, **kwargs: Any) -> None:
        return None

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        return None

    def lr_step(self, *args: Any, **kwargs: Any) -> None:
        return None

    def clip_grad_norm(self, *args: Any, **kwargs: Any) -> float:
        return 0.0

    def clip_grad_and_step(self, *args: Any, **kwargs: Any) -> None:
        return None

    def tinker_step(self, *, adam_params: Any = None, **kwargs: Any) -> None:
        return None

    def tinker_calculate_metric(self, is_training: bool, **kwargs: Any) -> dict[str, float]:
        return {'loss': 0.5, 'grad_norm': 0.1}

    def calculate_metric(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        return {'loss': 0.5, 'grad_norm': 0.1}

    def tinker_load(self, checkpoint_dir: str, **kwargs: Any) -> None:
        return None

    # ----- Configuration setters (R1.4) ---------------------------------- #

    def set_loss(self, *args: Any, **kwargs: Any) -> None:
        return None

    def set_optimizer(self, *args: Any, **kwargs: Any) -> None:
        return None

    def set_lr_scheduler(self, *args: Any, **kwargs: Any) -> None:
        return None

    def set_template(self, *args: Any, **kwargs: Any) -> None:
        return None

    def set_processor(self, *args: Any, **kwargs: Any) -> None:
        return None

    def add_metric(self, *args: Any, **kwargs: Any) -> None:
        return None

    def apply_patch(self, *args: Any, **kwargs: Any) -> None:
        return None

    # ----- Persistence stubs (R1.4) -------------------------------------- #

    def save(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {'status': 'ok', 'path': None}

    def load(self, *args: Any, **kwargs: Any) -> None:
        return None

    def resume_from_checkpoint(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {'status': 'ok', 'progress': {}}

    def get_state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    def get_train_configs(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    # ----- Adapter management (R1.5, R1.6, R1.7) ------------------------- #

    def add_adapter(self, adapter_name: str, **cfg: Any) -> None:
        """Record an adapter without loading real weights (R1.5)."""
        self._adapters[adapter_name] = dict(cfg)

    def add_adapter_to_model(self, adapter_name: str, config: Any = None, **cfg: Any) -> None:
        merged: dict[str, Any] = dict(cfg)
        if config is not None:
            merged.setdefault('config', config)
        self._adapters[adapter_name] = merged

    def remove_adapter(self, adapter_name: str) -> None:
        """Remove ``adapter_name`` (R1.6); raise on absent (R1.7)."""
        if adapter_name not in self._adapters:
            raise KeyError(f'adapter not present: {adapter_name}')
        del self._adapters[adapter_name]

    def has_adapter(self, adapter_name: str) -> bool:
        return adapter_name in self._adapters


def _input_seq_lengths(inputs: Any) -> list[int]:
    """Best-effort recovery of per-datum sequence lengths from heterogeneous inputs.

    The real backend pulls lengths from ``Datum.loss_fn_inputs['target_tokens']``,
    but we want to stay numpy-only and avoid importing the tinker types. Falls
    back to ``[1]`` so callers always get at least one record back.
    """
    if inputs is None:
        return [1]
    if isinstance(inputs, list):
        if not inputs:
            return [1]
        out: list[int] = []
        for item in inputs:
            length = _seq_length_of(item)
            out.append(length)
        return out
    return [_seq_length_of(inputs)]


def _seq_length_of(item: Any) -> int:
    # Datum-like: model_input.tokens or loss_fn_inputs['target_tokens']
    for attr in ('model_input', 'inputs', 'tokens'):
        v = getattr(item, attr, None)
        if v is None:
            continue
        tokens = getattr(v, 'tokens', v)
        if hasattr(tokens, '__len__'):
            return max(1, len(tokens))
    if isinstance(item, dict):
        for k in ('input_ids', 'tokens', 'target_tokens'):
            if k in item and hasattr(item[k], '__len__'):
                return max(1, len(item[k]))
    if hasattr(item, '__len__'):
        return max(1, len(item))
    return 1
