# Copyright (c) ModelScope Contributors. All rights reserved.
"""Mock model backend.

Provides ``TwinkleCompatMockModel``, a stand-in for the real
``TwinkleCompatTransformersModel`` whose only purpose is to exercise the
server's HTTP and dispatch paths without a real GPU model. Determinism is
keyed by ``(model_id, adapter_name, seed, input_shape)`` so repeated
requests with the same payload produce identical numpy-derived results.

The class is duck-typed against ``TwinkleCompatModelBase`` rather than
subclassing it — the base class lives in a torch-importing module.
"""
from __future__ import annotations

import numpy as np
import os
from typing import Any

from twinkle import remote_class, remote_function
from twinkle.utils.logger import get_logger
from twinkle.utils.seed import stable_seed

logger = get_logger()


@remote_class()
class TwinkleCompatMockModel:
    """Deterministic mock model for CPU-only testing."""

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
        if kwargs:
            logger.debug('MockModel ignoring unknown ctor kwargs: %s', sorted(kwargs))

    # ----- Forward family ------------------------------------------------ #

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
            rng = np.random.default_rng(stable_seed(self.model_id, adapter_name, self._rng_seed, idx, seq_len))
            logprobs = rng.uniform(-2.0, 0.0, size=seq_len).astype(np.float32)
            elementwise_loss = rng.uniform(0.0, 1.0, size=seq_len).astype(np.float32)
            out.append({
                'logprobs': logprobs.tolist(),
                'elementwise_loss': elementwise_loss.tolist(),
                'loss': float(loss_value),
            })
        return out

    @remote_function()
    def tinker_forward_only(self, *, inputs: Any, adapter_name: str | None = None, **kwargs: Any) -> list[Any]:
        return [_to_tinker_loss_outputs(self._build_forward_result(inputs, adapter_name)), 0.0]

    @remote_function()
    def tinker_forward_backward(self, *, inputs: Any, adapter_name: str, loss_fn: str, **kwargs: Any) -> list[Any]:
        loss_seed = stable_seed(self.model_id, adapter_name, self._rng_seed, 'loss', loss_fn)
        loss = float(np.random.default_rng(loss_seed).uniform(0.0, 1.0))
        return [_to_tinker_loss_outputs(self._build_forward_result(inputs, adapter_name, loss_value=loss)), loss]

    @remote_function()
    def forward(self, *, inputs: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return self._build_forward_result(inputs, kwargs.get('adapter_name'))

    @remote_function()
    def forward_only(self, *, inputs: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return self._build_forward_result(inputs, kwargs.get('adapter_name'))

    @remote_function()
    def forward_backward(self, *, inputs: Any, **kwargs: Any) -> list[Any]:
        loss = float(np.random.default_rng(self._rng_seed).uniform(0.0, 1.0))
        return [self._build_forward_result(inputs, kwargs.get('adapter_name'), loss_value=loss), loss]

    @remote_function()
    def calculate_loss(self, *, inputs: Any = None, **kwargs: Any) -> float:
        # Handler at twinkle_handlers.py:150 invokes with only ``adapter_name``;
        # the real backend keeps loss state on ``self`` so ``inputs`` is optional.
        return float(np.random.default_rng(self._rng_seed).uniform(0.0, 1.0))

    # ----- Backward / optimizer ------------------------------------------ #

    @remote_function()
    def backward(self, *args: Any, **kwargs: Any) -> None:
        return None

    @remote_function()
    def step(self, *args: Any, **kwargs: Any) -> None:
        return None

    @remote_function()
    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        return None

    @remote_function()
    def lr_step(self, *args: Any, **kwargs: Any) -> None:
        return None

    @remote_function()
    def clip_grad_norm(self, *args: Any, **kwargs: Any) -> float:
        return 0.0

    @remote_function()
    def clip_grad_and_step(self, *args: Any, **kwargs: Any) -> None:
        return None

    @remote_function()
    def tinker_step(self, *, adam_params: Any = None, **kwargs: Any) -> None:
        return None

    @remote_function()
    def tinker_calculate_metric(self, is_training: bool, **kwargs: Any) -> dict[str, float]:
        return {'loss': 0.5, 'grad_norm': 0.1}

    @remote_function()
    def calculate_metric(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        return {'loss': 0.5, 'grad_norm': 0.1}

    @remote_function()
    def tinker_load(self, checkpoint_dir: str, **kwargs: Any) -> None:
        return None

    # ----- Configuration setters ----------------------------------------- #

    @remote_function()
    def set_loss(self, *args: Any, **kwargs: Any) -> None:
        return None

    @remote_function()
    def set_optimizer(self, *args: Any, **kwargs: Any) -> None:
        return None

    @remote_function()
    def set_lr_scheduler(self, *args: Any, **kwargs: Any) -> None:
        return None

    @remote_function()
    def set_template(self, *args: Any, **kwargs: Any) -> None:
        return None

    @remote_function()
    def set_processor(self, *args: Any, **kwargs: Any) -> None:
        return None

    @remote_function()
    def add_metric(self, *args: Any, **kwargs: Any) -> None:
        return None

    @remote_function()
    def apply_patch(self, *args: Any, **kwargs: Any) -> None:
        return None

    # ----- Persistence stubs --------------------------------------------- #

    @remote_function()
    def save(self, name: str = 'latest', output_dir: str | None = None, **kwargs: Any) -> str:
        """Materialize an empty checkpoint dir so downstream existence checks pass.

        The real backend creates ``<output_dir>/<name>/`` on disk and returns its
        path. Tinker /asample later does ``os.path.exists(...)`` on a symlink
        pointing here (see ``sampler/tinker_handlers.py:72``), so the directory
        has to really exist for the mock path to clear.
        """
        target = os.path.join(output_dir or '/tmp/twinkle_mock_ckpt', name)
        os.makedirs(target, exist_ok=True)
        return target

    @remote_function()
    def load(self, *args: Any, **kwargs: Any) -> None:
        return None

    @remote_function()
    def resume_from_checkpoint(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {'status': 'ok', 'progress': {}}

    @remote_function()
    def get_state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    @remote_function()
    def get_train_configs(self, *args: Any, **kwargs: Any) -> str:
        # Real backend returns a JSON-serialized config string; handler wraps it
        # into ``GetTrainConfigsResponse.result: str`` (types/model.py:213).
        return '{}'

    @remote_function()
    def upload_to_hub(self, *args: Any, **kwargs: Any) -> None:
        return None

    # ----- Adapter management -------------------------------------------- #

    @remote_function()
    def add_adapter(self, adapter_name: str, **cfg: Any) -> None:
        """Record an adapter without loading real weights."""
        self._adapters[adapter_name] = dict(cfg)

    @remote_function()
    def add_adapter_to_model(self, adapter_name: str, config: Any = None, **cfg: Any) -> None:
        merged: dict[str, Any] = dict(cfg)
        if config is not None:
            merged.setdefault('config', config)
        self._adapters[adapter_name] = merged

    @remote_function()
    def remove_adapter(self, adapter_name: str) -> None:
        """Remove ``adapter_name``; raise ``KeyError`` if it was never added."""
        if adapter_name not in self._adapters:
            raise KeyError(f'adapter not present: {adapter_name}')
        del self._adapters[adapter_name]

    @remote_function()
    def has_adapter(self, adapter_name: str) -> bool:
        return adapter_name in self._adapters


def _to_tinker_loss_outputs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Wrap mock's plain numpy-derived dicts into ``tinker.TensorData`` instances.

    The Tinker handlers feed mock's output directly into
    ``tinker.types.ForwardBackwardOutput.loss_fn_outputs`` which is typed as
    ``List[Dict[str, TensorData]]``. The real backend constructs TensorData
    via ``TensorData.from_torch``; we synthesize the same shape from numpy
    lists here. Imported lazily so the module stays importable when the
    optional ``tinker`` package is absent.
    """
    from tinker.types import TensorData  # noqa: WPS433 (lazy on purpose)

    out: list[dict[str, Any]] = []
    for rec in records:
        logprobs = rec['logprobs']
        elementwise_loss = rec['elementwise_loss']
        loss = float(rec['loss'])
        out.append({
            'logprobs':
            TensorData(data=list(logprobs), dtype='float32', shape=[len(logprobs)]),
            'elementwise_loss':
            TensorData(data=list(elementwise_loss), dtype='float32', shape=[len(elementwise_loss)]),
            'loss':
            TensorData(data=[loss], dtype='float32', shape=[]),
        })
    return out


def _input_seq_lengths(inputs: Any) -> list[int]:
    """Best-effort recovery of per-datum sequence lengths from heterogeneous inputs.

    Falls back to ``[1]`` so callers always get at least one record back.
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
