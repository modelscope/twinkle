# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared utilities for E2E integration tests.

Provides reusable helpers for server health check, client initialization,
dataset preparation, and test assertions across all 12 test combinations
(2 backends x 2 clients x 3 tasks).
"""
from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

BASE_MODEL = 'Qwen/Qwen3.5-4B'
MODEL_ID = f'ms://{BASE_MODEL}'
BASE_URL = os.environ.get('TWINKLE_SERVER_URL', 'http://localhost:9000')
API_KEY = 'EMPTY_API_KEY'
TIMEOUT = 120  # seconds per operation before declaring hang
GRADIENT_ACCUMULATION_STEPS = 2  # Megatron requires GA >= 2


def get_backend() -> str:
    """Read backend type from environment variable."""
    backend = os.environ.get('TWINKLE_TEST_BACKEND', 'transformers').lower()
    assert backend in ('transformers', 'megatron'), (
        f'Invalid TWINKLE_TEST_BACKEND={backend!r}, must be transformers or megatron')
    return backend


# ═══════════════════════════════════════════════════════════════════════════
# Server Health Check
# ═══════════════════════════════════════════════════════════════════════════

def wait_for_server(url: str = BASE_URL, timeout: int = 300) -> None:
    """Wait for Twinkle server to become ready using Python requests."""
    import requests

    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f'{url}/-/routes', timeout=5)
            if resp.status_code == 200:
                elapsed = int(time.time() - start)
                log(f'Server ready (waited {elapsed}s)')
                return
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(5)
    raise TimeoutError(f'Server not ready after {timeout}s at {url}')


def log(msg: str) -> None:
    """Timestamped log output."""
    ts = time.strftime('%H:%M:%S')
    print(f'[{ts}][E2E] {msg}', flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# Dataset Factories
# ═══════════════════════════════════════════════════════════════════════════

def create_sft_dataset(data_slice=range(100)):
    """Create SelfCognition SFT dataset (small slice for speed)."""
    from twinkle.dataloader import DataLoader
    from twinkle.dataset import Dataset, DatasetMeta

    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=data_slice))
    dataset.set_template('Qwen3_5Template', model_id=MODEL_ID, max_length=256)
    dataset.map('SelfCognitionProcessor', init_args={'model_name': 'twinkle模型', 'model_author': 'ModelScope社区'})
    dataset.encode(batched=True)
    return dataset


def create_dpo_dataset(data_slice=range(50)):
    """Create EmojiDPO dataset with positive/negative format."""
    from twinkle.dataset import Dataset, DatasetMeta
    from twinkle.preprocessor import EmojiDPOProcessor

    dataset = Dataset(DatasetMeta('ms://hjh0119/shareAI-Llama3-DPO-zh-en-emoji', data_slice=data_slice))
    dataset.set_template('Qwen3_5Template', model_id=MODEL_ID, max_length=1024)
    dataset.map(EmojiDPOProcessor, init_args={'system': 'You are a helpful assistant.'})
    dataset.encode()
    return dataset


def create_grpo_dataset(data_slice=range(50)):
    """Create GSM8K dataset for GRPO training."""
    from twinkle.dataset import Dataset, DatasetMeta
    from twinkle.preprocessor.llm import GSM8KProcessor

    system_prompt = ('You are a helpful math assistant. Solve the problem with minimal but correct reasoning '
                     'and put your final answer within \\boxed{}.')
    dataset = Dataset(DatasetMeta('ms://modelscope/gsm8k', subset_name='main', split='train', data_slice=data_slice))
    dataset.set_template('Qwen3_5Template', model_id=MODEL_ID, max_length=2048, enable_thinking=False)
    dataset.map(GSM8KProcessor(system=system_prompt))
    dataset.encode(add_generation_prompt=True)
    return dataset


# ═══════════════════════════════════════════════════════════════════════════
# Twinkle Client Factories
# ═══════════════════════════════════════════════════════════════════════════

def init_twinkle_client_session():
    """Initialize the Twinkle client session."""
    from twinkle import init_twinkle_client
    return init_twinkle_client(base_url=BASE_URL, api_key=API_KEY)


def create_twinkle_sft_model():
    """Create Twinkle model configured for SFT (CrossEntropyLoss)."""
    from peft import LoraConfig
    from twinkle_client.model import MultiLoraTransformersModel

    model = MultiLoraTransformersModel(model_id=MODEL_ID)
    model.add_adapter_to_model(
        'default',
        LoraConfig(target_modules='all-linear'),
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    )
    model.set_template('Qwen3_5Template')
    model.set_processor('InputProcessor', padding_side='right')
    model.set_loss('CrossEntropyLoss')
    model.set_optimizer('Adam', lr=1e-4)
    return model


def create_twinkle_dpo_model():
    """Create Twinkle model configured for DPO training."""
    from peft import LoraConfig
    from twinkle_client.model import MultiLoraTransformersModel

    model = MultiLoraTransformersModel(model_id=MODEL_ID)
    model.add_adapter_to_model(
        'default',
        LoraConfig(target_modules='all-linear', r=8, lora_alpha=32, lora_dropout=0.05),
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    )
    model.set_template('Qwen3_5Template')
    model.set_processor('InputProcessor', padding_side='right')
    model.set_loss('DPOLoss', beta=0.1, loss_type='sigmoid', reference_free=False, sft_weight=1.0)
    model.add_metric('DPOMetric', beta=0.1)
    model.set_optimizer('Adam', lr=1e-4)
    return model


def create_twinkle_grpo_model():
    """Create Twinkle model configured for GRPO training."""
    from peft import LoraConfig
    from twinkle_client.model import MultiLoraTransformersModel

    model = MultiLoraTransformersModel(model_id=MODEL_ID)
    model.add_adapter_to_model(
        'default',
        LoraConfig(target_modules='all-linear', r=8, lora_alpha=32, lora_dropout=0.05),
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    )
    model.set_loss('GRPOLoss', epsilon=0.2, beta=0.0)
    model.set_optimizer('Adam', lr=2e-5)
    model.set_processor('InputProcessor')
    model.set_template('Qwen3_5Template', model_id=MODEL_ID)
    return model


def create_twinkle_sampler():
    """Create Twinkle vLLM sampler."""
    from twinkle_client.sampler import vLLMSampler

    sampler = vLLMSampler(model_id=MODEL_ID)
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID)
    return sampler


# ═══════════════════════════════════════════════════════════════════════════
# Tinker Client Factories
# ═══════════════════════════════════════════════════════════════════════════

def init_tinker_client_session():
    """Initialize the Tinker client session and return ServiceClient."""
    from twinkle import init_tinker_client
    init_tinker_client()
    from tinker import ServiceClient
    return ServiceClient(base_url=BASE_URL, api_key=API_KEY)


def create_tinker_training_client(rank: int = 8):
    """Create Tinker LoRA training client."""
    service_client = init_tinker_client_session()
    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=rank,
    )
    return training_client


# ═══════════════════════════════════════════════════════════════════════════
# Data Processing Utilities
# ═══════════════════════════════════════════════════════════════════════════

def convert_tensors(batch: List[Dict[str, Any]]) -> None:
    """Convert numpy/torch tensors to Python lists in-place for serialization."""
    import torch

    for row in batch:
        for key in list(row.keys()):
            val = row[key]
            if isinstance(val, np.ndarray):
                row[key] = val.tolist()
            elif isinstance(val, torch.Tensor):
                row[key] = val.cpu().numpy().tolist()
            elif isinstance(val, dict):
                for k2, v2 in val.items():
                    if isinstance(v2, np.ndarray):
                        val[k2] = v2.tolist()
                    elif isinstance(v2, torch.Tensor):
                        val[k2] = v2.cpu().numpy().tolist()


def prepare_dpo_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reorganize batch into DP-safe interleaved format [pos_1, neg_1, pos_2, neg_2, ...]."""
    result = []
    for row in batch:
        base_fields = {k: v for k, v in row.items() if k not in ('positive', 'negative')}
        pos_sample = {**base_fields, **row['positive']}
        neg_sample = {**base_fields, **row['negative']}
        result.append(pos_sample)
        result.append(neg_sample)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Assertions / Pass Criteria
# ═══════════════════════════════════════════════════════════════════════════

def assert_no_timeout(elapsed: float, label: str, timeout: float = TIMEOUT) -> None:
    """Assert operation completed within timeout (no NCCL hang)."""
    assert elapsed < timeout, (
        f'[{label}] TIMEOUT ({elapsed:.1f}s > {timeout}s) — possible NCCL hang!')


def assert_loss_decreases(losses: List[float], label: str) -> None:
    """Assert training loss shows a downward trend.

    Verifies that the average of the last 3 loss values is lower than
    the average of the first 3 loss values.
    """
    assert len(losses) >= 4, f'[{label}] Need at least 4 loss values, got {len(losses)}'
    first_avg = sum(losses[:3]) / 3
    last_avg = sum(losses[-3:]) / 3
    assert last_avg < first_avg, (
        f'[{label}] Loss did NOT decrease: first_3_avg={first_avg:.4f} >= last_3_avg={last_avg:.4f}')
    log(f'[{label}] Loss decreased: {first_avg:.4f} -> {last_avg:.4f}')


def assert_metrics_valid(metrics: Any, label: str) -> None:
    """Assert metrics contain finite (non-NaN, non-Inf) values."""
    if metrics is None:
        return
    if isinstance(metrics, dict):
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                assert math.isfinite(val), (
                    f'[{label}] Metric {key}={val} is not finite!')
    log(f'[{label}] Metrics valid: {metrics}')
