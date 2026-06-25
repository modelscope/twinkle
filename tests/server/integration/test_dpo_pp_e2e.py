"""DPO training E2E test on Megatron PP=2 backend.

Reproduces the PP deadlock where forward_only succeeds but forward_backward
hangs due to nccl_safe loss skip breaking pipeline P2P communication.

Flow (mirrors cookbook/client/twinkle/modelscope/dpo.py):
  Phase 1 — Setup: configure model with DPO loss
  Phase 2 — forward_only (ref_outputs): base model inference, no LoRA
  Phase 3 — forward_backward (DPO training): triggers PP P2P communication

No sampler required. Server config: server_config_4b_dpo_megatron.yaml

## How to run

    # 1. Start server (no sampler, 4 GPU model only)
    python tests/server/start_e2e_server.py \\
        --config tests/server/config/server_config_4b_dpo_megatron.yaml

    # 2. Run DPO PP test
    TWINKLE_TEST_GPU_E2E=1 python -u tests/server/integration/test_dpo_pp_e2e.py

Expected: forward_only succeeds, forward_backward either succeeds or
reproduces the PP deadlock (504 timeout / NCCL hang).
"""
from __future__ import annotations

import dotenv

dotenv.load_dotenv('.env')

import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from typing import Any, Dict, List  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from peft import LoraConfig  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get('TWINKLE_TEST_GPU_E2E', '0') != '1',
    reason='Set TWINKLE_TEST_GPU_E2E=1 to run real GPU E2E tests (requires running server)',
)

from twinkle import get_logger, init_twinkle_client  # noqa: E402
from twinkle.dataloader import DataLoader  # noqa: E402
from twinkle.dataset import Dataset, DatasetMeta  # noqa: E402
from twinkle.preprocessor import EmojiDPOProcessor  # noqa: E402
from twinkle_client.model import MultiLoraTransformersModel  # noqa: E402

logger = get_logger()

# ── Configuration ──
BASE_MODEL = 'Qwen/Qwen3.5-4B'
BASE_URL = 'http://localhost:9000'
API_KEY = 'EMPTY_API_KEY'
DATASET_ID = 'ms://hjh0119/shareAI-Llama3-DPO-zh-en-emoji'

BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
DPO_BETA = 0.1
SFT_WEIGHT = 1.0
LOSS_TYPE = 'sigmoid'
MAX_LENGTH = 2048
SYSTEM_PROMPT = 'You are a helpful assistant.'
DPO_TRAIN_STEPS = 4  # small number, just enough to trigger the bug
FORWARD_BACKWARD_TIMEOUT = 120  # seconds to wait before declaring hang


def _create_dpo_dataset() -> Dataset:
    """Create DPO dataset with positive/negative format (small slice for speed)."""
    dataset = Dataset(DatasetMeta(DATASET_ID, data_slice=range(100)))
    dataset.set_template('Qwen3_5Template', model_id=f'ms://{BASE_MODEL}', max_length=MAX_LENGTH)
    dataset.map(EmojiDPOProcessor, init_args={'system': SYSTEM_PROMPT})
    dataset.encode()
    return dataset


def _prepare_dpo_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Interleave positive/negative pairs: [pos_1, neg_1, pos_2, neg_2, ...].

    This DP-safe interleaving ensures each DP worker gets complete pairs
    after slicing.
    """
    result = []
    for row in batch:
        base_fields = {k: v for k, v in row.items() if k not in ('positive', 'negative')}
        pos_sample = {**base_fields, **row['positive']}
        neg_sample = {**base_fields, **row['negative']}
        result.append(pos_sample)
        result.append(neg_sample)
    return result


def _convert_tensors(batch: List[Dict[str, Any]]) -> None:
    """Convert numpy/torch tensors to lists for serialization (in-place)."""
    for row in batch:
        for key in row:
            if isinstance(row[key], np.ndarray):
                row[key] = row[key].tolist()
            elif isinstance(row[key], torch.Tensor):
                row[key] = row[key].cpu().numpy().tolist()


def _configure_dpo_model() -> MultiLoraTransformersModel:
    """Configure model with DPO loss, optimizer, and LoRA adapter."""
    model = MultiLoraTransformersModel(model_id=f'ms://{BASE_MODEL}')
    model.add_adapter_to_model(
        'default',
        LoraConfig(target_modules='all-linear', r=8, lora_alpha=32, lora_dropout=0.05),
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    )
    model.set_template('Qwen3_5Template')
    model.set_processor('InputProcessor', padding_side='right')
    model.set_loss('DPOLoss', beta=DPO_BETA, loss_type=LOSS_TYPE, reference_free=False, sft_weight=SFT_WEIGHT)
    model.add_metric('DPOMetric', beta=DPO_BETA)
    model.set_optimizer('Adam', lr=1e-4)
    return model


def main() -> int:
    client = init_twinkle_client(base_url=BASE_URL, api_key=API_KEY)
    logger.info('Available models:')
    for m in client.get_server_capabilities().supported_models:
        logger.info(f'  - {m.model_name}')

    # ── Phase 1: Setup ──
    logger.info('=' * 60)
    logger.info('Phase 1: Setup — configure DPO model + load dataset')
    logger.info('=' * 60)

    dataset = _create_dpo_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE)
    model = _configure_dpo_model()

    logger.info('Model and dataset ready. Starting DPO training loop...')
    logger.info(f'DPO config: beta={DPO_BETA}, loss_type={LOSS_TYPE}, sft_weight={SFT_WEIGHT}')
    logger.info(f'Training {DPO_TRAIN_STEPS} steps with GA={GRADIENT_ACCUMULATION_STEPS}')

    # ── Phase 2+3: DPO training loop ──
    logger.info('=' * 60)
    logger.info('Phase 2+3: DPO training (forward_only + forward_backward)')
    logger.info('=' * 60)

    step = 0
    for batch in dataloader:
        _convert_tensors(batch)
        dpo_batch = _prepare_dpo_batch(batch)

        # Phase 2: forward_only — get reference outputs (base model, no LoRA)
        logger.info(f'[Step {step + 1}] forward_only (ref_outputs) ...')
        t0 = time.time()
        ref_outputs = model.forward_only(inputs=dpo_batch, disable_lora=True)
        t_fo = time.time() - t0
        logger.info(f'[Step {step + 1}] forward_only OK ({t_fo:.1f}s)')

        # Phase 3: forward_backward — DPO training with ref_outputs
        logger.info(f'[Step {step + 1}] forward_backward (DPO loss) ...')
        t0 = time.time()
        model.forward_backward(inputs=dpo_batch, ref_outputs=ref_outputs.result)
        t_fb = time.time() - t0
        logger.info(f'[Step {step + 1}] forward_backward OK ({t_fb:.1f}s)')

        model.clip_grad_and_step()

        # Log metrics every GA steps
        step += 1
        if step % GRADIENT_ACCUMULATION_STEPS == 0:
            metrics = model.calculate_metric(is_training=True)
            logger.info(f'[Optim step {step // GRADIENT_ACCUMULATION_STEPS}] {metrics}')

        if step >= DPO_TRAIN_STEPS:
            break

    logger.info('=' * 60)
    logger.info('ALL DPO PHASES PASSED')
    logger.info('=' * 60)
    return 0


# ── pytest entry point ──

def test_dpo_pp_e2e():
    """Pytest-collected entry point for the DPO PP E2E suite."""
    rc = main()
    assert rc == 0, 'DPO PP E2E test failed'


if __name__ == '__main__':
    sys.exit(main())
