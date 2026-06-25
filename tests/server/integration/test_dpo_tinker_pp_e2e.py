"""Tinker client DPO test on Megatron PP=2 backend.

Reproduces the tinker_forward_only ragged logps error.

Usage:
    # 1. Start server (PP=2, no sampler)
    python tests/server/start_e2e_server.py \
        --config tests/server/config/server_config_4b_dpo_megatron.yaml

    # 2. Run test
    TWINKLE_TEST_GPU_E2E=1 python -u tests/server/integration/test_dpo_tinker_pp_e2e.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

SERVER_URL = os.environ.get('TWINKLE_SERVER_URL', 'http://localhost:9000')
BASE_MODEL = 'Qwen/Qwen3.5-4B'


def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}][INFO:twinkle] {msg}', flush=True)


def main():
    if not os.environ.get('TWINKLE_TEST_GPU_E2E'):
        log('SKIP: set TWINKLE_TEST_GPU_E2E=1 to run')
        return 0

    # ── Init tinker client ────────────────────────────────────────────
    from tinker import types
    from twinkle import init_tinker_client
    init_tinker_client()
    from tinker import ServiceClient

    service_client = ServiceClient(base_url=SERVER_URL, api_key='EMPTY_TOKEN')
    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL, rank=8,
    )
    log(f'Tinker training client ready (model={BASE_MODEL})')

    # ── Prepare DPO dataset ───────────────────────────────────────────
    from twinkle.dataset import Dataset, DatasetMeta
    from twinkle.preprocessor import EmojiDPOProcessor
    from twinkle.server.common import input_feature_to_datum

    log('Loading DPO dataset (10 samples)...')
    dataset = Dataset(DatasetMeta(f'ms://hjh0119/shareAI-Llama3-DPO-zh-en-emoji', data_slice=range(10)))
    dataset.set_template('Qwen3_5Template', model_id=f'ms://{BASE_MODEL}', max_length=1024)
    dataset.map(EmojiDPOProcessor, init_args={'system': 'You are a helpful assistant.'})
    dataset.encode()

    # Build interleaved [pos, neg, pos, neg] batch
    batch = list(dataset)[:4]
    dpo_batch = []
    for row in batch:
        for key in list(row.keys()):
            if isinstance(row[key], np.ndarray):
                row[key] = row[key].tolist()
            elif isinstance(row[key], torch.Tensor):
                row[key] = row[key].cpu().numpy().tolist()
        base_fields = {k: v for k, v in row.items() if k not in ('positive', 'negative')}
        dpo_batch.append({**base_fields, **row['positive']})
        dpo_batch.append({**base_fields, **row['negative']})
    log(f'DPO batch: {len(dpo_batch)} samples (interleaved pos/neg)')

    # Convert to Tinker Datums
    input_datums = [input_feature_to_datum(row) for row in dpo_batch]
    seq_lens = [d.loss_fn_inputs['target_tokens'].to_numpy().shape[0] for d in input_datums]
    log(f'Datum seq_lens: {seq_lens}')

    # ── Step 1: Reference forward (tinker_forward_only) ───────────────
    log('=' * 60)
    log('Step 1: tinker forward (reference, disable_lora=True)...')
    log('=' * 60)
    start = time.time()
    try:
        ref_result = training_client.forward(
            input_datums, 'cross_entropy',
            loss_fn_config={'disable_lora': True},
        ).result()
        elapsed = time.time() - start
        log(f'Step 1 OK ({elapsed:.1f}s), {len(ref_result.loss_fn_outputs)} outputs')

        # Show logprobs shapes
        for i, out in enumerate(ref_result.loss_fn_outputs):
            lp = out.get('logprobs')
            if lp is not None:
                arr = np.array(lp.tolist())
                log(f'  output[{i}] logprobs shape={arr.shape}')
    except Exception as e:
        elapsed = time.time() - start
        log(f'Step 1 FAILED ({elapsed:.1f}s): {type(e).__name__}: {e}')
        if elapsed > 120:
            log('TIMEOUT — likely NCCL hang!')
        log('Check server log for traceback')
        return 1

    # ── Step 2: Attach ref_logps to datums ────────────────────────────
    log('Step 2: Attaching ref_logps to datums...')
    for datum, ref_out in zip(input_datums, ref_result.loss_fn_outputs):
        ref_logprobs_np = np.array(ref_out['logprobs'].tolist(), dtype=np.float32)
        datum.loss_fn_inputs['ref_logps'] = types.TensorData.from_numpy(ref_logprobs_np)
        log(f'  ref_logps shape={ref_logprobs_np.shape}')

    # ── Step 3: DPO forward_backward (tinker_forward_backward) ────────
    log('=' * 60)
    log('Step 3: tinker forward_backward (DPO loss)...')
    log('=' * 60)
    start = time.time()
    try:
        fwdbwd_result = training_client.forward_backward(
            input_datums, 'importance_sampling',
            loss_fn_config={'dpo_beta': 0.1, 'dpo_sft_weight': 1.0},
        ).result()
        elapsed = time.time() - start
        log(f'Step 3 OK ({elapsed:.1f}s)')
    except Exception as e:
        elapsed = time.time() - start
        log(f'Step 3 FAILED ({elapsed:.1f}s): {type(e).__name__}: {e}')
        if elapsed > 120:
            log('TIMEOUT — likely NCCL hang!')
        return 1

    # ── Step 4: Optimizer step ────────────────────────────────────────
    log('Step 4: optim_step...')
    try:
        optim_result = training_client.optim_step(
            types.AdamParams(learning_rate=1e-4)
        ).result()
        log(f'Step 4 OK, metrics={optim_result.metrics}')
    except Exception as e:
        log(f'Step 4 FAILED: {e}')
        return 1

    log('=' * 60)
    log('ALL TINKER DPO PHASES PASSED')
    log('=' * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
