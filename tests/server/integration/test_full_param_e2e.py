"""Full-parameter (train_mode: full) e2e: create → SFT → guards → save → sample.

Drives the tinker-compatible client against a server whose model deployment
runs in exclusive full-parameter mode (no PEFT/MultiLora). Start the server
with a full-parameter variant, e.g.:

    python tests/server/start_e2e_server.py --variant full             # transformers DP=2
    python tests/server/start_e2e_server.py --variant full-fsdp2       # transformers FSDP2
    python tests/server/start_e2e_server.py --variant megatron-full    # megatron DP=2 PP=2
    python tests/server/start_e2e_server.py --variant megatron-full-tp2pp2
    python tests/server/start_e2e_server.py --variant megatron-full-tp2dp2

Phase A — create_full_training_client + SFT training, assert loss decreases
Phase B — create_lora_training_client must be rejected (mode mismatch)
Phase C — second create_full_training_client must be rejected (exclusive busy)
Phase D — save_weights_and_get_sampling_client → sample, non-empty output
Phase E — on-disk checkpoint is a FULL HF checkpoint (no adapter_config.json)

## How to run

    TWINKLE_TEST_GPU_E2E=1 python -u tests/server/integration/test_full_param_e2e.py

Expected last line: ``ALL PHASES PASSED``.
"""
from __future__ import annotations

import dotenv

dotenv.load_dotenv('.env')

import glob  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get('TWINKLE_TEST_GPU_E2E', '0') != '1',
    reason='Set TWINKLE_TEST_GPU_E2E=1 to run real GPU E2E tests (requires running server)',
)

from twinkle import get_logger, init_tinker_client  # noqa: E402
from twinkle.dataloader import DataLoader  # noqa: E402
from twinkle.dataset import Dataset, DatasetMeta  # noqa: E402
from twinkle.preprocessor import SelfCognitionProcessor  # noqa: E402
from twinkle.server.common import input_feature_to_datum  # noqa: E402

init_tinker_client()

from tinker import ServiceClient, types  # noqa: E402

logger = get_logger()

BASE_MODEL = 'Qwen/Qwen3.5-4B'
BASE_URL = os.environ.get('TWINKLE_SERVER_URL', 'http://localhost:9000')
API_KEY = os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY_TOKEN')
OUTPUTS_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'outputs')
TRAIN_STEPS = 16
LEARNING_RATE = 1e-5
SAMPLE_MAX_TOKENS = 32


def _build_dataloader(batch_size: int = 8):
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(500)))
    dataset.set_template('Qwen3_5Template', model_id=f'ms://{BASE_MODEL}', max_length=256)
    dataset.map(SelfCognitionProcessor('twinkle模型', 'twinkle团队'), load_from_cache_file=False)
    dataset.encode(batched=True, load_from_cache_file=False)
    return DataLoader(dataset=dataset, batch_size=batch_size)


def _loss_per_token(fwdbwd_result, input_datum) -> float:
    logprobs = np.concatenate([output['logprobs'].tolist() for output in fwdbwd_result.loss_fn_outputs])
    weights = np.concatenate([example.loss_fn_inputs['weights'].tolist() for example in input_datum])
    return float(-np.dot(logprobs, weights) / weights.sum())


def main() -> int:
    t0 = time.time()
    service_client = ServiceClient(base_url=BASE_URL, api_key=API_KEY)

    # ── Phase A: full-parameter training ──
    logger.info('=' * 60)
    logger.info('Phase A: create_full_training_client + SFT (%d steps)', TRAIN_STEPS)
    logger.info('=' * 60)
    training_client = service_client.create_full_training_client(base_model=BASE_MODEL)
    logger.info('Full training client created: model_id=%s', training_client.model_id)

    dataloader = _build_dataloader()
    losses = []
    step = 0
    for batch in dataloader:
        input_datum = [input_feature_to_datum(f) for f in batch]
        fwdbwd_future = training_client.forward_backward(input_datum, 'cross_entropy')
        optim_future = training_client.optim_step(types.AdamParams(learning_rate=LEARNING_RATE))
        fwdbwd_result = fwdbwd_future.result()
        optim_future.result()
        loss = _loss_per_token(fwdbwd_result, input_datum)
        losses.append(loss)
        step += 1
        logger.info('[A] step=%d loss=%.4f', step, loss)
        if step >= TRAIN_STEPS:
            break

    first3, last3 = float(np.mean(losses[:3])), float(np.mean(losses[-3:]))
    assert last3 < first3, f'Phase A FAIL: loss did not decrease (first3={first3:.4f}, last3={last3:.4f})'
    logger.info('Phase A OK: loss %.4f -> %.4f', first3, last3)

    # ── Phase B: LoRA create must be rejected on a full deployment ──
    logger.info('Phase B: create_lora_training_client must be rejected')
    try:
        service_client.create_lora_training_client(base_model=BASE_MODEL, rank=8)
        raise AssertionError('Phase B FAIL: LoRA create unexpectedly succeeded on a full deployment')
    except AssertionError:
        raise
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        assert 'full-parameter' in msg or 'lora_config' in msg, f'Phase B FAIL: unexpected error: {msg[:500]}'
        logger.info('Phase B OK: rejected with mode-mismatch error')

    # ── Phase C: second full tenant must be rejected (exclusive) ──
    logger.info('Phase C: second create_full_training_client must be rejected (busy)')
    second_client = ServiceClient(base_url=BASE_URL, api_key=API_KEY)
    try:
        second_client.create_full_training_client(base_model=BASE_MODEL)
        raise AssertionError('Phase C FAIL: second full tenant unexpectedly succeeded')
    except AssertionError:
        raise
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        assert 'exclusive' in msg or 'already' in msg, f'Phase C FAIL: unexpected error: {msg[:500]}'
        logger.info('Phase C OK: rejected with exclusive-busy error')

    # ── Phase D: save full weights for sampler + sample ──
    logger.info('Phase D: save_weights_and_get_sampling_client + sample')
    save_start = time.time()
    sampling_client = training_client.save_weights_and_get_sampling_client(name='full-e2e')
    logger.info('[D] sampler weights saved in %.0fs', time.time() - save_start)

    from twinkle.data_format import Message, Trajectory
    from twinkle.template import Template
    template = Template(model_id=f'ms://{BASE_MODEL}')
    trajectory = Trajectory(messages=[
        Message(role='system', content='You are a helpful assistant'),
        Message(role='user', content='你是谁？'),
    ])
    input_feature = template.batch_encode([trajectory], add_generation_prompt=True)[0]
    prompt = types.ModelInput.from_ints(input_feature['input_ids'].tolist())
    params = types.SamplingParams(max_tokens=SAMPLE_MAX_TOKENS, temperature=0.0)
    result = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=2).result()
    assert result.sequences and all(len(seq.tokens) > 0 for seq in result.sequences), \
        'Phase D FAIL: empty sampling output'
    for i, seq in enumerate(result.sequences):
        decoded = template.decode(seq.tokens)
        logger.info('[D] sample %d: %s', i, decoded[:120].replace('\n', ' '))
    logger.info('Phase D OK: sampled %d sequences on full weights', len(result.sequences))

    # ── Phase E: on-disk checkpoint format ──
    logger.info('Phase E: verify on-disk checkpoint is a full HF checkpoint')
    candidates = []
    for pattern in ('model.safetensors', 'model.safetensors.index.json'):
        candidates.extend(glob.glob(os.path.join(OUTPUTS_ROOT, '**', pattern), recursive=True))
    fresh = [p for p in candidates if os.path.getmtime(p) > save_start]
    assert fresh, f'Phase E FAIL: no fresh full-weight files under {os.path.abspath(OUTPUTS_ROOT)}'
    ckpt_dir = os.path.dirname(sorted(fresh, key=os.path.getmtime)[-1])
    assert not os.path.exists(os.path.join(ckpt_dir, 'adapter_config.json')), \
        f'Phase E FAIL: {ckpt_dir} contains adapter_config.json (LoRA format!)'
    assert not glob.glob(os.path.join(ckpt_dir, 'adapter_model*')), \
        f'Phase E FAIL: {ckpt_dir} contains adapter_model files (LoRA format!)'
    total_gb = sum(os.path.getsize(p) for p in glob.glob(os.path.join(ckpt_dir, '*.safetensors'))) / 1e9
    logger.info('Phase E OK: %s is a full checkpoint (%.1f GB safetensors)', ckpt_dir, total_gb)

    logger.info('Total elapsed: %.0fs', time.time() - t0)
    logger.info('ALL PHASES PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
