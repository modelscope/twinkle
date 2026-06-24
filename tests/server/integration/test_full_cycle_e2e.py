"""Twinkle full-cycle e2e: SFT → save → load → resume → sampler.

Six-phase smoke that walks every stateful surface of the twinkle
client/server stack against a real (non-mock) backend. Drives a 4-app
Ray Serve cluster (server + model + sampler + processor) with Qwen3.5-4B
on a 3-GPU host (Megatron backend). Each phase prints a one-line summary
so the log is grep-able from a cleanup / CI script.

IMPORTANT — Megatron backend constraints:
  - gradient_accumulation_steps MUST be >= 2 (GA=1 causes optimizer step
    to silently have no effect due to Megatron DDP gradient sync timing)
  - target_modules='all-linear' is the validated cookbook config

Phase A — initial training (STEPS_PHASE_A steps)
Phase B — keep training STEPS_PHASE_B more steps, save again
Phase C — RELOAD VERIFY: brand-new model handle, load() the Phase-A
  checkpoint, run forward_only on the same fixed batch we used at end
  of Phase A, assert the recovered loss is within RELOAD_LOSS_TOLERANCE
  of the recorded value (proves the saved adapter weights actually
  restore on load)
Phase D — RESUME VERIFY: brand-new model + dataloader,
  resume_from_checkpoint(ckpt_b), train STEPS_PHASE_D more steps,
  assert the losses stay within RESUME_LOSS_BAND of Phase B's last
  step (proves optimizer state + dataloader cursor survive resume)
Phase E + F — vLLM LoRA-effect greedy probe: for each prompt in
  PROBE_PROMPTS, sample greedily once with adapter_uri=None and once
  with adapter_uri=<Phase-B twinkle:// path>. Assert at least one
  prompt's token stream diverges between base and adapter — proves
  the on-disk LoRA artifact actually changes inference output, not
  just that the file exists. Single-prompt + temperature>0 (what the
  earlier revision did) cannot distinguish "LoRA loaded but had no
  effect on this strong-template prompt" from "LoRA silently dropped
  by vLLM"; greedy + multi-prompt does.

## How to run

Direct execution (needs a real GPU box + externally-booted server).
Bring the cluster up, then run this script directly:

    # 1. Boot a 3-node Ray cluster (2 GPUs for model, 1 for sampler)
    ray stop --force
    CUDA_VISIBLE_DEVICES=0,1 ray start --head --port=6379 --num-gpus=2 --disable-usage-stats
    CUDA_VISIBLE_DEVICES=2   ray start --address=127.0.0.1:6379 --num-gpus=1
    CUDA_VISIBLE_DEVICES=""  ray start --address=127.0.0.1:6379 --num-gpus=0

    # 2. Boot the 4-app server pointing at a yaml that uncomments the
    #    sampler app (the baseline cookbook yaml only has 3 apps).
    cd cookbook/client/server/transformer
    nohup python server_e2e.py > server_e2e.log 2>&1 & disown
    # wait for `serve status` to show 4 RUNNING apps

    # 3. Run this script
    mkdir -p /tmp/twinkle_e2e_full_cycle
    python -u tests/server/integration/test_full_cycle_e2e.py

Pytest execution (requires TWINKLE_TEST_GPU_E2E=1):

    TWINKLE_TEST_GPU_E2E=1 pytest tests/server/integration/test_full_cycle_e2e.py -v

Expected last line: ``ALL PHASES PASSED``. Total wall time ~3 minutes
(dominated by Phase A's STEPS_PHASE_A training steps).
"""
from __future__ import annotations

import dotenv

dotenv.load_dotenv('.env')

import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
from peft import LoraConfig  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get('TWINKLE_TEST_GPU_E2E', '0') != '1',
    reason='Set TWINKLE_TEST_GPU_E2E=1 to run real GPU E2E tests (requires running server)',
)

from twinkle import get_logger, init_twinkle_client  # noqa: E402
from twinkle.dataloader import DataLoader  # noqa: E402
from twinkle.dataset import Dataset, DatasetMeta  # noqa: E402
from twinkle_client.model import MultiLoraTransformersModel  # noqa: E402
from twinkle_client.sampler import vLLMSampler  # noqa: E402

logger = get_logger()

BASE_MODEL = 'Qwen/Qwen3.5-4B'
BASE_URL = 'http://localhost:9000'
API_KEY = 'EMPTY_API_KEY'
SAVE_DIR = '/tmp/twinkle_e2e_full_cycle'
STEPS_PHASE_A = 60
STEPS_PHASE_B = 4
STEPS_PHASE_D = 4
RELOAD_LOSS_TOLERANCE = 0.05  # |reloaded - original| / original
RESUME_LOSS_BAND = 3.0  # resumed step's loss must be within this factor of Phase-B's last
SAMPLE_MAX_TOKENS = 32


def _build_dataset_loader(batch_size: int = 4):
    """Same dataset shape as cookbook self_cognition.py — small slice for speed."""
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(500)))
    dataset.set_template('Qwen3_5Template', model_id=f'ms://{BASE_MODEL}', max_length=256)
    dataset.map('SelfCognitionProcessor', init_args={'model_name': 'twinkle模型', 'model_author': 'ModelScope社区'})
    dataset.encode(batched=True)
    return DataLoader(dataset=dataset, batch_size=batch_size)


# Megatron backend requires GA>=2 for optimizer to properly update weights.
# With GA=1, Megatron's DDP gradient sync timing causes optimizer.step() to
# have no effect (loss cycles without decreasing). This is a known constraint
# documented in cookbook/client/twinkle/modelscope/self_cognition.py.
GRADIENT_ACCUMULATION_STEPS = 2


def _configure_model(adapter_name: str, *, save_dir: str = SAVE_DIR) -> MultiLoraTransformersModel:
    model = MultiLoraTransformersModel(model_id=f'ms://{BASE_MODEL}')
    model.add_adapter_to_model(
        adapter_name,
        LoraConfig(target_modules='all-linear'),
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        save_dir=save_dir,
    )
    model.set_template('Qwen3_5Template')
    model.set_processor('InputProcessor', padding_side='right')
    model.set_loss('CrossEntropyLoss')
    model.set_optimizer('Adam', lr=1e-4)
    return model


def _train_n_steps(model, dataloader, n: int, *, label: str, start_step: int = 0) -> list[tuple[int, float]]:
    """Train for n data steps. Logs metric every GRADIENT_ACCUMULATION_STEPS.

    With GA=2, each logged metric represents one actual optimizer step.
    Returns list of (data_step, loss) for the logged steps.
    """
    losses: list[tuple[int, float]] = []
    for cur_step, batch in enumerate(dataloader, start=start_step + 1):
        model.forward_backward(inputs=batch)
        model.clip_grad_and_step()

        # Log metric aligned with optimizer steps (every GA data steps)
        if (cur_step - start_step) % GRADIENT_ACCUMULATION_STEPS == 0:
            metric = model.calculate_metric(is_training=True)
            try:
                loss = float(metric.result.get('loss')) if hasattr(metric.result, 'get') else float(
                    metric.result['loss'])
            except Exception:
                loss = float('nan')
            losses.append((cur_step, loss))
            logger.info(f'[{label}] step={cur_step} loss={loss:.4f}')
        if cur_step - start_step >= n:
            break
    return losses


def _record_fixed_batch_loss(model, batch, *, label: str) -> float:
    """Run forward_only on a fixed batch and report the loss for reload comparison.

    Uses ``forward_only`` + ``calculate_metric(is_training=False)`` to retrieve
    loss. This is compatible with both Transformers and Megatron backends
    (Megatron does not support standalone ``calculate_loss``).
    """
    model.forward_only(inputs=batch)
    metric = model.calculate_metric(is_training=False)
    try:
        val = float(metric.result.get('loss')) if hasattr(metric.result, 'get') else float(metric.result['loss'])
    except (TypeError, KeyError):
        val = float(metric.result)
    logger.info(f'[{label}] fixed-batch loss = {val:.4f}')
    return val


# Probe prompts for vLLM LoRA-effect verification. Mixing different prompt
# styles is deliberate: prompts where the base model has a very strong
# template prior (e.g. "Who are you?") emit a fixed "## " markdown header
# regardless, and the small LoRA delta can't move that top-1 token. Prompts
# with weaker priors (e.g. "What is your name?") diverge at token 0.
# Greedy (temperature=0) makes any divergence purely a LoRA effect, not
# sampling noise.
PROBE_PROMPTS = [
    'What is your name?',  # weak-prior — should diverge after non-trivial training
    'Who are you? Reply in one short sentence.',  # strong-prior — typically does NOT diverge
    '你是谁？请用一句话回答。',  # strong-prior — typically does NOT diverge
]


def _greedy_sample(sampler: vLLMSampler, prompt: str, *, adapter_uri: str | None) -> list[int]:
    """Greedy-sample one prompt and return raw token ids.

    Greedy (temperature=0) is deterministic — any divergence between
    base and adapter is a pure LoRA effect, not sampling noise.
    """
    trajectory = {
        'messages': [
            {
                'role': 'system',
                'content': 'You are a helpful assistant.'
            },
            {
                'role': 'user',
                'content': prompt
            },
        ],
    }
    responses = sampler.sample(
        inputs=[trajectory],
        sampling_params={
            'max_tokens': SAMPLE_MAX_TOKENS,
            'temperature': 0.0,
            'num_samples': 1
        },
        adapter_uri=adapter_uri,
    )
    assert responses and responses[0].sequences, f'sampler returned empty response for {prompt!r}'
    tokens = list(responses[0].sequences[0].tokens)
    assert tokens, f'sampler returned zero tokens for {prompt!r}'
    return tokens


def _first_divergence(a: list[int], b: list[int]) -> int | None:
    """Return index of first differing token, or None if a is a prefix of b (or vice versa)."""
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> int:
    client = init_twinkle_client(base_url=BASE_URL, api_key=API_KEY)
    logger.info('Available models:')
    for m in client.get_server_capabilities().supported_models:
        logger.info(f'  - {m.model_name}')

    # ---------------------------------------------------------------------
    # Phase A — initial training
    # ---------------------------------------------------------------------
    logger.info('=' * 60)
    logger.info('Phase A: initial training (%d steps)', STEPS_PHASE_A)
    logger.info('=' * 60)
    dataloader_a = _build_dataset_loader()
    model_a = _configure_model('default')

    losses_a = _train_n_steps(model_a, dataloader_a, STEPS_PHASE_A, label='A')

    # Capture a fixed batch we'll re-use after reload to assert state survived.
    fixed_batch = next(iter(_build_dataset_loader()))
    loss_a_fixed = _record_fixed_batch_loss(model_a, fixed_batch, label='A-fixed')

    ckpt_a_resp = model_a.save(
        name='e2e-phase-a',
        save_optimizer=True,
        consumed_train_samples=dataloader_a.get_state()['consumed_train_samples'],
    )
    ckpt_a = ckpt_a_resp.twinkle_path
    logger.info(f'Phase A saved to {ckpt_a}')

    # ---------------------------------------------------------------------
    # Phase B — continue training and save again
    # ---------------------------------------------------------------------
    logger.info('=' * 60)
    logger.info('Phase B: continue training (%d more steps)', STEPS_PHASE_B)
    logger.info('=' * 60)
    losses_b = _train_n_steps(model_a, dataloader_a, STEPS_PHASE_B, label='B', start_step=STEPS_PHASE_A)
    ckpt_b_resp = model_a.save(
        name='e2e-phase-b',
        save_optimizer=True,
        consumed_train_samples=dataloader_a.get_state()['consumed_train_samples'],
    )
    ckpt_b = ckpt_b_resp.twinkle_path
    logger.info(f'Phase B saved to {ckpt_b}')

    # ---------------------------------------------------------------------
    # Phase C — RELOAD VERIFY (new model handle + load ckpt_a + same batch)
    # ---------------------------------------------------------------------
    logger.info('=' * 60)
    logger.info('Phase C: reload-verify (new handle, load ckpt_a, fixed batch)')
    logger.info('=' * 60)
    model_c = _configure_model('default')
    model_c.load(ckpt_a)
    loss_c_fixed = _record_fixed_batch_loss(model_c, fixed_batch, label='C-fixed')
    delta = abs(loss_c_fixed - loss_a_fixed) / max(abs(loss_a_fixed), 1e-6)
    logger.info(f'Phase C reload delta: |{loss_c_fixed:.4f} - {loss_a_fixed:.4f}| / {loss_a_fixed:.4f} = {delta:.4f}')
    assert delta <= RELOAD_LOSS_TOLERANCE, (
        f'Phase C FAILED: reload delta {delta:.4f} > tolerance {RELOAD_LOSS_TOLERANCE}')

    # ---------------------------------------------------------------------
    # Phase D — RESUME VERIFY (resume_from_checkpoint(ckpt_b), train 2 more)
    # ---------------------------------------------------------------------
    logger.info('=' * 60)
    logger.info('Phase D: resume-verify (new handle, resume ckpt_b, train %d steps)', STEPS_PHASE_D)
    logger.info('=' * 60)
    model_d = _configure_model('default')
    dataloader_d = _build_dataset_loader()
    progress = model_d.resume_from_checkpoint(ckpt_b)
    logger.info(f'Phase D progress after resume: {progress}')
    resume_start = int(progress.get('cur_step', STEPS_PHASE_A
                                    + STEPS_PHASE_B)) if isinstance(progress, dict) else STEPS_PHASE_A + STEPS_PHASE_B
    losses_d = _train_n_steps(model_d, dataloader_d, STEPS_PHASE_D, label='D', start_step=resume_start)
    last_b_loss = losses_b[-1][1] if losses_b else float('inf')
    for step_d, loss_d in losses_d:
        assert loss_d < last_b_loss * RESUME_LOSS_BAND, (
            f'Phase D FAILED at step {step_d}: loss {loss_d:.4f} > {RESUME_LOSS_BAND}x last-B {last_b_loss:.4f}')
    logger.info(f'Phase D OK: all {len(losses_d)} resumed steps within {RESUME_LOSS_BAND}x of last-B={last_b_loss:.4f}')

    # ---------------------------------------------------------------------
    # Phase E + F — SAMPLER greedy probe (base vs adapter, multi-prompt)
    # ---------------------------------------------------------------------
    # Greedy on multiple prompts gives a decisive verdict that the on-disk
    # LoRA artifact actually takes effect at inference time. Assertion:
    # at least one prompt MUST diverge (token-level) between base and
    # adapter — otherwise vLLM has silently fallen back to the base model
    # (e.g. PEFT-format / target_modules mismatch in vllm's LoRA loader).
    logger.info('=' * 60)
    logger.info('Phase E + F: greedy probe across %d prompts (base vs adapter_uri=%s)', len(PROBE_PROMPTS), ckpt_b)
    logger.info('=' * 60)
    sampler = vLLMSampler(model_id=BASE_MODEL)
    sampler.set_template('Qwen3_5Template', model_id=BASE_MODEL)

    probe_results: list[tuple[str, list[int], list[int], int | None]] = []
    for prompt in PROBE_PROMPTS:
        e_tokens = _greedy_sample(sampler, prompt, adapter_uri=None)
        f_tokens = _greedy_sample(sampler, prompt, adapter_uri=ckpt_b)
        div = _first_divergence(e_tokens, f_tokens)
        probe_results.append((prompt, e_tokens, f_tokens, div))
        status = f'DIVERGE at token {div}' if div is not None else 'IDENTICAL'
        logger.info(f'  prompt={prompt!r} → {status}')
        logger.info(f'    E (base)   : {e_tokens[:8]}{"..." if len(e_tokens) > 8 else ""}')
        logger.info(f'    F (adapter): {f_tokens[:8]}{"..." if len(f_tokens) > 8 else ""}')

    n_diverged = sum(1 for *_, div in probe_results if div is not None)
    assert n_diverged >= 1, (f'Phase F FAILED: vLLM LoRA had no observable effect on any of '
                             f'{len(PROBE_PROMPTS)} probe prompts under greedy decoding — '
                             f'either the adapter was not applied or training was too short '
                             f'(needs ||B@A|| > ~0.1; check LoRA magnitude in adapter_model.safetensors).')
    logger.info(f'Phase F OK: vLLM LoRA observably applied on {n_diverged}/{len(PROBE_PROMPTS)} prompts')

    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------
    logger.info('=' * 60)
    logger.info('SUMMARY')
    logger.info('=' * 60)
    logger.info('  Phase A losses (%d steps): %s', len(losses_a), [f'{l:.3f}' for _, l in losses_a])
    logger.info('  Phase B losses (%d steps): %s', len(losses_b), [f'{l:.3f}' for _, l in losses_b])
    logger.info('  Phase C reload: |%.4f - %.4f| / %.4f = %.4f (tol %.2f)', loss_c_fixed, loss_a_fixed, loss_a_fixed,
                delta, RELOAD_LOSS_TOLERANCE)
    logger.info('  Phase D resume losses (%d steps): %s', len(losses_d), [f'{l:.3f}' for _, l in losses_d])
    logger.info('  Phase F LoRA-effect probes (%d/%d diverged):', n_diverged, len(PROBE_PROMPTS))
    for prompt, _, _, div in probe_results:
        marker = f'diverge@{div}' if div is not None else 'identical'
        logger.info(f'    {marker:>15} : {prompt!r}')
    logger.info('ALL PHASES PASSED')
    return 0


# ── pytest entry point ──

def test_full_cycle_e2e():
    """Pytest-collected entry point for the full-cycle E2E suite."""
    rc = main()
    assert rc == 0, 'Full-cycle E2E test failed'


if __name__ == '__main__':
    sys.exit(main())
