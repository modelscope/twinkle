"""HotpotQA GRPO baseline — full context, no chunking, no compression, no tools.

This is the **control group** for ``grpo_condensed.py``. Both scripts share:
  * dataset (HotpotQA fullwiki, hard split)
  * preprocessing (``HotpotQAProcessor`` with ``[K] Title: ...`` passages)
  * GRPO infra (model / sampler / device mesh / hyperparams)
  * rollout class (``MultiTurnRollout`` from ``multi_turn.py``)

The only differences are intentional:
  * no ``NativeChunker`` / ``ModelCondenser`` (full passages go in verbatim)
  * no tools registered (``ToolManager()`` is empty)
  * ``max_turns=1`` so the rollout is effectively single-turn
  * simplified system prompt (no ``<block_N>`` / ``extract_condensed`` syntax)
  * ``F1Reward + CoTReward`` only (no ``ToolExploreReward``)
  * traces → ``rollout_trace_baseline.jsonl``
  * checkpoints prefixed ``hotpotqa-grpo-baseline-*``

Keeping the same ``MultiTurnRollout`` code path on both sides means any
training-loop-level discrepancy between the two runs is attributable to
the chunk+condense pipeline, not to differences in rollout plumbing.
"""

import os
import re
from typing import Any, Dict, List, Optional

import swanlab
from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import Message, SamplingParams, Trajectory
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.metric import CompletionRewardMetric
from twinkle.model import TransformersModel
from twinkle.preprocessor.base import Preprocessor
from twinkle.processor import InputProcessor
from twinkle.sampler import vLLMSampler
from twinkle.template import Qwen3_5Template
from twinkle_agentic.reward import F1Reward, CoTReward
from twinkle_agentic.rollout.multi_turn import MultiTurnRollout
from twinkle_agentic.tools.tool_manager import ToolManager

logger = get_logger()

MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen3.5-4B')
USE_MEGATRON = bool(int(os.environ.get('USE_MEGATRON', '1')))

MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 4))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 4))
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', 8))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 4096))
LEARNING_RATE = float(os.environ.get('LR', 1e-5))
NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', 10))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 0))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 8))
MINI_BATCH_SIZE = int(os.environ.get('MINI_BATCH_SIZE', 8))
MICRO_BATCH_SIZE = int(os.environ.get('MICRO_BATCH_SIZE', 2))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 1))
ADAPTER_NAME = 'default'
SAVE_STEPS = int(os.environ.get('SAVE_STEPS', 1000))
LORA_RANK = int(os.environ.get('LORA_RANK', 16))

# Single-turn baseline; tools are not registered, but we keep MultiTurnRollout
# to share the rollout code path with the condensed variant. ``max_turns=1``
# guarantees the loop runs exactly one sampling pass per trajectory.
MAX_TURNS = int(os.environ.get('MAX_TURNS', 1))

HOTPOTQA_NUM_PROC = int(os.environ.get('HOTPOTQA_NUM_PROC', 16))
HOTPOTQA_MAX_LENGTH = int(os.environ.get('HOTPOTQA_MAX_LENGTH', 64000))

F1_REWARD_WEIGHT = float(os.environ.get('F1_REWARD_WEIGHT', 1.0))
COT_REWARD_WEIGHT = float(os.environ.get('COT_REWARD_WEIGHT', 0.2))

WRONG_IDS_FILE = os.environ.get('WRONG_IDS_FILE', '')

_ROLLOUT_TRACE_DIR = os.environ.get(
    'ROLLOUT_TRACE_BASELINE_DIR', 'rollout_trace_baseline')

SYSTEM_PROMPT = """You are a careful multi-hop QA assistant.

You will receive a question and a set of supporting passages. Each passage \
is shown inline as plain text in the form `[K] Title: ...`, where `K` is the \
passage index. All passages are already complete — there is no extraction \
or expansion step.

## Workflow

Step 1: Read every passage and identify which ones are relevant to the question.
Step 2: Reason step by step, citing the passage indices you used.
   Step N:   From passage [K], I learn that [fact A].
   Step N+1: From passage [M], I learn that [fact B].
   Step N+2: Combining these, the answer is ...
Step 3: Emit the final answer in `\\boxed{...}`.

Only answer when you are confident in the supporting facts.

## Output Format
End your final response with \\boxed{answer}, e.g. \\boxed{Delhi}.
Keep the boxed text short: a name, entity, date, or "yes"/"no".
Answers not inside \\boxed{} will not be scored."""


_F1_REWARD: Optional[F1Reward] = F1Reward()
_COT_REWARD: Optional[CoTReward] = CoTReward()


def compute_rewards(trajectories: List[Dict[str, Any]]):
    f1 = _F1_REWARD(trajectories)
    cot = _COT_REWARD(trajectories)
    total = [
        F1_REWARD_WEIGHT * a + COT_REWARD_WEIGHT * c
        for a, c in zip(f1, cot)
    ]
    return total, f1, cot


class HotpotQAProcessor(Preprocessor):
    """Same processor as ``grpo_condensed.py`` — passages are emitted as
    ``[K] Title: ...`` lines. The downstream is what differs: the baseline
    feeds the full context straight to the model (no ``<block_N>`` wrapping,
    no chunking, no condensation)."""

    def __init__(self, system: str = SYSTEM_PROMPT, levels=None):
        self.system = system
        self.levels = levels

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = [r for r in rows if r is not None]
        rows = self.map_row_to_col(rows)
        return rows

    @staticmethod
    def _format_context(context: Dict[str, Any]) -> str:
        titles = context.get('title', []) or []
        sentences = context.get('sentences', []) or []
        lines = []
        for i, (title, sents) in enumerate(zip(titles, sentences), start=1):
            if isinstance(sents, list):
                body = ' '.join(s.strip() for s in sents if s and s.strip())
            else:
                body = str(sents).strip()
            lines.append(f'[{i}] {title}: {body}')
        return '\n\n'.join(lines)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Trajectory]:
        if self.levels is not None and (row.get('level') or '').strip().lower() not in self.levels:
            return None
        question = row['question']
        answer = row.get('answer', '') or ''
        context_block = self._format_context(row.get('context', {}) or {})
        user_msg = f'Question: {question}\n\nContext:\n\n{context_block}'
        messages = [
            Message(role='system', content=self.system),
            Message(role='user', content=user_msg),
        ]
        return Trajectory(messages=messages, user_data=[('ground_truth', answer.strip())])


def create_hotpotqa_dataset() -> Dataset:
    dataset = Dataset()
    dataset.add_dataset(DatasetMeta(
        'hf://hotpotqa/hotpot_qa', subset_name='fullwiki', split='train'))

    _wrong_ids_path = WRONG_IDS_FILE.strip()
    if _wrong_ids_path:
        with open(_wrong_ids_path, 'r', encoding='utf-8') as fh:
            _ids = frozenset(ln.strip() for ln in fh if ln.strip())
        if _ids:
            _key = next(iter(dataset.datasets.keys()))
            _before = len(dataset.datasets[_key])
            dataset.datasets[_key] = dataset.datasets[_key].filter(
                lambda row: row.get('id') in _ids)
            dataset.dataset = dataset.datasets[_key]
            logger.info(f'[WRONG_IDS_FILE] {_wrong_ids_path}: {_before} -> {len(dataset.dataset)} rows')

    dataset.set_template(
        'Qwen3_5Template', model_id=MODEL_ID, max_length=HOTPOTQA_MAX_LENGTH,
        truncation_strategy='delete', enable_thinking=False)
    _HOTPOTQA_COLS = ['id', 'question', 'answer', 'type', 'level',
                      'supporting_facts', 'context']
    dataset.map(HotpotQAProcessor(system=SYSTEM_PROMPT, levels=['hard']),
                remove_columns=_HOTPOTQA_COLS)
    return dataset


# Matches a LaTeX ``\boxed{...}`` final-answer marker — used to flag
# rollouts that never committed an answer. Brace-balanced is overkill for
# a logging heuristic; a non-greedy ``[^}]*`` is good enough.
_BOXED_RE = re.compile(r'\\boxed\{[^}]*\}')

# Pulls the leading number out of pre-formatted metric strings such as
# ``'0.03 iters/s'`` / ``'1.000000e-05'`` / ``'30 seconds'`` emitted by
# ``TrainMetric`` and ``GRPOMetric``. We use this in ``_coerce_for_swanlab``
# so swanlab can build line charts instead of dropping those keys with a
# ``failed to create chart for key '...': invalid value type`` warning.
_LEADING_NUMBER_RE = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')


def _coerce_for_swanlab(log_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Cast string-valued metrics to float for swanlab line charts.

    ``TrainMetric.calculate()`` and ``GRPOMetric.calculate()`` return
    pre-formatted strings (``'0.03 iters/s'``, ``'1.000000e-05'``,
    ``'30 seconds'``, ``'0.8321'``). swanlab cannot build a line chart
    from a string value and emits one warning per key per step. We extract
    the leading number where possible; keys whose value can't be parsed
    as a scalar are left as-is so they still show up in the text log.
    """
    coerced: Dict[str, Any] = {}
    for k, v in log_dict.items():
        if isinstance(v, bool) or isinstance(v, (int, float)):
            coerced[k] = v
            continue
        if isinstance(v, str):
            m = _LEADING_NUMBER_RE.search(v)
            if m:
                try:
                    coerced[k] = float(m.group())
                    continue
                except ValueError:
                    pass
        coerced[k] = v
    return coerced


def _last_assistant_text(trajectory: Dict[str, Any]) -> Optional[str]:
    """Return the text of the last ``assistant`` message, or ``None``.

    ``content`` can be ``str`` | ``None`` | ``dict`` (single multimodal
    part) | ``list[dict]`` (multiple parts). The downstream caller feeds
    this into ``_BOXED_RE.search(...)``, so we collapse the visible text
    into a single string and ignore non-text parts (images etc.).
    """
    for m in reversed(trajectory.get('messages', [])):
        if m.get('role') != 'assistant':
            continue
        c = m.get('content')
        if c is None:
            return None
        if isinstance(c, str):
            return c
        if isinstance(c, dict):
            return c.get('text') if c.get('type') == 'text' else None
        if isinstance(c, list):
            parts = [p.get('text') or '' for p in c
                     if isinstance(p, dict) and p.get('type') == 'text']
            return '\n'.join(parts) if parts else None
        return str(c)
    return None


def _compute_rollout_diagnostics(
    trajectories: List[Dict[str, Any]],
    n_turns_per_rollout: List[int],
    per_rollout_completion_length: List[int],
) -> Dict[str, float]:
    """Aggregate rollout diagnostics for swanlab logging.

    Stripped-down version of the condensed variant's diagnostics — without
    chunking we only care about (a) the longest non-trainable prefix
    (system prompt + full passages), and (b) whether the rollout produced
    a `\\boxed{}` final answer at all. ``avg_turns`` is logged for symmetry
    even though it should be exactly 1.0 with ``MAX_TURNS=1``.
    """
    out: Dict[str, float] = {}
    if n_turns_per_rollout:
        out['avg_turns'] = sum(n_turns_per_rollout) / len(n_turns_per_rollout)

    _max_non_trainable = 0
    for t, comp_len in zip(trajectories, per_rollout_completion_length):
        ids = t.get('input_ids') or []
        non_trainable = max(0, len(ids) - int(comp_len or 0))
        if non_trainable > _max_non_trainable:
            _max_non_trainable = non_trainable
    out['non_trainable_tokens'] = _max_non_trainable

    if trajectories:
        n_no_boxed = sum(
            0 if _BOXED_RE.search(_last_assistant_text(t) or '') else 1
            for t in trajectories)
        out['no_boxed_rate'] = n_no_boxed / len(trajectories)

        def _content_chars(c: Any) -> int:
            if not c:
                return 0
            if isinstance(c, str):
                return len(c)
            if isinstance(c, dict):
                if c.get('type') == 'text':
                    return len(c.get('text') or '')
                return 0
            if isinstance(c, list):
                total = 0
                for part in c:
                    if isinstance(part, dict) and part.get('type') == 'text':
                        total += len(part.get('text') or '')
                    elif isinstance(part, str):
                        total += len(part)
                return total
            # Unknown shape -- fall back to ``str()`` length rather than
            # crashing, so a template quirk never breaks metric logging.
            return len(str(c))

        msg_chars_total, prompt_chars, asst_chars = [], [], []
        for t in trajectories:
            total_i = prompt_i = asst_i = 0
            for m in (t.get('messages') or []):
                role = m.get('role')
                if role == 'system':
                    continue
                n = _content_chars(m.get('content'))
                total_i += n
                if role in ('user', 'tool'):
                    prompt_i += n
                elif role == 'assistant':
                    asst_i += n
            msg_chars_total.append(total_i)
            prompt_chars.append(prompt_i)
            asst_chars.append(asst_i)
        out['avg_chars_total_no_sys'] = sum(msg_chars_total) / len(msg_chars_total)
        out['avg_chars_prompt_no_sys'] = sum(prompt_chars) / len(prompt_chars)
        out['avg_chars_assistant'] = sum(asst_chars) / len(asst_chars)
    return out


def main():
    swanlab.init(project='twinkle')

    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
        DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)), device_type='GPU'),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS,
                       groups=device_groups, lazy_collect=False)

    logger.info('Building HotpotQA dataset (baseline, full context)')
    _prebuilt_dataset = create_hotpotqa_dataset()
    logger.info('Dataset ready: %d rows', len(_prebuilt_dataset))

    GLOBAL_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    batches_per_epoch = max(1, len(_prebuilt_dataset) // GLOBAL_BATCH_SIZE)
    # Single-turn baseline: every rollout produces exactly one assistant
    # turn, so the per-batch optim-step count equals
    #   ceil(GLOBAL_BATCH_SIZE * NUM_GENERATIONS / MINI_BATCH_SIZE).
    optim_steps_per_batch = max(1, (GLOBAL_BATCH_SIZE * NUM_GENERATIONS
                                     + MINI_BATCH_SIZE - 1) // MINI_BATCH_SIZE)
    steps_per_epoch = batches_per_epoch * optim_steps_per_batch
    derived_total_steps = NUM_EPOCHS * steps_per_epoch
    total_steps = min(MAX_STEPS, derived_total_steps) if MAX_STEPS > 0 else derived_total_steps
    logger.info('Training horizon: %d steps (%d epochs × %d batches × %d steps/batch)',
                total_steps, NUM_EPOCHS, batches_per_epoch, optim_steps_per_batch)

    lora_config = LoraConfig(
        target_modules='all-linear', r=LORA_RANK,
        lora_alpha=LORA_RANK * 2, lora_dropout=0.05)

    if USE_MEGATRON:
        from twinkle.model.megatron import MegatronModel
        model = MegatronModel(
            model_id=MODEL_ID, device_mesh=model_mesh, remote_group='model',
            mixed_precision='bf16', variable_seq_lengths=True)
    else:
        model = TransformersModel(
            model_id=MODEL_ID, device_mesh=model_mesh, remote_group='model')

    model.add_adapter_to_model(ADAPTER_NAME, lora_config,
                               gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    if USE_MEGATRON:
        model.set_optimizer('default', lr=LEARNING_RATE)
        model.set_lr_scheduler('default', lr_decay_steps=total_steps, max_lr=LEARNING_RATE)
    else:
        model.set_optimizer('AdamW', lr=LEARNING_RATE)
        model.set_lr_scheduler('CosineAnnealingLR', T_max=total_steps, eta_min=0)

    model.set_loss('GRPOLoss', epsilon=0.2)
    model.set_processor(InputProcessor, padding_free=True)
    model.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False, max_length=HOTPOTQA_MAX_LENGTH)

    model.add_metric('GRPOMetric', is_training=True)

    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8, 'max_model_len': 32768,
            'max_lora_rank': 32, 'enable_lora': True,
            'enable_tower_connector_lora': True,
        },
        device_mesh=sampler_mesh, remote_group='sampler')
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False, max_length=HOTPOTQA_MAX_LENGTH)
    rollout_template = Qwen3_5Template(
        MODEL_ID, max_length=HOTPOTQA_MAX_LENGTH, enable_thinking=False)

    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)

    dataloader = DataLoader(
        dataset=lambda: _prebuilt_dataset,
        batch_size=GLOBAL_BATCH_SIZE, min_batch_size=GLOBAL_BATCH_SIZE)

    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1,
        temperature=1.0, top_p=0.95)

    def _trace_should_store(traj):
        return _F1_REWARD([traj])[0] == 0.0

    def _trace_is_success(traj):
        return _F1_REWARD([traj])[0] > 0.0

    rollout = MultiTurnRollout(
        sampler=sampler,
        template=rollout_template,
        tool_manager=ToolManager(),
        sampling_params=sampling_params,
        max_turns=MAX_TURNS,
        trace_dir=_ROLLOUT_TRACE_DIR or None,
        trace_callback=_trace_should_store,
        success_callback=_trace_is_success,
    )

    optim_step = 0
    logger.info('Starting HotpotQA GRPO baseline (no chunk / no condense / no tools)')

    def _epoch_cycle(dl, n_epochs):
        for ep in range(1, n_epochs + 1):
            logger.info(f'=== Epoch {ep}/{n_epochs} (step={optim_step}/{total_steps}) ===')
            for batch in dl:
                yield batch

    for batch in _epoch_cycle(dataloader, NUM_EPOCHS):
        if optim_step >= total_steps:
            break

        metrics.reset()
        expand_prompts = [p for prompt in batch for p in [prompt] * NUM_GENERATIONS]

        ckpt_manager.sync_weights(merge_and_sync=False)
        sampler.reset_prefix_cache()

        # Single batched rollout: each trajectory produces exactly one
        # assistant turn (tools are unregistered, ``max_turns=1``).
        all_trajectories: List[Dict[str, Any]] = rollout(expand_prompts)
        n_turns_per_rollout = [int(t.get('turns') or 0) for t in all_trajectories]
        per_rollout_completion_length = [
            sum(1 for l in (t.get('labels') or []) if l != -100)
            for t in all_trajectories]

        total_rewards, f1_rewards, cot_rewards = compute_rewards(all_trajectories)

        metrics.accumulate(
            completion_lengths=per_rollout_completion_length,
            rewards={'total': total_rewards, 'f1': f1_rewards, 'cot': cot_rewards})

        rollout_advantages = advantage_fn(
            total_rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()

        all_input_data: List[Any] = []
        all_old_logps: List[List[float]] = []
        advantages: List[float] = []
        for t, adv in zip(all_trajectories, rollout_advantages):
            all_input_data.append(t)
            all_old_logps.append([lp[0][1] for lp in (t.get('logprobs') or [])])
            advantages.append(adv)

        total_completions = len(all_input_data)
        aligned_completions = (total_completions // MODEL_GPUS) * MODEL_GPUS
        if aligned_completions < total_completions:
            logger.info(
                '[dp-align] dropping %d tail sample(s): total=%d -> aligned=%d (dp=%d)',
                total_completions - aligned_completions,
                total_completions, aligned_completions, MODEL_GPUS)
        for mb_start in range(0, aligned_completions, MINI_BATCH_SIZE):
            mb_end = min(mb_start + MINI_BATCH_SIZE, aligned_completions)
            model.forward_backward(
                inputs=all_input_data[mb_start:mb_end],
                old_logps=all_old_logps[mb_start:mb_end],
                advantages=advantages[mb_start:mb_end],
                micro_batch_size=MICRO_BATCH_SIZE)
            model.clip_grad_and_step()
            optim_step += 1
            if optim_step >= total_steps:
                break
            if optim_step % SAVE_STEPS == 0:
                model.save(f'hotpotqa-grpo-baseline-checkpoint-{optim_step}')

        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric(is_training=True))
        log_dict.update(_compute_rollout_diagnostics(
            all_trajectories, n_turns_per_rollout, per_rollout_completion_length))
        swanlab.log(_coerce_for_swanlab(log_dict))
        metrics.reset()
        logger.info(f'[Step {optim_step}/{total_steps}] {log_dict}')

    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('hotpotqa-grpo-baseline-final')


if __name__ == '__main__':
    main()
