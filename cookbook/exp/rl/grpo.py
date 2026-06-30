"""Pure GRPO training on AoPS dataset (no RAG, ablation baseline).

Architecture (8 GPUs):
  - 4 GPUs: sampler/rollout (vLLM TP=4)
  - 4 GPUs: training model (FSDP)

Pipeline per step:
  1. DataLoader yields a batch of math problems
  2. Sampler generates rollouts
  3. Reward (accuracy + format + gibberish) → GRPO advantage → model update

Launch:
    python cookbook/exp/rl/grpo.py
"""
import json
import os
import re
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.metric import CompletionRewardMetric
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.reward.base import Reward
from twinkle.sampler import vLLMSampler
from twinkle.template import Qwen3_5Template

logger = get_logger()

# ============================================================================
# Configuration
# ============================================================================
MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen3.5-4B')

# GPU layout: 4 rollout + 4 train = 8
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 4))
MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 4))
NUM_GPUS = SAMPLER_GPUS + MODEL_GPUS

# Training hyperparams
NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', 8))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 32768))
LEARNING_RATE = float(os.environ.get('LR', 1e-5))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 5000))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 8))
MINI_BATCH_SIZE = int(os.environ.get('MINI_BATCH_SIZE', 8))
MICRO_BATCH_SIZE = int(os.environ.get('MICRO_BATCH_SIZE', 2))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 1))
SAVE_STEPS = int(os.environ.get('SAVE_STEPS', 100))
ADV_CLIP = float(os.environ.get('ADV_CLIP', 2.0))

# Dataset
AOPS_DATASET_ID = os.environ.get('AOPS_DATASET_ID', 'AI-MO/aops')
AOPS_SEED = int(os.environ.get('AOPS_SEED', 100))

# Output / diagnostics
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', './outputs/grpo')

# System prompt
SYSTEM_PROMPT = (
    'You are an expert competition mathematician. '
    'Solve the problem step by step. Put your final answer inside \\boxed{}. '
    'For multiple-choice questions, put the option LETTER (A/B/C/D/E) inside \\boxed{}.'
)


# ============================================================================
# Reward
# ============================================================================
class AoPSAccuracyReward(Reward):
    """Accuracy reward via boxed answer extraction + robust equivalence matching."""

    @staticmethod
    def extract_boxed(text: str) -> str:
        idx = text.rfind('\\boxed{')
        if idx == -1:
            return ''
        start = idx + len('\\boxed{')
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1
        if depth == 0:
            return text[start:j - 1].strip()
        return ''

    _MCQ_GT_RE = re.compile(
        r'^\\?(?:textbf|mathbf|text|mathrm)\{?\(?([A-E])[)}\s\\]*(.*)',
        re.DOTALL)
    _MCQ_PAREN_RE = re.compile(r'^\(?([A-E])\)?[\s\\]+(.*)', re.DOTALL)
    _MCQ_SINGLE_LETTER_RE = re.compile(r'^[A-E]$')
    _VAR_PREFIX_RE = re.compile(r'^[a-zA-Z](?:\([^)]*\))?\s*=\s*(.+)', re.DOTALL)
    _EQ_RHS_RE = re.compile(r'^.+=\s*(.+)$')

    @staticmethod
    def normalize_answer(ans: str) -> str:
        if not ans:
            return ''
        s = ans.strip()
        m = re.match(r'^\\?(?:textbf|text|mathrm|mathbf)?\{?\(?([A-E])\)?\}?$', s)
        if m:
            return m.group(1)
        s = s.replace(' ', '')
        s = s.replace(r'\,', '')
        s = s.replace(r'\;', '')
        s = s.replace(r'\!', '')
        s = re.sub(r'\\(?:text|mathrm|mathbf|textbf|operatorname)\{([^}]*)\}', r'\1', s)
        s = s.replace(r'\displaystyle', '')
        s = re.sub(r'\\(?:left|right)[.()\[\]|]', '', s)
        s = s.replace(r'\dfrac', r'\frac')
        s = s.replace(r'\tfrac', r'\frac')
        s = s.strip('$').strip()
        s = re.sub(r'\{[a-zA-Z]+\}$', '', s)
        s = re.sub(r'\^\{\\circ\}|\^\\circ|°|\\circ', '', s)
        s = re.sub(r'\\(?:quad|qquad|\s)', '', s)
        s = s.replace(r'\minus{}', '-').replace(r'\minus', '-')

        def _frac_to_slash(m):
            text = m.group(0)
            pos = text.index('{') + 1
            depth, num_start = 1, pos
            while depth > 0:
                if text[pos] == '{': depth += 1
                elif text[pos] == '}': depth -= 1
                pos += 1
            numer = text[num_start:pos - 1]
            pos += 1
            den_start = pos
            depth = 1
            while depth > 0:
                if text[pos] == '{': depth += 1
                elif text[pos] == '}': depth -= 1
                pos += 1
            denom = text[den_start:pos - 1]
            return f'({numer})/({denom})'

        s = re.sub(
            r'\\frac\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
            _frac_to_slash, s)
        s = re.sub(r'(?<!\w)(\d+)/(\d+)(?!\w)', r'(\1)/(\2)', s)
        return s

    @classmethod
    def _strip_var_prefix(cls, s: str) -> str:
        m = cls._VAR_PREFIX_RE.match(s)
        return m.group(1).strip() if m else s

    @classmethod
    def _extract_mcq_parts(cls, s: str):
        m = cls._MCQ_GT_RE.match(s)
        if m:
            return m.group(1), m.group(2).strip()
        m = cls._MCQ_PAREN_RE.match(s)
        if m:
            return m.group(1), m.group(2).strip()
        m2 = re.search(r'\(?([A-E])\)?\s*$', s)
        if m2 and len(s) > 3:
            return m2.group(1), s[:m2.start()].strip()
        return None, None

    @staticmethod
    def _try_numeric_equal(a: str, b: str) -> bool:
        import math

        def _try_eval(s: str):
            try:
                return float(s.replace('(', '').replace(')', ''))
            except (ValueError, ZeroDivisionError):
                pass
            s_stripped = re.sub(r'[a-zA-Z]+$', '', s.replace('(', '').replace(')', '')).strip()
            if s_stripped and s_stripped != s:
                try:
                    return float(s_stripped)
                except (ValueError, ZeroDivisionError):
                    pass
            frac_re = re.compile(r'^\(([^)]+)\)/\(([^)]+)\)$')
            m = frac_re.match(s)
            if m:
                try:
                    return float(m.group(1)) / float(m.group(2))
                except (ValueError, ZeroDivisionError):
                    pass
            expr = s
            expr = expr.replace('\\pi', str(math.pi))
            expr = expr.replace('\\e', str(math.e))
            expr = re.sub(r'\\sqrt\{([^}]+)\}', r'(\1)**0.5', expr)
            expr = re.sub(r'\\sqrt\[3\]\{([^}]+)\}', r'(\1)**(1/3)', expr)
            expr = re.sub(r'\\sqrt\[([^]]+)\]\{([^}]+)\}', r'(\2)**(1/\1)', expr)
            expr = expr.replace('{', '(').replace('}', ')')
            expr = expr.replace('\\cdot', '*').replace('\\times', '*')
            expr = re.sub(r'(\d)\(', r'\1*(', expr)
            try:
                val = eval(expr, {"__builtins__": {}, "math": math, "pi": math.pi, "e": math.e}, {})
                return float(val)
            except Exception:
                pass
            return None

        va, vb = _try_eval(a), _try_eval(b)
        if va is not None and vb is not None:
            return abs(va - vb) < 1e-6 * max(1, abs(va), abs(vb))
        return False

    @classmethod
    def _normalize_tuple(cls, s: str) -> str:
        return re.sub(r'[\s()\[\]]', '', s)

    @classmethod
    def _try_sympy_equal(cls, a: str, b: str) -> bool:
        try:
            from sympy.parsing.latex import parse_latex
            from sympy import simplify, nsimplify
            expr_a = parse_latex(a)
            expr_b = parse_latex(b)
            diff = simplify(nsimplify(expr_a - expr_b))
            return diff == 0
        except Exception:
            return False

    @classmethod
    def answers_match(cls, predicted: str, reference: str) -> bool:
        if not predicted or not reference:
            return False

        norm_p = cls.normalize_answer(predicted)
        norm_r = cls.normalize_answer(reference)

        if norm_p == norm_r:
            return True
        if norm_p.lower() == norm_r.lower():
            return True
        if cls._try_numeric_equal(norm_p, norm_r):
            return True

        stripped_p = cls.normalize_answer(cls._strip_var_prefix(predicted))
        stripped_r = cls.normalize_answer(cls._strip_var_prefix(reference))
        if stripped_p and stripped_r and stripped_p == stripped_r:
            return True
        if stripped_p and stripped_r and cls._try_numeric_equal(stripped_p, stripped_r):
            return True

        ref_letter, ref_value = cls._extract_mcq_parts(reference)
        if ref_letter:
            if norm_p == ref_letter or predicted.strip().upper() == ref_letter:
                return True
            if ref_value:
                norm_ref_val = cls.normalize_answer(ref_value)
                if norm_p == norm_ref_val or cls._try_numeric_equal(norm_p, norm_ref_val):
                    return True
        pred_letter, pred_value = cls._extract_mcq_parts(predicted)
        if pred_letter:
            if norm_r == pred_letter or reference.strip().upper() == pred_letter:
                return True
            if pred_value:
                norm_pred_val = cls.normalize_answer(pred_value)
                if norm_r == norm_pred_val or cls._try_numeric_equal(norm_r, norm_pred_val):
                    return True
        if cls._MCQ_SINGLE_LETTER_RE.match(reference.strip()):
            if cls._MCQ_SINGLE_LETTER_RE.match(predicted.strip().upper()):
                return predicted.strip().upper() == reference.strip().upper()

        tuple_p = cls._normalize_tuple(norm_p)
        tuple_r = cls._normalize_tuple(norm_r)
        if ',' in tuple_p and tuple_p == tuple_r:
            return True

        if '=' in norm_r and '=' not in norm_p:
            parts = norm_r.split('=')
            for part in parts:
                part = part.strip()
                if part == norm_p or cls._try_numeric_equal(part, norm_p):
                    return True
        if '=' in norm_p and '=' not in norm_r:
            parts = norm_p.split('=')
            for part in parts:
                part = part.strip()
                if part == norm_r or cls._try_numeric_equal(part, norm_r):
                    return True
        if '=' in norm_p and '=' in norm_r:
            pp = [x.strip() for x in norm_p.split('=')]
            rp = [x.strip() for x in norm_r.split('=')]
            if set(pp) == set(rp):
                return True

        def _sort_factors(s):
            tokens = re.findall(r'\\?[a-zA-Z]+\{[^}]*\}|\\?[a-zA-Z]+|\d+|[^a-zA-Z\d\\{}]', s)
            return ''.join(sorted(tokens))
        if _sort_factors(norm_p) == _sort_factors(norm_r):
            return True

        if cls._try_sympy_equal(predicted, reference):
            return True

        return False

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for traj in trajectories:
            messages = traj.get('messages', [])
            completion = ''
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '')
                    break
            user_data = traj.get('user_data') or []
            gt = ''
            for item in user_data:
                if item[0] == 'ground_truth':
                    gt = item[1]
                    break
            predicted = self.extract_boxed(completion)
            correct = self.answers_match(predicted, gt)
            rewards.append(1.0 if correct else 0.0)
        return rewards


class FormatReward(Reward):
    """Reward for having \\boxed{} in the output."""

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for traj in trajectories:
            messages = traj.get('messages', [])
            completion = ''
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '')
                    break
            has_boxed = '\\boxed{' in completion
            rewards.append(0.5 if has_boxed else 0.0)
        return rewards


class GibberishPenalty(Reward):
    """Negative reward for degenerate outputs (gibberish/random unicode tail)."""

    TAIL_CHARS = 400
    GIBBERISH_THRESHOLD = 0.20

    @classmethod
    def is_gibberish(cls, text: str) -> bool:
        if not text:
            return False
        tail = text[-cls.TAIL_CHARS:] if len(text) > cls.TAIL_CHARS else text
        non_math_non_ascii = 0
        for c in tail:
            code = ord(c)
            if code > 127 and not (0x4e00 <= code <= 0x9fff):
                non_math_non_ascii += 1
        return non_math_non_ascii > len(tail) * cls.GIBBERISH_THRESHOLD

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for traj in trajectories:
            messages = traj.get('messages', [])
            completion = ''
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '')
                    break
            rewards.append(-0.5 if self.is_gibberish(completion) else 0.0)
        return rewards


def compute_rewards(trajectories: List[Dict[str, Any]]
                    ) -> Tuple[List[float], List[float], List[float]]:
    acc_fn = AoPSAccuracyReward()
    fmt_fn = FormatReward()
    gib_fn = GibberishPenalty()
    acc = acc_fn(trajectories)
    fmt = fmt_fn(trajectories)
    gib = gib_fn(trajectories)
    total = [a + f + g for a, f, g in zip(acc, fmt, gib)]
    return total, fmt, acc


# ============================================================================
# Dataset: AoPS boxed problems
# ============================================================================
def create_aops_dataset():
    """Load AoPS and create GRPO-style dataset (prompt only, with ground_truth in user_data)."""
    from modelscope import MsDataset
    from twinkle.data_format import Message, Trajectory

    ds = MsDataset.load(AOPS_DATASET_ID, split='train',
                        download_mode='reuse_dataset_if_exists')
    rows = []
    for row in ds:
        if not row['metadata'].get('boxed'):
            continue
        ref = AoPSAccuracyReward.extract_boxed(row['solution'])
        if not ref:
            continue
        rows.append({'problem': row['problem'], 'ground_truth': ref})

    logger.info(f'[aops] loaded {len(rows)} boxed problems')
    rng = random.Random(AOPS_SEED)
    rng.shuffle(rows)

    trajectories = []
    for r in rows:
        traj = Trajectory(
            messages=[
                Message(role='system', content=SYSTEM_PROMPT),
                Message(role='user', content=r['problem']),
            ],
            user_data=[('ground_truth', r['ground_truth'])],
        )
        trajectories.append(traj)

    data_meta = DatasetMeta(data=trajectories)
    dataset = Dataset(data_meta)
    dataset.set_template('Qwen3_5Template', model_id=MODEL_ID,
                         max_length=16384, truncation_strategy='delete',
                         enable_thinking=True)
    dataset.encode(add_generation_prompt=True)
    return dataset


# ============================================================================
# Main
# ============================================================================
def main():
    sampler_start = 0
    model_start = sampler_start + SAMPLER_GPUS

    device_groups = [
        DeviceGroup(name='sampler', ranks=list(range(sampler_start, model_start)),
                    device_type='GPU', gpus_per_worker=SAMPLER_GPUS),
        DeviceGroup(name='model', ranks=list(range(model_start, NUM_GPUS)),
                    device_type='GPU'),
    ]

    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, fsdp_size=MODEL_GPUS, ulysses_size=2)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, tp_size=SAMPLER_GPUS)

    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS,
                       groups=device_groups, lazy_collect=False)

    # -- Training model (full-parameter) --
    model = TransformersModel(
        model_id=MODEL_ID, device_mesh=model_mesh, remote_group='model')
    model.set_optimizer('AdamW', lr=LEARNING_RATE)
    model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)
    model.set_loss('GSPOLoss', epsilon=0.2, epsilon_high=0.28, beta=0.0)
    model.set_processor(InputProcessor)
    model.set_template('Qwen3_5Template', model_id=MODEL_ID,
                       enable_thinking=True, max_length=32768)

    # -- Rollout sampler --
    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': 32768,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID,
                         enable_thinking=True, max_length=32768)

    # -- Checkpoint & DataLoader --
    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)

    GLOBAL_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    dataloader = DataLoader(
        dataset=create_aops_dataset,
        batch_size=GLOBAL_BATCH_SIZE,
        min_batch_size=GLOBAL_BATCH_SIZE,
        device_mesh=model_mesh,
        remote_group='model',
    )

    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1,
        temperature=1.0, top_p=0.95)

    optim_step = 0
    logger.info('Starting pure GRPO training (no RAG)')
    logger.info(get_device_placement())

    # -- Diagnostics --
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    diag_path = os.path.join(OUTPUT_DIR, 'diagnostics.jsonl')
    diag_f = open(diag_path, 'a', encoding='utf-8')
    logger.info(f'[diag] diagnostics → {diag_path}')

    def _content_to_str(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return ''.join(
                b.get('text', '') if isinstance(b, dict) else str(b)
                for b in content)
        return str(content)

    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break

        metrics.reset()

        # Build prompts (direct, no RAG)
        prompts = []
        for item in batch:
            msgs = item.get('messages', [])
            prob = ''
            for m in msgs:
                if m.get('role') == 'user':
                    prob = m.get('content', '')
                    if isinstance(prob, list):
                        prob = ''.join(p.get('text', '') for p in prob if isinstance(p, dict))
                    break
            ud = item.get('user_data', [])
            gt = ''
            for pair in ud:
                if pair[0] == 'ground_truth':
                    gt = pair[1]
                    break
            prompts.append({
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': prob},
                ],
                'user_data': [('ground_truth', gt)],
            })

        # Expand for NUM_GENERATIONS and sample
        expand_prompts = []
        for prompt in prompts:
            expand_prompts.extend([prompt] * NUM_GENERATIONS)

        ckpt_manager.sync_weights(merge_and_sync=False)
        sampler.reset_prefix_cache()

        sample_responses = sampler.sample(expand_prompts, sampling_params)

        # Collect rollouts
        all_input_data: List[Dict[str, Any]] = []
        all_old_logps: List[List[float]] = []
        all_completion_lengths: List[int] = []

        for sample_response in sample_responses:
            for sequence in sample_response.sequences:
                all_input_data.append(sequence.new_input_feature)
                all_old_logps.append([logprob[0][1] for logprob in sequence.logprobs])
                all_completion_lengths.append(len(sequence.tokens))

        # Rewards
        total_rewards, format_rewards, accuracy_rewards = compute_rewards(all_input_data)

        # Zero out rewards for rollouts that hit the max_tokens ceiling
        max_len_threshold = int(MAX_NEW_TOKENS * 0.95)
        for i in range(len(all_input_data)):
            if all_completion_lengths[i] >= max_len_threshold:
                total_rewards[i] = 0.0
                accuracy_rewards[i] = 0.0
                format_rewards[i] = 0.0

        # Per-step reward summary
        n_correct = sum(1 for a in accuracy_rewards if a > 0)
        diag_f.write(json.dumps({
            'step': optim_step, 'type': 'reward_summary',
            'n_samples': len(accuracy_rewards),
            'accuracy': n_correct / len(accuracy_rewards) if accuracy_rewards else 0,
            'mean_reward': sum(total_rewards) / len(total_rewards) if total_rewards else 0,
        }, ensure_ascii=False) + '\n')

        metrics.accumulate(
            completion_lengths=all_completion_lengths,
            rewards={
                'total': total_rewards,
                'format': format_rewards,
                'accuracy': accuracy_rewards,
            },
        )

        # GRPO advantage
        advantages = advantage_fn(
            total_rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()
        if ADV_CLIP > 0:
            advantages = [max(-ADV_CLIP, min(ADV_CLIP, a)) for a in advantages]

        # Log rollout responses
        _extract_boxed = AoPSAccuracyReward.extract_boxed
        for ridx, traj in enumerate(all_input_data):
            msgs = traj.get('messages', [])
            assistant_text = _content_to_str(next(
                (m['content'] for m in reversed(msgs) if m.get('role') == 'assistant'), ''))
            user_text = _content_to_str(next(
                (m['content'] for m in msgs if m.get('role') == 'user'), ''))
            user_data = traj.get('user_data') or []
            gt = next((v for k, v in user_data if k == 'ground_truth'), '')
            problem_idx = ridx // NUM_GENERATIONS
            grp_start = problem_idx * NUM_GENERATIONS
            grp_end = grp_start + NUM_GENERATIONS
            grp_acc = sum(accuracy_rewards[grp_start:grp_end]) / NUM_GENERATIONS

            diag_f.write(json.dumps({
                'step': optim_step, 'type': 'rollout',
                'idx': ridx,
                'problem_idx': problem_idx,
                'problem': user_text,
                'response': assistant_text,
                'ground_truth': gt,
                'predicted': _extract_boxed(assistant_text),
                'reward': total_rewards[ridx],
                'accuracy_reward': accuracy_rewards[ridx],
                'format_reward': format_rewards[ridx],
                'advantage': advantages[ridx],
                'completion_length': all_completion_lengths[ridx],
                'group_accuracy': grp_acc,
            }, ensure_ascii=False) + '\n')

        diag_f.flush()

        # Filter out all-same reward problem groups (no gradient signal)
        filtered_inputs, filtered_old_logps, filtered_advantages = [], [], []
        for g in range(BATCH_SIZE):
            g_start = g * NUM_GENERATIONS
            g_end = g_start + NUM_GENERATIONS
            grp_adv = advantages[g_start:g_end]
            if all(abs(a) < 1e-8 for a in grp_adv):
                continue
            filtered_inputs.extend(all_input_data[g_start:g_end])
            filtered_old_logps.extend(all_old_logps[g_start:g_end])
            filtered_advantages.extend(grp_adv)

        # Mini-batch training with gradient accumulation
        # Process MICRO_BATCH_SIZE samples per forward, accumulate grad_accum_steps
        # times before one optimizer step. clip_grad_norm normalizes by accumulated
        # num_tokens, ensuring mathematical equivalence with larger batch forward.
        total_completions = len(filtered_inputs)
        if total_completions == 0:
            logger.info(f'[Step {optim_step}] all groups filtered (uniform rewards), skip training')
            continue

        grad_accum_steps = MINI_BATCH_SIZE // MICRO_BATCH_SIZE
        accum_count = 0
        for mb_start in range(0, total_completions, MICRO_BATCH_SIZE):
            mb_end = min(mb_start + MICRO_BATCH_SIZE, total_completions)
            mb_inputs = filtered_inputs[mb_start:mb_end]
            mb_old_logps = filtered_old_logps[mb_start:mb_end]
            mb_advantages = filtered_advantages[mb_start:mb_end]

            model.forward_backward(
                inputs=mb_inputs,
                old_logps=mb_old_logps,
                ref_logps=mb_old_logps,
                advantages=mb_advantages,
            )
            accum_count += 1

            if accum_count % grad_accum_steps == 0:
                model.clip_grad_and_step()
                optim_step += 1

                if optim_step >= MAX_STEPS:
                    break
                if optim_step % SAVE_STEPS == 0:
                    model.save(f'grpo-checkpoint-{optim_step}')

        # Flush remaining accumulated gradients (incomplete window at tail)
        if accum_count % grad_accum_steps != 0:
            model.clip_grad_and_step()
            optim_step += 1

        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric(is_training=True))
        metrics.reset()
        logger.info(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')

    diag_f.close()
    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('grpo-final')


if __name__ == '__main__':
    main()
