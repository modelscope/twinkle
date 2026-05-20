import copy
import math
import os
import re
from typing import Any, Dict, List, Optional

import torch
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
from twinkle_agentic.chunker.native import NativeChunker
from twinkle_agentic.condenser import ModelCondenser
from twinkle_agentic.reward import F1Reward, CoTReward, ToolExploreReward
from twinkle_agentic.rollout.multi_turn_condense import MultiTurnCondenseRollout
from twinkle_agentic.tools.tool_manager import ToolManager

logger = get_logger()

MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen3.5-4B')
USE_MEGATRON = bool(int(os.environ.get('USE_MEGATRON', '0')))

MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 4))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 4))
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', 8))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 4096))
LEARNING_RATE = float(os.environ.get('LR', 1e-5))
NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', 1))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 0))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 8))
MINI_BATCH_SIZE = int(os.environ.get('MINI_BATCH_SIZE', 8))
MICRO_BATCH_SIZE = int(os.environ.get('MICRO_BATCH_SIZE', 2))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 1))
ADAPTER_NAME = 'default'
SAVE_STEPS = int(os.environ.get('SAVE_STEPS', 1000))
LORA_RANK = int(os.environ.get('LORA_RANK', 16))

MAX_TURNS = int(os.environ.get('MAX_TURNS', 4))
MAX_TRAJECTORY_TOKENS = int(os.environ.get('MAX_TRAJECTORY_TOKENS', 8192))
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 1024))

HOTPOTQA_NUM_PROC = int(os.environ.get('HOTPOTQA_NUM_PROC', 16))
HOTPOTQA_MAX_LENGTH = int(os.environ.get('HOTPOTQA_MAX_LENGTH', 64000))

F1_REWARD_WEIGHT = float(os.environ.get('F1_REWARD_WEIGHT', 1.0))
COT_REWARD_WEIGHT = float(os.environ.get('COT_REWARD_WEIGHT', 0))
TOOL_BONUS_WEIGHT = float(os.environ.get('TOOL_BONUS_WEIGHT', 0.0))
TOOL_BONUS_F1_THRESHOLD = float(
    os.environ.get('TOOL_BONUS_F1_THRESHOLD', 0.5))

# KL penalty coefficient; 0 disables KL (and skips the ref forward pass entirely).
# CISPO is token-level and DOES support per-token KL — small positive value (e.g. 0.005) recommended as anchor.
KL_BETA = float(os.environ.get('KL_BETA', 0.01))

# Entropy bonus coefficient; 0 disables the entropy compute path entirely.
# Typical GRPO values: 0.001–0.01. Loss is: L = L_PPO + beta*KL - entropy_coef*H.
ENTROPY_COEF = float(os.environ.get('ENTROPY_COEF', 0.0))

# Per-token oracle bonus coefficient; 0 disables. Typical: 0.05–0.2.
# Loss becomes: L = L_PPO + beta*KL - entropy_coef*H - token_bonus_coef*(oracle_logps - rollout_logps)
ORACLE_BONUS_COEF = float(os.environ.get('ORACLE_BONUS_COEF', 0.0))

# CISPO token-level IS clamp thresholds (MiniMax CISPO defaults: 0.2 / 0.28 asymmetric).
CISPO_EPS_LOW = float(os.environ.get('CISPO_EPS_LOW', 0.2))
CISPO_EPS_HIGH = float(os.environ.get('CISPO_EPS_HIGH', 0.2))

# High-KL token capture: top-K per microbatch dumped into log_dict['_high_kl_records']. 0 = disabled.
HIGH_KL_TOPK = int(os.environ.get('HIGH_KL_TOPK', 0))

INIT_LORA_PATH = os.environ.get('INIT_LORA_PATH', 'output/condensed_sft_ddp/last-checkpoint')
DATASET_PATH = os.environ.get(
    'DATASET_PATH',
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'hotpotqa_fullwiki_reannotated_12k.jsonl'))
F1_BINARY_THRESHOLD = float(os.environ.get('F1_BINARY_THRESHOLD', 0.5))

_ROLLOUT_TRACE_DIR = os.environ.get('ROLLOUT_TRACE_DIR', 'rollout_trace')
ORACLE_HINT = bool(int(os.environ.get('ORACLE_HINT', '0')))


# [EXP-ORACLE] staged hint injection — appended to the Question line so skip_pattern keeps it uncompressed.
def _oracle_hint_stage(step: int, total_steps: int) -> int:
    """0 = explicit titles, 1 = vague count, 2 = no hint."""
    return 0
    # if total_steps <= 0:
    #     return 0
    # third = max(1, total_steps // 3)
    # if step < third:
    #     return 0
    # if step < 2 * third:
    #     return 1
    # return 2



def _make_oracle_hint_callback(total_steps: int):
    """Return a post_compress_callback that injects oracle hints with actual block IDs.

    Called by MultiTurnCondenseRollout after compression + metadata merge, so
    ``compressed['user_data']`` carries sf_titles and ``chunks`` carries the
    condensed/raw status of each passage.

    Stages (determined by global_step / total_steps):
      0 — explicit block IDs for supporting-fact passages
      1 — block count only (no IDs)
      2 — no hint
    """
    _q_split = re.compile(r'(Question:\s*.+?)(\n\nContext:)', re.DOTALL)

    def _callback(compressed, chunks, **kwargs):
        step = kwargs.get('global_step', 0)
        stage = _oracle_hint_stage(step, total_steps)
        if stage == 2:
            return compressed

        user_data = compressed.get('user_data') or []
        sf_titles = [v for k, v in user_data if k == 'sf_title' and v]
        if not sf_titles:
            return compressed
        sf_set = set(sf_titles)

        # Map sf_titles → block IDs by walking condensed chunks
        block_id = 0
        sf_block_ids = []
        for c in chunks.chunks:
            if c.get('type') != 'text':
                continue
            content = c.get('content')
            if not isinstance(content, str) or not content:
                continue
            if c.get('role') == 'tool':
                continue
            raw = c.get('raw')
            if not (isinstance(raw, dict) and raw.get('condensed')):
                continue
            block_id += 1
            original = raw.get('original', '')
            if isinstance(original, str):
                for title in sf_set:
                    if original.startswith(f'{title}: ') or original.startswith(f'{title}:'):
                        sf_block_ids.append(block_id)
                        break

        if stage == 0:
            if sf_block_ids:
                ids_str = ', '.join(str(b) for b in sf_block_ids)
                hint = (f'\n[Oracle Hint] Block {ids_str} contain(s) the supporting facts. '
                        'Call `extract_condensed` to expand them if you need more detail information.')
            else:
                n = len(sf_set)
                word = {1: 'One', 2: 'Two', 3: 'Three'}.get(n, str(n))
                hint = (f'\n[Oracle Hint] {word} short passage(s) contain the supporting facts; '
                        'they are uncompressed — read them directly.')
        else:
            hint = (f'\n[Oracle Hint] Some compressed block(s) contain the supporting facts; '
                    'call `extract_condensed` to expand them if you need more detail information.')

        for m in (compressed.get('messages') or []):
            if m.get('role') != 'user':
                continue
            c = m.get('content')
            if isinstance(c, str):
                m['content'] = _q_split.sub(
                    lambda g: g.group(1) + hint + g.group(2), c, count=1)
            elif isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get('type') == 'text':
                        part['text'] = _q_split.sub(
                            lambda g: g.group(1) + hint + g.group(2),
                            part.get('text') or '', count=1)
                        break
            break
        return compressed

    return _callback

SYSTEM_PROMPT = """You are a careful multi-hop QA assistant.

## Context Format (Mixed)
The context you receive is a **mix of two forms**:

1. **Compressed blocks** — long passages wrapped in `<block_N>...</block_N>`, \
   displayed as a Markdown digest in **telegraphic style** (no \
   articles / "is" / "are"; colons and commas mean "is" / "has") \
   with two sections:
   - **Summary**: overview plus facts strongly related to the question, stated explicitly.
   - **More**: a collapsed INDEX of category keywords hinting at extra details hidden in the full text (call `extract_condensed` to see them).
   Reading example: `India: 7th largest by area. Borders: Pakistan, \
   China.` means "India is the 7th largest country by area and \
   shares borders with Pakistan and China."
2. **Raw passages** — short passages shown inline as plain text (`Title: \
   body`) **without** any `<block_N>` wrapping. These are already the full \
   text; nothing is hidden.

Only the `<block_N>`-wrapped blocks are compressed and can be expanded. \
Block ids `N` are 1-based and assigned in the order compressed blocks \
appear in the context, so they are always contiguous (`<block_1>`, \
`<block_2>`, `<block_3>`, ...). Raw passages have no block id and cannot \
be extracted — they are already complete.

## Workflow

### Phase 1 — Scan and Decide
Step 1: Read each compressed block's Summary, and read raw \
passages directly, to get an overview.
Step 2: For compressed blocks, check the More keywords to judge whether \
hidden details are needed.
Step 3: Decide which compressed blocks to expand, then call \
`extract_condensed` with their block ids. Raw passages need no extraction.

### Phase 2 — Reason and Answer
After the tool returns the full text, continue stepping through the evidence:
Step N:   From block X (or the raw passage titled "..."), I learn that [fact A].
Step N+1: From block Y, I need to call `extract_condensed` to get more information, because this block is related to...
Step N+2: Combining these, the answer is ...
\\boxed{answer}

You may call `extract_condensed` several times to expand more blocks if the information is not enough, only answer the question if you are sure about the facts.
The `blocks` parameter accepts **exactly one integer** per call (e.g. `3`); lists are rejected. Expand additional blocks by issuing separate `extract_condensed` calls, one per block. Only pass ids that actually appear as `<block_N>` in the context, and do **not** request the same block twice — its text is already in the conversation after the first expansion.

## Tool Call Format
<tool_call>
<function=extract_condensed>
<parameter=blocks>
3
</parameter>
</function>
</tool_call>

## Output Format
End your final response with \\boxed{answer}, e.g. \\boxed{Delhi}.
Keep the boxed text short: a name, entity, date, or "yes"/"no".
Answers not inside \\boxed{} will not be scored."""


_F1_REWARD: Optional[F1Reward] = F1Reward()
_COT_REWARD: Optional[CoTReward] = CoTReward()
_TOOL_EXPLORE_REWARD: Optional[ToolExploreReward] = ToolExploreReward(
    f1_threshold=TOOL_BONUS_F1_THRESHOLD)


def compute_rewards(trajectories: List[Dict[str, Any]]):
    f1_raw = _F1_REWARD(trajectories)
    f1 = [1.0 if v >= F1_BINARY_THRESHOLD else 0.0 for v in f1_raw] if F1_BINARY_THRESHOLD > 0 else f1_raw
    cot = _COT_REWARD(trajectories)
    tool_explore = _TOOL_EXPLORE_REWARD(trajectories)
    total = [
        F1_REWARD_WEIGHT * a + COT_REWARD_WEIGHT * c + TOOL_BONUS_WEIGHT * te
        for a, c, te in zip(f1, cot, tool_explore)
    ]
    return total, f1, cot, tool_explore


class HotpotQAProcessor(Preprocessor):
    def __init__(self, system: str = SYSTEM_PROMPT):
        self.system = system

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
        for title, sents in zip(titles, sentences):
            if isinstance(sents, list):
                body = ' '.join(s.strip() for s in sents if s and s.strip())
            else:
                body = str(sents).strip()
            lines.append(f'{title}: {body}')
        return '\n\n'.join(lines)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Trajectory]:
        if (row.get('verdict') or '').strip().lower() == 'drop':
            return None
        question = row.get('question_fixed') or row['question']
        answers = row.get('answers')
        if isinstance(answers, list) and answers:
            gold = [str(a).strip() for a in answers if str(a).strip()]
        else:
            gold = [s for s in [(row.get('answer', '') or '').strip()] if s]
        context_block = self._format_context(row.get('context', {}) or {})
        user_msg = f'Question: {question}\n\nContext:\n\n{context_block}'
        messages = [
            Message(role='system', content=self.system),
            Message(role='user', content=user_msg),
        ]
        # [EXP-ORACLE] carry supporting_facts titles via user_data; rollout injects post-compression block hint
        sf = row.get('supporting_facts') or {}
        sf_titles = sf.get('title') or []
        sf_unique = list(dict.fromkeys(t for t in sf_titles if t))
        user_data = [('ground_truth', g) for g in gold] + [('sf_title', t) for t in sf_unique]
        return Trajectory(messages=messages, user_data=user_data)


def create_hotpotqa_dataset() -> Dataset:
    dataset = Dataset()
    dataset.add_dataset(DatasetMeta(DATASET_PATH))
    logger.info('[dataset] loaded %s: %d rows', DATASET_PATH, len(dataset))

    dataset.set_template(
        'Qwen3_5Template', model_id=MODEL_ID, max_length=HOTPOTQA_MAX_LENGTH,
        truncation_strategy='delete', enable_thinking=False)
    _HOTPOTQA_COLS = ['id', 'question', 'question_fixed', 'answers',
                      'original_answer', 'type', 'level', 'verdict',
                      'reasoning', 'supporting_facts', 'context']
    dataset.map(HotpotQAProcessor(system=SYSTEM_PROMPT), remove_columns=_HOTPOTQA_COLS)
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
    f1_rewards: Optional[List[float]] = None,
    old_logps: Optional[List[List[float]]] = None,
) -> Dict[str, float]:
    """Aggregate rollout diagnostics for swanlab logging.

    All inputs are already flat:
      * ``trajectories[i]`` is the merged trajectory dict returned by
        :class:`MultiTurnCondenseRollout` (contains ``messages``,
        ``input_ids``, ``labels``, ``turns`` at top level).
      * ``n_turns_per_rollout[i] == trajectories[i]['turns']``.
      * ``per_rollout_completion_length[i]`` == number of trainable
        tokens in the trajectory (labels != -100).
    """
    out: Dict[str, float] = {}
    if n_turns_per_rollout:
        out['avg_turns'] = sum(n_turns_per_rollout) / len(n_turns_per_rollout)

    # ``non_trainable_tokens`` is the longest non-trainable prefix across
    # the batch: ``len(input_ids) - sum(1 for l in labels if l != -100)``.
    # Tracks how much the condensed context + system prompt is eating the
    # context budget (it does NOT equal the first-turn prompt length
    # because multi-turn runs also contribute non-trainable tokens from
    # the ``tool`` observations between assistant turns).
    _max_non_trainable = 0
    for t, comp_len in zip(trajectories, per_rollout_completion_length):
        ids = t.get('input_ids') or []
        non_trainable = max(0, len(ids) - int(comp_len or 0))
        if non_trainable > _max_non_trainable:
            _max_non_trainable = non_trainable
    out['non_trainable_tokens'] = _max_non_trainable

    if trajectories:
        tool_counts = [
            sum(len(m.get('tool_calls') or [])
                for m in t.get('messages', []) if m.get('role') == 'assistant')
            for t in trajectories]
        out['avg_tool_calls'] = sum(tool_counts) / len(tool_counts)
        out['tool_use_rate'] = sum(1 for c in tool_counts if c > 0) / len(tool_counts)
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

    if f1_rewards is not None and old_logps is not None and f1_rewards:
        per_traj_mean = [
            (sum(lp) / len(lp)) if lp else 0.0 for lp in old_logps]
        pos_logp = [m for m, f1 in zip(per_traj_mean, f1_rewards) if f1 > 0]
        zero_logp = [m for m, f1 in zip(per_traj_mean, f1_rewards) if f1 <= 0]
        out['f1_correct_rate'] = len(pos_logp) / len(f1_rewards)
        out['f1_zero_rate'] = len(zero_logp) / len(f1_rewards)
        out['mean_old_logp_f1_pos'] = (sum(pos_logp) / len(pos_logp)) if pos_logp else 0.0
        out['mean_old_logp_f1_zero'] = (sum(zero_logp) / len(zero_logp)) if zero_logp else 0.0
        out['policy_confidence_f1_pos'] = math.exp(out['mean_old_logp_f1_pos'])
        out['policy_confidence_f1_zero'] = math.exp(out['mean_old_logp_f1_zero'])
    return out


def _build_oracle_inputs(
    mb_inputs: List[Dict[str, Any]],
    f1_labels: List[bool],
    template,
) -> Optional[List[Dict[str, Any]]]:
    """Build oracle-context inputs at the TOKEN level for per-token bonus computation.

    The approach:
      1. Find ``first_trainable`` from labels (first position != -100).
         Due to NTP shift, input_ids[first_trainable] is the last prefix token (e.g. \\n
         after ``assistant``) and labels[first_trainable] is the first response token target.
      2. Construct oracle messages: [system, user_with_oracle_suffix].
      3. Encode with template (add_generation_prompt=True) → oracle_prefix_ids ending with
         the same assistant header token.
      4. Concatenate: oracle_prefix_ids + input_ids[first_trainable+1:] (response tokens).
      5. Labels: [-100]*(len(oracle_prefix)-1) + labels[first_trainable:] so the last prefix
         position predicts the first response token.

    For F1=0 samples: copied unchanged (bonus zeroed by _compute_token_bonus).
    """
    _q_line_re = re.compile(r'Question:\s*(.+?)(?:\n|$)', re.DOTALL)
    oracle_inputs = []
    any_modified = False

    for inp, is_pos in zip(mb_inputs, f1_labels):
        if not is_pos:
            oracle_inputs.append(inp)
            continue

        user_data = inp.get('user_data') or []
        sf_titles = [v for k, v in user_data if k == 'sf_title' and v]
        gts = [v for k, v in user_data if k == 'ground_truth' and v]
        if not sf_titles and not gts:
            oracle_inputs.append(inp)
            continue

        labels = inp.get('labels') or []
        input_ids = inp.get('input_ids') or []
        if not labels or not input_ids:
            oracle_inputs.append(inp)
            continue

        # 1. Find first trainable position
        first_trainable = None
        for i, l in enumerate(labels):
            if l != -100:
                first_trainable = i
                break
        
        assert first_trainable is not None

        # 2. Extract question from first user message
        question = None
        msgs = inp.get('messages') or []
        for m in msgs:
            if m.get('role') != 'user':
                continue
            c = m.get('content')
            text = c if isinstance(c, str) else (
                next((p.get('text') for p in c if isinstance(p, dict) and p.get('type') == 'text'), '')
                if isinstance(c, list) else '')
            q_match = _q_line_re.match(text or '')
            if q_match:
                question = q_match.group(1).strip()
            break

        if not question:
            oracle_inputs.append(inp)
            continue

        # 3. Build oracle user message (concise: question + oracle hints only)
        hint_parts = []
        if sf_titles:
            hint_parts.append('Supporting passages: ' + ', '.join(f'"{t}"' for t in sf_titles))
        if gts:
            hint_parts.append('Answer: ' + '; '.join(gts))
        hint_parts.append('You must call `extract_condensed` to read the right original passage from the condensed block with thinking steps, and give the final correct answer')
        oracle_suffix = '\n[Oracle Context] ' + '. '.join(hint_parts) + '.'
        oracle_user_content = f'Question: {question}{oracle_suffix}'

        oracle_msgs = [
            Message(role='system', content=SYSTEM_PROMPT),
            Message(role='user', content=oracle_user_content),
        ]

        # 4. Encode oracle prefix (ends with <|im_start|>assistant\n)
        oracle_feature = template.encode(
            Trajectory(messages=oracle_msgs), add_generation_prompt=True)
        oracle_prefix_ids = list(oracle_feature['input_ids'])

        # 5. Splice: oracle_prefix + response_tokens
        response_tokens = list(input_ids[first_trainable + 1:])
        response_labels = list(labels[first_trainable:])

        oracle_input_ids = oracle_prefix_ids + response_tokens
        # Last position of oracle prefix predicts first response token
        oracle_labels = [-100] * (len(oracle_prefix_ids) - 1) + response_labels

        assert len(oracle_input_ids) == len(oracle_labels)
        seq_len = len(oracle_input_ids)
        # Start from original keys to keep collator-compatible shape
        oi = dict(inp)
        oi['input_ids'] = oracle_input_ids
        oi['labels'] = oracle_labels
        oi['attention_mask'] = [1] * seq_len
        oi['messages'] = None
        oi['length'] = seq_len
        # Replicate mrope position_ids shape from original input
        orig_pos = inp.get('position_ids')
        if isinstance(orig_pos, torch.Tensor) and orig_pos.dim() == 3:
            n_dims = orig_pos.shape[0]
            pos_range = torch.arange(seq_len).unsqueeze(0).unsqueeze(0)
            oi['position_ids'] = pos_range.expand(n_dims, 1, seq_len)
        else:
            oi['position_ids'] = list(range(seq_len))
        if 'mm_token_type_ids' in inp:
            oi['mm_token_type_ids'] = torch.zeros(1, seq_len)
        oracle_inputs.append(oi)
        any_modified = True

    return oracle_inputs if any_modified else None


def _compute_token_bonus(
    oracle_logps: Any,
    old_logps: List[List[float]],
    f1_labels: List[bool],
    oracle_inputs: List[Dict[str, Any]],
) -> List[List[float]]:
    """Compute per-token bonus = oracle_logps - rollout_logps, zeroed for F1=0 samples.

    oracle_logps is full-sequence form [batch, padded_seq] from forward_only + collector.
    We extract valid positions using oracle_inputs[i]['labels'] mask to get response-only
    logps aligned 1:1 with old_logps.
    """
    import torch

    if isinstance(oracle_logps, torch.Tensor):
        oracle_logps = oracle_logps.float().cpu()

    bonus = []
    for i, (is_pos, old_lp) in enumerate(zip(f1_labels, old_logps)):
        if not is_pos or not old_lp:
            bonus.append([0.0] * len(old_lp) if old_lp else [])
            continue

        n = len(old_lp)
        oracle_labels = oracle_inputs[i].get('labels') or []

        # Build mask from oracle labels to extract valid (trainable) positions
        if isinstance(oracle_logps, torch.Tensor):
            orc_row = oracle_logps[i]
            mask = torch.tensor([l != -100 for l in oracle_labels], dtype=torch.bool)
            seq_len = min(len(mask), orc_row.numel())
            orc_valid = orc_row[:seq_len][mask[:seq_len]].tolist()
        else:
            orc_row = oracle_logps[i] if i < len(oracle_logps) else []
            if isinstance(orc_row, torch.Tensor):
                orc_row = orc_row.float().cpu().tolist()
            elif not isinstance(orc_row, (list, tuple)):
                orc_row = []
            orc_valid = [v for v, l in zip(orc_row, oracle_labels) if l != -100]

        assert len(orc_valid) == n
        bonus.append([o - r for o, r in zip(orc_valid, old_lp)])
    return bonus


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

    logger.info('Building HotpotQA dataset')
    _prebuilt_dataset = create_hotpotqa_dataset()
    logger.info('Dataset ready: %d rows', len(_prebuilt_dataset))

    GLOBAL_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    batches_per_epoch = max(1, len(_prebuilt_dataset) // GLOBAL_BATCH_SIZE)
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
    if INIT_LORA_PATH:
        model.load(INIT_LORA_PATH, adapter_name=ADAPTER_NAME)
        logger.info('Loaded cold-start LoRA from %s', INIT_LORA_PATH)
    if USE_MEGATRON:
        model.set_optimizer('default', lr=LEARNING_RATE)
        model.set_lr_scheduler('default', lr_decay_steps=total_steps, max_lr=LEARNING_RATE)
    else:
        model.set_optimizer('AdamW', lr=LEARNING_RATE)
        model.set_lr_scheduler('CosineAnnealingLR', T_max=total_steps, eta_min=0)

    model.set_loss('GRPOLoss', epsilon=CISPO_EPS_LOW, epsilon_high=CISPO_EPS_HIGH,
                   beta=KL_BETA, entropy_coef=ENTROPY_COEF, token_bonus_coef=ORACLE_BONUS_COEF)
    model.set_processor(InputProcessor, padding_free=True)
    model.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False, max_length=HOTPOTQA_MAX_LENGTH)

    model.add_metric('GRPOMetric', is_training=True,
                     epsilon=CISPO_EPS_LOW, epsilon_high=CISPO_EPS_HIGH,
                     top_k_kl=HIGH_KL_TOPK)

    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8, 'max_model_len': 32768,
            'max_lora_rank': 32, 'enable_lora': True,
            'enable_tower_connector_lora': True,
            'max_loras': 5
        },
        device_mesh=sampler_mesh, remote_group='sampler')
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False, max_length=HOTPOTQA_MAX_LENGTH)
    rollout_template = Qwen3_5Template(
        MODEL_ID, max_length=HOTPOTQA_MAX_LENGTH, enable_thinking=False)

    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)
    chunker = NativeChunker(
        chunk_size=CHUNK_SIZE,
        passage_boundary_re=r'(?<=\n\n)',
    )
    # ``\A`` anchor: prevents a ``Question:`` line inside a passage from being misread as the query.
    _question_re = re.compile(r'\AQuestion:\s*(.+)')

    def _extract_question(chunk):
        content = chunk.get('content')
        if chunk.get('type') != 'text' or not isinstance(content, str):
            return None
        m = _question_re.search(content)
        return m.group(1).strip() if m else None

    condenser = ModelCondenser(
        sampler=sampler,
        compression_ratio=2.0,
        sampling_params=SamplingParams(
            max_tokens=1024, num_samples=1, temperature=0.4, top_p=0.9),
        min_chars=200,
        template=rollout_template,
        lora_path='ms://twinkle-kit/Qwen3.5-4B-Condenser',
        skip_pattern=r'^Question:',
        related_query=_extract_question,
    )

    dataloader = DataLoader(
        dataset=lambda: _prebuilt_dataset,
        batch_size=GLOBAL_BATCH_SIZE, min_batch_size=GLOBAL_BATCH_SIZE)

    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1,
        temperature=1.0, top_p=0.95,
        stop=['</tool_call>'])

    def _trace_should_store(traj):
        return _F1_REWARD([traj])[0] == 0.0

    def _trace_is_success(traj):
        return _F1_REWARD([traj])[0] > 0.0

    rollout = MultiTurnCondenseRollout(
        sampler=sampler,
        template=rollout_template,
        tool_manager=ToolManager(),
        chunker=chunker,
        condenser=condenser,
        sampling_params=sampling_params,
        max_turns=MAX_TURNS,
        max_trajectory_tokens=MAX_TRAJECTORY_TOKENS,
        trace_dir=_ROLLOUT_TRACE_DIR or None,
        trace_callback=_trace_should_store,
        success_callback=_trace_is_success,
        post_compress_callback=(
            _make_oracle_hint_callback(total_steps) if ORACLE_HINT else None),
    )

    optim_step = 0
    logger.info('Starting HotpotQA GRPO training (LLM condenser variant)')

    def _epoch_cycle(dl, n_epochs):
        for ep in range(1, n_epochs + 1):
            logger.info(f'=== Epoch {ep}/{n_epochs} (step={optim_step}/{total_steps}) ===')
            for batch in dl:
                yield batch

    for batch in _epoch_cycle(dataloader, NUM_EPOCHS):
        if optim_step >= total_steps:
            break

        # Single source of truth for the step shown in swanlab / logger / rollout-trace filename.
        # Equals the number of optimizer updates already completed when this rollout was sampled.
        batch_step = optim_step

        metrics.reset()
        expand_prompts = [p for prompt in batch for p in [prompt] * NUM_GENERATIONS]

        ckpt_manager.sync_weights(merge_and_sync=False)
        sampler.reset_prefix_cache()

        # Batched multi-turn rollout with chunk+condense pre-processing.
        # Each returned trajectory is a flat dict containing ``messages``,
        # ``input_ids``, ``labels``, ``attention_mask``, ``position_ids``,
        # ``turns``, ``logprobs``, ``stop_reason``, ``truncated``.
        all_trajectories: List[Dict[str, Any]] = rollout(expand_prompts, global_step=batch_step)
        n_turns_per_rollout = [int(t.get('turns') or 0) for t in all_trajectories]
        per_rollout_completion_length = [
            sum(1 for l in (t.get('labels') or []) if l != -100)
            for t in all_trajectories]

        total_rewards, f1_rewards, cot_rewards, tool_explore_rewards = \
            compute_rewards(all_trajectories)

        rollout_advantages = advantage_fn(
            total_rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()

        all_f1_labels: List[bool] = [f > 0 for f in f1_rewards]
        n_pos = sum(1 for p in all_f1_labels if p)
        n_neg = sum(1 for p in all_f1_labels if not p)
        pos_with_neg_adv = sum(1 for p, a in zip(all_f1_labels, rollout_advantages) if p and a < 0)
        neg_with_pos_adv = sum(1 for p, a in zip(all_f1_labels, rollout_advantages) if not p and a > 0)

        # Skip homogeneous groups where gradient signal is meaningless
        f1_pos_rate = n_pos / len(f1_rewards) if f1_rewards else 0.5
        if f1_pos_rate > 0.9 or f1_pos_rate < 0.1:
            logger.info('[skip-homogeneous] f1_pos_rate=%.3f, skipping training update', f1_pos_rate)
            metrics.accumulate(
                completion_lengths=per_rollout_completion_length,
                rewards={'total': total_rewards, 'f1': f1_rewards,
                         'cot': cot_rewards, 'tool_explore': tool_explore_rewards})
            log_dict = metrics.calculate()
            log_dict.update(_compute_rollout_diagnostics(
                all_trajectories, n_turns_per_rollout, per_rollout_completion_length,
                f1_rewards=f1_rewards, old_logps=[[lp[0][1] for lp in (t.get('logprobs') or [])] for t in all_trajectories]))
            log_dict['skipped'] = True
            log_dict['pos_neg_adv_rate'] = pos_with_neg_adv / n_pos if n_pos else 0.0
            log_dict['neg_pos_adv_rate'] = neg_with_pos_adv / n_neg if n_neg else 0.0
            log_dict['adv_max'] = max(rollout_advantages) if rollout_advantages else 0.0
            log_dict['adv_min'] = min(rollout_advantages) if rollout_advantages else 0.0
            swanlab.log(_coerce_for_swanlab(log_dict), step=batch_step)
            metrics.reset()
            logger.info(f'[Step {batch_step}/{total_steps}] [SKIPPED] {log_dict}')
            optim_step += optim_steps_per_batch
            continue

        metrics.accumulate(
            completion_lengths=per_rollout_completion_length,
            rewards={'total': total_rewards, 'f1': f1_rewards,
                     'cot': cot_rewards, 'tool_explore': tool_explore_rewards})

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
            mb_inputs = all_input_data[mb_start:mb_end]
            # Reference log-probs for KL: same policy model with LoRA adapter disabled (= base model).
            # Skipped when KL_BETA == 0 to save one extra forward per mini-batch.
            ref_logps = None
            if KL_BETA > 0.0:
                ref_outputs = model.forward_only(inputs=mb_inputs, disable_lora=True)
                ref_logps = ref_outputs.get('logps') if isinstance(ref_outputs, dict) else getattr(ref_outputs, 'logps', None)
            # [EXP-ORACLE] per-token bonus: forward with oracle context, diff against rollout logps
            mb_token_bonus = None
            if ORACLE_BONUS_COEF > 0.0:
                mb_oracle_inputs = _build_oracle_inputs(
                    mb_inputs, all_f1_labels[mb_start:mb_end], rollout_template)
                if mb_oracle_inputs is not None:
                    oracle_outputs = model.forward_only(inputs=mb_oracle_inputs)
                    oracle_logps = oracle_outputs.get('logps') if isinstance(oracle_outputs, dict) else getattr(oracle_outputs, 'logps', None)
                    if oracle_logps is not None:
                        mb_token_bonus = _compute_token_bonus(
                            oracle_logps, all_old_logps[mb_start:mb_end],
                            all_f1_labels[mb_start:mb_end], mb_oracle_inputs)
            model.forward_backward(
                inputs=mb_inputs,
                old_logps=all_old_logps[mb_start:mb_end],
                advantages=advantages[mb_start:mb_end],
                ref_logps=ref_logps,
                token_bonus=mb_token_bonus,
                positive_mask=all_f1_labels[mb_start:mb_end],
                micro_batch_size=MICRO_BATCH_SIZE)
            model.clip_grad_and_step()
            optim_step += 1
            if optim_step >= total_steps:
                break
            if optim_step % SAVE_STEPS == 0:
                model.save(f'hotpotqa-grpo-tools-llmcondense-checkpoint-{optim_step}')

        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric(is_training=True))
        log_dict.update(_compute_rollout_diagnostics(
            all_trajectories, n_turns_per_rollout, per_rollout_completion_length,
            f1_rewards=f1_rewards, old_logps=all_old_logps))
        log_dict['pos_neg_adv_rate'] = pos_with_neg_adv / n_pos if n_pos else 0.0
        log_dict['neg_pos_adv_rate'] = neg_with_pos_adv / n_neg if n_neg else 0.0
        log_dict['adv_max'] = max(rollout_advantages) if rollout_advantages else 0.0
        log_dict['adv_min'] = min(rollout_advantages) if rollout_advantages else 0.0
        # Pop high-KL token records before swanlab.log: list-of-dict won't render as a chart.
        _hk = log_dict.pop('_high_kl_records', None)
        if _hk:
            _tok = rollout_template.tokenizer
            for r in _hk:
                gsi = r.get('gsi')
                tid = all_trajectories[gsi].get('id') if gsi is not None and 0 <= gsi < len(all_trajectories) else None
                try:
                    tok_text = _tok.decode([r['token_id']])
                except Exception:
                    tok_text = None
                logger.info(
                    '[high-kl] step=%d gsi=%s tid=%s pos=%s tok=%r kl=%.4f r=%.4f lp_new=%.4f lp_old=%.4f',
                    batch_step, gsi, tid, r.get('pos'), tok_text,
                    r.get('kl'), r.get('ratio'), r.get('logp_new'), r.get('logp_old'))
        swanlab.log(_coerce_for_swanlab(log_dict), step=batch_step)
        metrics.reset()
        logger.info(f'[Step {batch_step}/{total_steps}] {log_dict}')

    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('hotpotqa-grpo-tools-llmcondense-final')


if __name__ == '__main__':
    main()
