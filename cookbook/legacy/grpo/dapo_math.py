"""
DAPO-Math-17k GRPO training demo for Twinkle.

Uses the AI-ModelScope/DAPO-Math-17k dataset (~17k competition-level math
problems).  All ground-truth answers in this dataset are **integers**, making
reward verification straightforward and reliable.

Reward = accuracy_reward only (no format reward).
  - From the last 300 chars of model output, extracts:
    1) ``Answer: <number>``  (the prompt asks for this format)
    2) ``\\boxed{<number>}`` (fallback)
  - Normalizes both prediction and ground truth to integers.
  - Returns 1.0 for correct, 0.0 for incorrect.

Designed for strong instruct models (e.g. Qwen3-30B-A3B-Instruct) that do
NOT use <think> tags.  The dataset prompt already contains step-by-step
instructions, so no additional system prompt is needed.

Reference reward implementations:
  - verl:  verl/utils/reward_score/math_dapo.py  (compute_score)
  - slime: slime/rollout/rm_hub/math_dapo_utils.py (compute_score)
"""
import gc
import os
import re
import time
from typing import List, Tuple, Dict, Any, Optional

from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import SamplingParams, SampleResponse
from twinkle.data_format import Trajectory, InputFeature, Message
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import Preprocessor
from twinkle.processor import InputProcessor
from twinkle.reward.base import Reward
from twinkle.sampler import vLLMSampler
from twinkle.template import Template
from twinkle.metric import CompletionRewardMetric

logger = get_logger()

# ========== Configuration ==========
MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen3-30B-A3B-Instruct-2507')
USE_MEGATRON = bool(int(os.environ.get('USE_MEGATRON', '1')))

MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 4))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 4))
SAMPLER_TP = int(os.environ.get('SAMPLER_TP', 1))
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS
PP_SIZE = 4
NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', 4))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 4096))
LEARNING_RATE = float(os.environ.get('LR', 1e-5))
GRPO_EPSILON = float(os.environ.get('GRPO_EPSILON', 0.2))
GRPO_BETA = float(os.environ.get('GRPO_BETA', 0.0))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 200))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 1))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 1))
TEMPERATURE = float(os.environ.get('TEMPERATURE', 1.0))
WEIGHT_SYNC_INTERVAL = int(os.environ.get('WEIGHT_SYNC_INTERVAL', 1))
ADAPTER_NAME = 'default'
DATA_NUM = int(os.environ.get('DATA_NUM', 17000))  # DAPO-Math-17k has ~17k samples

# SwanLab experiment tracking
USE_SWANLAB = bool(int(os.environ.get('USE_SWANLAB', '1')))
if USE_SWANLAB:
    import swanlab
    swanlab.login(api_key=os.environ['SWANLAB_API_KEY'], save=True)
    swanlab.init(project="twinkle-math", config={
        'model_id': MODEL_ID,
        'dataset': 'DAPO-Math-17k',
        'num_gpus': NUM_GPUS,
        'model_gpus': MODEL_GPUS,
        'sampler_gpus': SAMPLER_GPUS,
        'num_generations': NUM_GENERATIONS,
        'max_new_tokens': MAX_NEW_TOKENS,
        'learning_rate': LEARNING_RATE,
        'grpo_epsilon': GRPO_EPSILON,
        'grpo_beta': GRPO_BETA,
        'batch_size': BATCH_SIZE,
        'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
    })


# ========== DAPO Math Reward (adapted from verl/slime math_dapo) ==========
# All answers in DAPO-Math-17k are integers, so verification is simpler
# than general MATH problems.

# --- Normalization constants (from verl/slime math_dapo_utils) ---
_SUBSTITUTIONS = [
    ("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""), (r"\ ", ""),
    (" ", ""), ("mbox", "text"), (",\\text{and}", ","),
    ("\\text{and}", ","), ("\\text{m}", "\\text{}"),
]
_REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "hours",
    "km", "units", "\\ldots", "sue", "points", "feet", "minutes",
    "digits", "cents", "degrees", "cm", "gm", "pounds", "meters",
    "meals", "edges", "students", "childrentickets", "multiples",
    "\\text{s}", "\\text{.}", "\\text{\ns}", "\\text{}^2",
    "\\text{}^3", "\\text{\n}", "\\text{}", r"\mathrm{th}",
    r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
    "<|im_end|>", "<|endoftext|>",
]


def _normalize_final_answer(answer: str) -> str:
    """Normalize a math answer string for comparison.

    Adapted from verl/slime math_dapo_utils.normalize_final_answer.
    """
    answer = str(answer)
    answer = answer.split("=")[-1]
    for before, after in _SUBSTITUTIONS:
        answer = answer.replace(before, after)
    for expr in _REMOVED_EXPRESSIONS:
        answer = answer.replace(expr, "")
    # Strip LaTeX wrappers
    answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", answer)
    answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", answer)
    answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", answer)
    answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", answer)
    answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", answer)
    answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", answer)
    answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", answer)
    answer = answer.replace("$", "")
    if answer.replace(",", "").isdigit():
        answer = answer.replace(",", "")
    return answer.strip()


def _last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last \\boxed{...} from a string."""
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return string[idx: right_brace_idx + 1] if right_brace_idx is not None else None


def _remove_boxed(s: str) -> str:
    """Remove \\boxed{} wrapper."""
    left = "\\boxed{"
    if not (s.startswith(left) and s.endswith("}")):
        return s
    return s[len(left): -1]


def _extract_answer_minerva(
    solution: str,
    pattern: str = r"(?i)Answer\s*:\s*([^\n]+)",
) -> Optional[str]:
    """Extract answer via 'Answer: ...' pattern (Minerva-style).

    This is the primary extraction method since the DAPO prompt asks:
    "The last line of your response should be of the form Answer: $Answer"
    """
    matches = re.findall(pattern, solution)
    if matches:
        return _normalize_final_answer(matches[-1])
    return None


def _extract_answer_boxed(solution: str) -> Optional[str]:
    """Extract answer from \\boxed{} (fallback)."""
    boxed = _last_boxed_only_string(solution)
    if boxed is not None:
        try:
            return _remove_boxed(boxed)
        except Exception:
            pass
    return None


def compute_dapo_score(completion: str, ground_truth: str) -> Dict[str, Any]:
    """Compute DAPO-Math reward score for a single sample.

    Adapted from verl/utils/reward_score/math_dapo.py compute_score.
    All DAPO-Math-17k answers are integers, so we normalize to int for
    comparison.

    Returns dict with 'score' (1.0 or 0.0), 'acc' (bool), 'pred' (str).
    """
    # Only look at the tail for efficiency
    tail = completion[-300:] if len(completion) > 300 else completion

    # Normalize ground truth to integer string
    try:
        gt_normalized = str(int(float(ground_truth)))
    except (ValueError, OverflowError):
        gt_normalized = _normalize_final_answer(ground_truth)

    # Try "Answer: ..." extraction first (matches the prompt format)
    pred = _extract_answer_minerva(tail)
    if pred is not None:
        pred_normalized = _normalize_final_answer(pred)
        try:
            pred_int = str(int(float(pred_normalized)))
            correct = (pred_int == gt_normalized)
        except (ValueError, OverflowError):
            correct = (pred_normalized == gt_normalized)
        return {"score": 1.0 if correct else 0.0, "acc": correct, "pred": pred}

    # Fallback: try \boxed{}
    pred = _extract_answer_boxed(tail)
    if pred is not None:
        pred_normalized = _normalize_final_answer(pred)
        try:
            pred_int = str(int(float(pred_normalized)))
            correct = (pred_int == gt_normalized)
        except (ValueError, OverflowError):
            correct = (pred_normalized == gt_normalized)
        return {"score": 1.0 if correct else 0.0, "acc": correct, "pred": pred}

    return {"score": 0.0, "acc": False, "pred": None}


# ========== Preprocessor ==========
class DAPOMathProcessor(Preprocessor):
    """Preprocessor for DAPO-Math-17k dataset.

    Dataset fields:
      - data_source: "math_dapo"
      - prompt: list of messages [{"role": "user", "content": "..."}]
      - reward_model: {"ground_truth": "<integer>", "style": "..."}
      - ability: "MATH"
      - extra_info: {"index": "..."}

    The prompt already contains instructions ("Solve ... step by step.
    The last line of your response should be of the form Answer: $Answer"),
    so no additional system prompt is needed.
    """

    def __call__(self, row) -> Trajectory:
        # prompt is already a list of message dicts
        prompt_messages = row['prompt']
        ground_truth = row['reward_model']['ground_truth']

        messages = [
            Message(role=msg['role'], content=msg['content'])
            for msg in prompt_messages
        ]
        return Trajectory(
            messages=messages,
            user_data=[('ground_truth', str(ground_truth))],
        )


# ========== Reward Functions ==========
class DAPOMathAccuracyReward(Reward):
    """Accuracy reward for DAPO-Math-17k.

    Extracts the answer from model output and compares with ground truth.
    Uses the same verification logic as verl/slime math_dapo.
    Returns 1.0 for correct, 0.0 for incorrect.
    """

    def __call__(
        self, trajectories: List[Trajectory], ground_truths: List[Trajectory]
    ) -> List[float]:
        rewards = []
        for trajectory in trajectories:
            messages = trajectory.get('messages', [])
            completion = ''
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '')
                    break

            gt = ''
            user_data = trajectory.get('user_data', [])
            if isinstance(user_data, list):
                for item in user_data:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        if item[0] == 'ground_truth':
                            gt = str(item[1])
                            break

            if completion and gt:
                result = compute_dapo_score(completion, gt)
                rewards.append(result['score'])
            else:
                rewards.append(0.0)
        return rewards


def create_dapo_math_dataset():
    """Create DAPO-Math-17k dataset.

    Downloads from ModelScope: AI-ModelScope/DAPO-Math-17k
    """
    meta = DatasetMeta(
        "ms://AI-ModelScope/DAPO-Math-17k",
        split='train',
        data_slice=range(DATA_NUM),
    )
    dataset = Dataset(meta)
    dataset.set_template("Template", model_id=MODEL_ID, max_length=2048)
    dataset.map(DAPOMathProcessor())
    dataset.encode(add_generation_prompt=True)
    return dataset


def compute_rewards(
    trajectories: List[Trajectory],
) -> Tuple[List[float], List[float]]:
    """Compute accuracy rewards for DAPO-Math.

    Returns (total_rewards, accuracy_rewards).
    No format reward — instruct model does not need thinking tags.
    """
    accuracy_reward_fn = DAPOMathAccuracyReward()
    accuracy_rewards = accuracy_reward_fn(trajectories, [])
    return accuracy_rewards, accuracy_rewards


# ========== Main ==========
def main():
    from twinkle.utils.import_utils import requires
    requires("vllm>=0.15.0")

    device_groups = [
        DeviceGroup(
            name='model',
            ranks=list(range(MODEL_GPUS)),
            device_type='GPU',
            gpus_per_worker=1,
        ),
        DeviceGroup(
            name='sampler',
            ranks=list(range(MODEL_GPUS, NUM_GPUS)),
            device_type='GPU',
            gpus_per_worker=SAMPLER_TP,
        ),
    ]
    if USE_MEGATRON:
        model_mesh = DeviceMesh.from_sizes(
            dp_size=1, 
            tp_size=2,
            pp_size=2,
            ep_size=2,
        )
    else:
        model_mesh = DeviceMesh.from_sizes(
            world_size=MODEL_GPUS, dp_size=MODEL_GPUS,
        )
    assert SAMPLER_GPUS % SAMPLER_TP == 0
    sampler_mesh = DeviceMesh.from_sizes(
        world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS // SAMPLER_TP,
        tp_size=SAMPLER_TP,
    )
    twinkle.initialize(
        mode='ray',
        nproc_per_node=NUM_GPUS,
        groups=device_groups,
        lazy_collect=False,
    )

    lora_config = LoraConfig(
        target_modules=['linear_qkv', 'linear_proj'],
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
    )

    # ── Sampler ───────────────────────────────────────────────────────
    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': 8192,
            'max_loras': 1,
            'max_lora_rank': 32,
            'enable_sleep_mode': False,
            'enable_lora': True,
            "logprobs_mode": "processed_logprobs",
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template(Template, model_id=MODEL_ID)

    # ── Model ─────────────────────────────────────────────────────────
    if USE_MEGATRON:
        from twinkle.model.megatron import MegatronModel
        model = MegatronModel(
            model_id=MODEL_ID,
            device_mesh=model_mesh,
            remote_group='model',
            mixed_precision='bf16',
            recompute_granularity='full',
            recompute_num_layers=None,
        )
    else:
        model = TransformersModel(
            model_id=MODEL_ID,
            device_mesh=model_mesh,
            remote_group='model',
        )

    model.add_adapter_to_model(
        ADAPTER_NAME,
        lora_config,
        gradient_accumulation_steps=1,
    )
    if USE_MEGATRON:
        model.set_optimizer(
            'default', lr=LEARNING_RATE,
        )
        model.set_lr_scheduler(
            'default',
            lr_decay_steps=MAX_STEPS,
            max_lr=LEARNING_RATE,
        )
    else:
        model.set_optimizer(
            'AdamW', lr=LEARNING_RATE,
        )
        model.set_lr_scheduler(
            'CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0,
        )
    model.set_loss(
        'GRPOLoss',
        epsilon=GRPO_EPSILON,
        beta=GRPO_BETA,
    )
    model.set_processor(InputProcessor)
    model.set_template('Template', model_id=MODEL_ID)

    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)

    # Global batch = prompts for one full gradient accumulation cycle
    GLOBAL_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    dataloader = DataLoader(
        dataset=create_dapo_math_dataset,
        batch_size=GLOBAL_BATCH_SIZE,
        min_batch_size=GLOBAL_BATCH_SIZE,
        device_mesh=model_mesh,
        remote_group='model',
        num_workers=0,
    )
    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()

    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.95,
    )

    # ── Training loop ─────────────────────────────────────────────────
    optim_step = 0
    logger.info(get_device_placement())

    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break

        step_start = time.perf_counter()
        metrics.reset()
        timings: Dict[str, float] = {
            'weight_sync': 0.0,
            'generate': 0.0,
            'reward': 0.0,
            'advantage': 0.0,
            'train': 0.0,
            'total': 0.0,
        }

        global_prompts = batch if isinstance(batch, list) else [batch]

        t0 = time.perf_counter()
        if optim_step % WEIGHT_SYNC_INTERVAL == 0:
            ckpt_manager.sync_weights(merge_and_sync=False)
            sampler.reset_prefix_cache()
        timings['weight_sync'] = time.perf_counter() - t0

        t1 = time.perf_counter()
        sample_response = sampler.sample(
            global_prompts*NUM_GENERATIONS,
            sampling_params,
            num_samples=1,
        )
        timings['generate'] = time.perf_counter() - t1

        all_input_data: List[Dict[str, Any]] = []
        all_old_logps: List[List[float]] = []
        all_completion_lengths: List[int] = []

        for sequence in sample_response.sequences:
            all_input_data.append(sequence.new_input_feature)
            all_old_logps.append(sequence.logprobs)
            all_completion_lengths.append(len(sequence.tokens))

        if not all_input_data:
            logger.warning(
                f"Optim step {optim_step}: No valid samples, skipping"
            )
            continue

        # ========== 3. Rewards ==========
        t2 = time.perf_counter()
        total_rewards, accuracy_rewards = compute_rewards(all_input_data)
        timings['reward'] = time.perf_counter() - t2

        metrics.accumulate(
            None,
            None,
            generate_time=timings['generate'],
            weight_sync_time=timings['weight_sync'],
            completion_lengths=all_completion_lengths,
            rewards={
                'total': total_rewards,
                'accuracy': accuracy_rewards,
            },
        )

        # ========== 4. Advantages ==========
        t3 = time.perf_counter()
        advantages = advantage_fn(
            total_rewards,
            num_generations=NUM_GENERATIONS,
            scale='group',
        )
        advantages = advantages.tolist()
        timings['advantage'] = time.perf_counter() - t3

        frac_zero_std = (
            1.0 if all(abs(a) < 1e-8 for a in advantages) else 0.0
        )

        # ========== 5. Training ==========
        t4 = time.perf_counter()

        model.forward_backward(
            inputs=all_input_data,
            advantages=advantages,
            old_logps=all_old_logps,
        )

        model.clip_grad_and_step()
        timings['train'] = time.perf_counter() - t4

        gc.collect()
        from twinkle import torch_util
        torch_util.empty_cache()

        timings['total'] = time.perf_counter() - step_start
        optim_step += 1

        # ========== 6. Log ==========
        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric(is_training=True))
        log_dict['train/frac_reward_zero_std'] = frac_zero_std
        log_dict['train/optim_step'] = optim_step
        for k, v in timings.items():
            log_dict[f'time/{k}'] = round(v, 2)

        if USE_SWANLAB:
            swanlab.log(log_dict)
        logger.info(f"[Step {optim_step}/{MAX_STEPS}] {log_dict}")

    logger.info(f"Training completed. optim_steps={optim_step}")
    model.save('grpo-math-checkpoint')


if __name__ == '__main__':
    main()
