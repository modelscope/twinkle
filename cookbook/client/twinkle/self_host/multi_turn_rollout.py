# Twinkle Client - Multi-turn agentic rollout Example
#
# Client counterpart of the ray-local core-lib example
# ``cookbook/rl/multi_turn/multi_turn_grpo.py``. Instead of a Ray ``MultiTurnRollout``
# + ``EnvPool``, it drives the SAME multi-turn "sample -> tool -> bridge -> sample"
# loop over HTTP with ``twinkle_client.rollout.ClientMultiTurnRollout`` and the
# client ``vLLMSampler``.
#
# Differences from the ray-local example (by design):
#   * EnvPool is a Ray-only component and is NOT usable from the client, so this
#     example uses a self-contained Python tool (a small calculator) wrapped in a
#     ``ToolManager`` instead of an interactive environment.
#   * ``ClientMultiTurnRollout`` samples with ``num_samples=1``; for a GRPO group we
#     replicate each prompt ``NUM_GENERATIONS`` times as separate trajectories
#     (exactly what the ray-local example does with ``BATCH_SIZE * NUM_GENERATIONS``).
#
# The server must be running first (see server.py / server_config.yaml). Both the
# model and sampler services must be configured.
#
# NOTE on weight sync: ``ClientMultiTurnRollout`` samples with the sampler's
# currently loaded server-side weights. A full RL loop would periodically
# ``model.save(is_sampler=True)`` and point the sampler at the saved adapter; that
# sync is intentionally omitted here to keep the rollout example focused.

import dotenv

dotenv.load_dotenv('.env')

from typing import Any, Dict, List

from peft import LoraConfig

from twinkle import get_logger, init_twinkle_client
from twinkle.advantage import GRPOAdvantage
from twinkle.data_format import SamplingParams
from twinkle.template import Qwen3_5Template
from twinkle_agentic.tools.base import Tool
from twinkle_agentic.tools.tool_manager import ToolManager
from twinkle_client.model import MultiLoraTransformersModel
from twinkle_client.rollout import ClientMultiTurnRollout
from twinkle_client.sampler import vLLMSampler

logger = get_logger()

# ========== Configuration ==========
MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
ADAPTER_NAME = 'default'
NUM_GENERATIONS = 2       # GRPO group size (rollout runs num_samples=1 per trajectory)
MAX_NEW_TOKENS = 512
MAX_TURNS = 4
LEARNING_RATE = 1e-5
MAX_STEPS = 3

SYSTEM_PROMPT = ('You are a careful assistant. When a calculation is needed, call the '
                 '`calculator` tool with a single arithmetic ``expression`` (e.g. "12*7+3") '
                 'and then give the final numeric answer.')

# A few arithmetic questions; each is expanded into NUM_GENERATIONS trajectories.
QUESTIONS: List[Dict[str, Any]] = [
    {'q': 'What is 12 * 7 + 3?', 'answer': 87},
    {'q': 'Compute (45 - 6) * 2.', 'answer': 78},
]


# ========== A self-contained tool (replaces EnvPool) ==========
class CalculatorTool(Tool):
    """Evaluate a single arithmetic ``expression`` string and return the result."""

    _ALLOWED = set('0123456789+-*/(). ')

    def tool_info(self) -> Dict[str, Any]:
        return {
            'type': 'function',
            'function': {
                'name': 'calculator',
                'description': 'Evaluate one arithmetic expression and return its numeric value.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'expression': {
                            'type': 'string',
                            'description': 'The arithmetic expression to evaluate, e.g. "12*7+3".',
                        }
                    },
                    'required': ['expression'],
                },
            },
        }

    def __call__(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        expr = str(arguments.get('expression', '')).strip()
        if not expr:
            return 'Error: missing "expression".'
        if not set(expr) <= self._ALLOWED:
            return 'Error: expression may only contain digits, spaces and + - * / ( ) .'
        try:
            # Safe: the whitelist above forbids names, calls and attribute access.
            return str(eval(expr, {'__builtins__': {}}, {}))  # noqa: S307
        except Exception as e:  # noqa
            return f'Error: could not evaluate {expr!r}: {type(e).__name__}'


def build_trajectories(tool_schema: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand each question into NUM_GENERATIONS identical trajectories (GRPO group)."""
    trajectories: List[Dict[str, Any]] = []
    for item in QUESTIONS:
        for _ in range(NUM_GENERATIONS):
            trajectories.append({
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': item['q']},
                ],
                'tools': tool_schema,
            })
    return trajectories


def compute_rewards(trajectories: List[Dict[str, Any]]) -> List[float]:
    """+1 if the correct numeric answer appears in the final assistant message."""
    rewards: List[float] = []
    n_per_q = NUM_GENERATIONS
    for i, traj in enumerate(trajectories):
        expected = str(QUESTIONS[i // n_per_q]['answer'])
        final = ''
        for msg in reversed(traj.get('messages', [])):
            if msg.get('role') == 'assistant':
                final = str(msg.get('content') or '')
                break
        rewards.append(1.0 if expected in final else 0.0)
    return rewards


def train():
    # Step 1: connect to the running Twinkle server.
    init_twinkle_client(base_url='http://127.0.0.1:8000', api_key='EMPTY_TOKEN')

    # Step 2: training model (GRPO), mirroring the ray-local example's config.
    model = MultiLoraTransformersModel(model_id=MODEL_ID)
    model.add_adapter_to_model(ADAPTER_NAME, LoraConfig(target_modules='all-linear', r=16, lora_alpha=32))
    model.set_loss('GRPOLoss', epsilon=0.2)
    model.set_optimizer('Adam', lr=LEARNING_RATE)
    model.set_processor('InputProcessor')
    model.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False)

    # Step 3: client sampler (HTTP).
    sampler = vLLMSampler(model_id=MODEL_ID)
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False)

    # Step 4: multi-turn rollout. Needs a LOCAL template for bridge-token stitching
    # (rendering tool turns + the next generation prompt) and a ToolManager.
    rollout_template = Qwen3_5Template(model_id=MODEL_ID, max_length=8192, enable_thinking=False)
    rollout_template.truncation_strategy = 'delete'
    tool_manager = ToolManager([CalculatorTool()])
    tool_schema = tool_manager.tool_infos()

    rollout = ClientMultiTurnRollout(
        sampler=sampler,
        template=rollout_template,
        tool_manager=tool_manager,
        sampling_params=SamplingParams(
            max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1, temperature=1.0, top_p=0.95),
        max_turns=MAX_TURNS,
    )

    advantage_fn = GRPOAdvantage()

    for step in range(MAX_STEPS):
        # 1. Run the batched multi-turn rollout (sample -> tool -> bridge -> sample).
        trajectories = build_trajectories(tool_schema)
        rolled = rollout(trajectories, tool_manager=tool_manager)

        # 2. Read back per-trajectory logprobs (top-1 chosen-token logprob) and rewards.
        all_inputs: List[Dict[str, Any]] = []
        all_old_logps: List[List[float]] = []
        for traj in rolled:
            logprobs = traj.get('logprobs') or []
            all_old_logps.append([lp[0][1] for lp in logprobs])
            all_inputs.append(traj)
        rewards = compute_rewards(rolled)

        avg_turns = sum(t.get('turns') or 0 for t in rolled) / len(rolled)
        logger.info(f'[Step {step}] avg_reward={sum(rewards) / len(rewards):.3f} avg_turns={avg_turns:.1f}')

        # 3. GRPO advantages (group-relative within NUM_GENERATIONS).
        advantages = advantage_fn(rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()
        if all(abs(a) < 1e-8 for a in advantages):
            logger.info(f'[Step {step}] all advantages zero, skipping update')
            continue

        # 4. Policy update over the rolled-out trajectories.
        model.forward_backward(inputs=all_inputs, advantages=advantages, old_logps=all_old_logps)
        model.clip_grad_and_step()
        logger.info(f'[Step {step}] {model.calculate_metric(is_training=True).result}')

    logger.info('Multi-turn rollout example finished.')


if __name__ == '__main__':
    train()
