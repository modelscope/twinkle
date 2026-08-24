# Twinkle Client - Multi-turn agentic rollout Example
#
# Client counterpart of the ray-local core-lib example
# ``cookbook/rl/multi_turn/multi_turn_grpo.py``. It drives the same multi-turn
# "sample -> environment action -> observation -> sample" loop over HTTP with
# ``twinkle_client.rollout.ClientMultiTurnRollout`` and an embedded OpenEnv
# environment for each trajectory.
#
# Differences from the ray-local example (by design):
#   * The client does not need a distributed Ray EnvPool. It creates one local
#     ``OpenEnv`` instance per trajectory and exposes it through ``EnvTool``.
#   * ``ClientMultiTurnRollout`` samples with ``num_samples=1``; the GRPO batch uses
#     ``BATCH_SIZE * NUM_GENERATIONS`` independent environment trajectories, matching
#     the ray-local example.
#
# The server must be running first (see server.py / server_config.yaml). Both the
# model and sampler services must be configured.
#
# NOTE on weight sync: ``ClientMultiTurnRollout`` samples with the sampler's
# currently loaded server-side weights. A full RL loop would periodically
# ``model.save(is_sampler=True)`` and point the sampler at the saved adapter; that
# sync is intentionally omitted here to keep the rollout example focused.

import os
from typing import Any, Dict, List, Tuple

import dotenv
from peft import LoraConfig

from twinkle import get_logger, init_twinkle_client
from twinkle.advantage import GRPOAdvantage
from twinkle.data_format import SamplingParams
from twinkle.template import Qwen3_5Template
from twinkle_agentic.envs import EnvTool, OpenEnv
from twinkle_agentic.tools.tool_manager import ToolManager
from twinkle_client.model import MultiLoraTransformersModel
from twinkle_client.rollout import ClientMultiTurnRollout
from twinkle_client.sampler import vLLMSampler

dotenv.load_dotenv('.env')

logger = get_logger()

# ========== Configuration ==========
BASE_MODEL = os.environ.get('TWINKLE_MODEL_ID', 'Qwen/Qwen3.5-4B')
MODEL_ID = f'ms://{BASE_MODEL}'
ADAPTER_NAME = 'default'
NUM_GENERATIONS = 2       # GRPO group size (rollout runs num_samples=1 per trajectory)
BATCH_SIZE = 2
MAX_NEW_TOKENS = 512
MAX_TURNS = 4
LEARNING_RATE = 1e-5
MAX_STEPS = 3
OPENENV_NAME = os.environ.get('OPENENV_NAME', 'openspiel_env')
OPENENV_GAME = os.environ.get('OPENENV_GAME', 'blackjack')

BLACKJACK_TOOL_SCHEMA = [
    {
        'type': 'function',
        'function': {
            'name': 'play',
            'description': 'Take an action in the blackjack game.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'action': {
                        'type': 'string',
                        'enum': ['hit', 'stand'],
                        'description': 'Choose "hit" to draw a card or "stand" to keep the current hand.',
                    },
                },
                'required': ['action'],
            },
        },
    },
]

BLACKJACK_ACTION_MAP = {'hit': 0, 'stand': 1}

SYSTEM_PROMPT = """You are a skilled blackjack player. You will be told your current hand and the dealer's visible card.

Your goal is to win the game by getting as close to 21 as possible without going over.

Use the `play` tool to choose either `hit` or `stand`. Reason briefly before each action. Once the environment reports that the game is over, give a short final answer without calling another tool."""


def blackjack_action_mapper(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Convert play(action=...) into the action format expected by OpenSpiel."""
    action = str(arguments.get('action', 'stand')).lower().strip()
    return {
        'action_id': BLACKJACK_ACTION_MAP.get(action, BLACKJACK_ACTION_MAP['stand']),
        'game_name': OPENENV_GAME,
    }


def create_env_tool(env: OpenEnv) -> EnvTool:
    """Expose the blackjack OpenEnv action through ToolManager."""
    function = BLACKJACK_TOOL_SCHEMA[0]['function']
    return EnvTool(
        env=env,
        tool_name=function['name'],
        description=function['description'],
        parameters=function['parameters'],
    )


def prepare_trajectories(
    n_trajectories: int,
) -> Tuple[List[Dict[str, Any]], List[ToolManager], List[List[EnvTool]], List[OpenEnv]]:
    """Create and reset one independent OpenEnv instance per trajectory."""
    trajectories = []
    tool_managers = []
    env_tools_list = []
    environments = []

    try:
        for _ in range(n_trajectories):
            env = OpenEnv(
                env_name=OPENENV_NAME,
                env_kwargs={'game_name': OPENENV_GAME},
                action_mapper=blackjack_action_mapper,
            )
            environments.append(env)
            initial_observation = env.reset().observation

            env_tools = [create_env_tool(env)]
            tool_manager = ToolManager(env_tools)
            trajectories.append({
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': initial_observation},
                ],
                'tools': tool_manager.tool_infos(),
            })
            tool_managers.append(tool_manager)
            env_tools_list.append(env_tools)
    except Exception:
        for env in environments:
            env.close()
        raise

    return trajectories, tool_managers, env_tools_list, environments


def extract_rewards(env_tools_list: List[List[EnvTool]]) -> List[float]:
    """Read the terminal reward produced by each OpenEnv episode."""
    return [env_tools[0].episode_reward if env_tools else 0.0 for env_tools in env_tools_list]


def train():
    # Step 1: connect to the running Twinkle server.
    init_twinkle_client(
        base_url=os.environ.get('TWINKLE_SERVER_URL', 'http://localhost:8000'),
        api_key=os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY_TOKEN'),
    )

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

    # Step 4: multi-turn rollout. Each call receives trajectory-bound ToolManagers
    # backed by independent OpenEnv instances.
    rollout_template = Qwen3_5Template(model_id=MODEL_ID, max_length=8192, enable_thinking=False)
    rollout_template.truncation_strategy = 'delete'

    rollout = ClientMultiTurnRollout(
        sampler=sampler,
        template=rollout_template,
        sampling_params=SamplingParams(
            max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1, temperature=1.0, top_p=0.95),
        max_turns=MAX_TURNS,
    )

    advantage_fn = GRPOAdvantage()

    for step in range(MAX_STEPS):
        # 1. Reset independent OpenEnv episodes and run the batched rollout.
        n_trajectories = BATCH_SIZE * NUM_GENERATIONS
        trajectories, tool_managers, env_tools_list, environments = prepare_trajectories(n_trajectories)
        try:
            rolled = rollout(trajectories, tool_manager=tool_managers)
            rewards = extract_rewards(env_tools_list)
        finally:
            for env in environments:
                env.close()

        # 2. Read back per-trajectory logprobs (top-1 chosen-token logprob).
        all_inputs: List[Dict[str, Any]] = []
        all_old_logps: List[List[float]] = []
        for traj in rolled:
            logprobs = traj.get('logprobs') or []
            all_old_logps.append([lp[0][1] for lp in logprobs])
            all_inputs.append(traj)

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
