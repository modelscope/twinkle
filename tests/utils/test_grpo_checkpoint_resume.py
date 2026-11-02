import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'rl_checkpoint',
    ROOT / 'src/twinkle/utils/rl_checkpoint.py',
)
checkpoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint)


class SimulatedCrash(Exception):
    pass


class Model:
    def __init__(self):
        self.step = 0
        self.seen = []

    def forward_backward(self, inputs, old_logps, advantages, **kwargs):
        assert len(inputs) == len(old_logps) == len(advantages)
        self.seen.extend(
            [x, logp, adv]
            for x, logp, adv in zip(inputs, old_logps, advantages)
        )

    def clip_grad_and_step(self, **kwargs):
        self.step += 1

    def calculate_metric(self, **kwargs):
        return {}

    def save(self, name, output_dir, **kwargs):
        assert kwargs['save_optimizer'] is True
        folder = Path(output_dir) / name
        folder.mkdir(parents=True)
        (folder / 'model.json').write_text(json.dumps({
            'cur_step': self.step,
            'seen': self.seen,
            'consumed_train_samples': kwargs['consumed_train_samples'],
        }))

    def resume_from_checkpoint(self, folder, **kwargs):
        state = json.loads((Path(folder) / 'model.json').read_text())
        self.step = state['cur_step']
        self.seen = state['seen']
        return state


class Loader:
    def __init__(self):
        self.consumed = 8

    def get_state(self):
        return {'consumed_train_samples': self.consumed}

    def resume_from_checkpoint(self, consumed_train_samples):
        self.consumed = consumed_train_samples

    def __iter__(self):
        return iter(())


def recipe_tail():
    # Execute the actual save function, loop and final-save statements.
    # Do not import the recipe: that would initialize CLI and sampler imports.
    tree = ast.parse((ROOT / 'cookbook/rl/grpo/grpo.py').read_text())
    main = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == 'main'
    )
    start = next(
        i for i, node in enumerate(main.body)
        if isinstance(node, ast.FunctionDef)
        and node.name == 'save_checkpoint'
    )
    return compile(
        ast.Module(body=main.body[start:], type_ignores=[]),
        '<actual-grpo-loop>',
        'exec',
    )


def run_recipe(model, loader, output_dir, step, pending, crash_step=None, extra_env=None):
    def save(*args, **kwargs):
        path = checkpoint.save_rl_checkpoint(*args, **kwargs)
        if (crash_step is not None
                and kwargs['name'] == f'grpo-gsm8k-checkpoint-{crash_step}'):
            raise SimulatedCrash()
        return path

    env = {
        'model': model,
        'dataloader': loader,
        'optim_step': step,
        'pending': copy.deepcopy(pending),
        'resume_config': {'mini_batch_size': 2},
        'args': SimpleNamespace(
            training=SimpleNamespace(output_dir=str(output_dir)),
        ),
        'ADAPTER_NAME': 'default',
        'MAX_STEPS': 4,
        'MINI_BATCH_SIZE': 2,
        'MICRO_BATCH_SIZE': 1,
        'SAVE_STEPS': 1,
        'save_rl_checkpoint': save,
        'get_device_placement': lambda: {},
        'logger': SimpleNamespace(info=lambda *args: None),
    }
    from typing import Any, Dict, List

    env.update({'Any': Any, 'Dict': Dict, 'List': List})
    if extra_env:
        env.update(extra_env)
    exec(recipe_tail(), env)
    return env


@pytest.mark.parametrize('sample_count', [7, 8])
@pytest.mark.parametrize('crash_step', [1, 2, 4])
def test_restart_matches_uninterrupted_updates(
    tmp_path, sample_count, crash_step
):
    pending = {
        'inputs': list(range(sample_count)),
        'old_logps': [[-i - 0.5] for i in range(sample_count)],
        'advantages': [i + 0.25 for i in range(sample_count)],
        'metrics': {'train/total_reward': 1.0},
    }

    baseline = Model()
    run_recipe(baseline, Loader(), tmp_path / 'baseline', 0, pending)

    interrupted = Model()
    with pytest.raises(SimulatedCrash):
        run_recipe(
            interrupted, Loader(), tmp_path / 'interrupted',
            0, pending, crash_step=crash_step,
        )

    saved = (
        tmp_path / 'interrupted'
        / f'grpo-gsm8k-checkpoint-{crash_step}'
    )
    resumed, loader = Model(), Loader()
    loader.consumed = 0
    step, remaining = checkpoint.load_rl_checkpoint(
        resumed, loader, saved,
        adapter_name='default',
        config={'mini_batch_size': 2},
    )

    assert step == crash_step
    assert loader.consumed == 8
    if crash_step < 4:
        assert remaining['inputs'] == pending['inputs'][crash_step * 2:]
        assert remaining['old_logps'] == pending['old_logps'][crash_step * 2:]
        assert remaining['advantages'] == pending['advantages'][crash_step * 2:]
    else:
        assert remaining is None

    run_recipe(
        resumed, loader, tmp_path / 'resumed', step, remaining,
    )

    assert resumed.step == baseline.step == 4
    assert resumed.seen == baseline.seen
    assert len(resumed.seen) == sample_count
    assert (
        tmp_path / 'resumed'
        / 'grpo-gsm8k-checkpoint-4-final' / 'rl_state.pkl'
    ).is_file()


def test_failed_model_save_does_not_commit_rl_state(tmp_path):
    model = Model()

    def fail_save(name, output_dir, **kwargs):
        (Path(output_dir) / name).mkdir(parents=True)
        raise RuntimeError('simulated storage failure')

    model.save = fail_save
    with pytest.raises(RuntimeError, match='simulated storage failure'):
        checkpoint.save_rl_checkpoint(
            model, Loader(), name='broken', output_dir=str(tmp_path),
            adapter_name='default', optim_step=0,
            pending=None, config={},
        )

    assert not (tmp_path / 'broken' / 'rl_state.pkl').exists()


@pytest.mark.parametrize(
    'progress',
    [
        {'cur_step': 9, 'consumed_train_samples': 8},
        {'cur_step': 0, 'consumed_train_samples': 99},
    ],
)
def test_inconsistent_progress_is_rejected(tmp_path, progress):
    model, loader = Model(), Loader()
    folder = checkpoint.save_rl_checkpoint(
        model, loader, name='saved', output_dir=str(tmp_path),
        adapter_name='default', optim_step=0,
        pending=None, config={},
    )
    model.resume_from_checkpoint = lambda *args, **kwargs: progress

    with pytest.raises(ValueError):
        checkpoint.load_rl_checkpoint(
            model, loader, folder,
            adapter_name='default', config={},
        )


def test_existing_checkpoint_is_not_overwritten(tmp_path):
    model, loader = Model(), Loader()
    kwargs = dict(
        name='saved', output_dir=str(tmp_path),
        adapter_name='default', optim_step=0,
        pending=None, config={},
    )
    folder = checkpoint.save_rl_checkpoint(model, loader, **kwargs)
    before = (Path(folder) / 'rl_state.pkl').read_bytes()

    with pytest.raises(FileExistsError):
        checkpoint.save_rl_checkpoint(model, loader, **kwargs)

    assert (Path(folder) / 'rl_state.pkl').read_bytes() == before


@pytest.mark.parametrize('crash_step', [1, 2])
def test_restart_across_new_rollout(tmp_path, crash_step):
    import random

    class BatchLoader(Loader):
        def __init__(self):
            self.consumed = 0

        def __iter__(self):
            while self.consumed < 4:
                start = self.consumed
                self.consumed += 2
                yield [start, start + 1]

    class Metrics:
        def reset(self):
            pass

        def accumulate(self, **kwargs):
            pass

        def calculate(self):
            return {'train/total_reward': 1.0}

    def execute(model, loader, folder, step, pending, crash=None):
        seeds = []

        def sample(prompts, params):
            seeds.append(params.seed)
            rng = random.Random(params.seed)
            responses = []
            for prompt in prompts:
                sequences = []
                for _ in range(params.num_samples):
                    token = rng.randrange(1000000)
                    sequences.append(SimpleNamespace(
                        new_input_feature=[prompt, token],
                        logprobs=[[(token, -token / 1000000)]],
                        tokens=[token],
                    ))
                responses.append(SimpleNamespace(sequences=sequences))
            return responses

        def rewards(inputs):
            values = [float(i % 2) for i in range(len(inputs))]
            return values, values, values

        env = {
            'random': random,
            'SamplingParams': SimpleNamespace,
            'NUM_GENERATIONS': 2,
            'MAX_NEW_TOKENS': 8,
            'metrics': Metrics(),
            'ckpt_manager': SimpleNamespace(sync_weights=lambda **k: None),
            'sampler': SimpleNamespace(
                reset_prefix_cache=lambda: None,
                sample=sample,
            ),
            'compute_rewards': rewards,
            'advantage_fn': lambda values, **k: SimpleNamespace(
                tolist=lambda: values,
            ),
        }
        run_recipe(
            model, loader, folder, step, pending,
            crash_step=crash, extra_env=env,
        )
        return seeds

    original_rng = random.getstate()
    try:
        random.seed(2026)
        baseline = Model()
        baseline_seeds = execute(
            baseline, BatchLoader(), tmp_path / 'baseline', 0, None,
        )
        assert len(baseline_seeds) == 2
        assert baseline_seeds[0] != baseline_seeds[1]

        random.seed(2026)
        with pytest.raises(SimulatedCrash):
            execute(
                Model(), BatchLoader(), tmp_path / 'interrupted',
                0, None, crash=crash_step,
            )

        # Simulate unrelated RNG consumption before checkpoint restoration.
        random.seed(999)
        resumed, loader = Model(), BatchLoader()
        saved = (
            tmp_path / 'interrupted'
            / f'grpo-gsm8k-checkpoint-{crash_step}'
        )
        step, pending = checkpoint.load_rl_checkpoint(
            resumed, loader, saved,
            adapter_name='default', config={'mini_batch_size': 2},
        )
        resumed_seeds = execute(
            resumed, loader, tmp_path / 'resumed', step, pending,
        )

        assert resumed_seeds == baseline_seeds[1:]
        assert resumed.seen == baseline.seen
        assert resumed.step == baseline.step == 4
        assert loader.consumed == 4
    finally:
        random.setstate(original_rng)
