from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

from twinkle_client.types import DataRef


MODULE_PATH = (
    Path(__file__).parents[2] / 'cookbook' / 'client' / 'async_rl' / 'client_orchestrated_grpo.py'
)


def _load_module():
    spec = importlib.util.spec_from_file_location('client_orchestrated_grpo', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rollout_and_train_overlap_with_fifo_policy_publication(monkeypatch, capsys) -> None:
    module = _load_module()
    monkeypatch.setattr(module, 'BATCH_SIZE', 2)
    monkeypatch.setattr(module, 'NUM_GENERATIONS', 2)
    monkeypatch.setattr(module, 'TRAIN_MINI_BATCH_SIZE', 2)
    monkeypatch.setattr(module, 'MAX_STALENESS', 1)
    monkeypatch.setattr(module, 'MAX_PARTITIONS', 3)

    first_train_started = asyncio.Event()
    rollout_snapshots = []
    events = []

    async def fake_rollout(_sampler, prompt, policy, _semaphore, _group_id):
        name = prompt['name']
        rollout_snapshots.append((name, policy.version, policy.adapter_uri))
        events.append(f'rollout-start:{name}')
        if name == 'p0-g1':
            await first_train_started.wait()
        events.append(f'rollout-done:{name}')
        return DataRef(
            ref_id=name,
            size=module.NUM_GENERATIONS,
            fields=['train_input', 'sampled_logprobs', 'decoded'],
            kind='rollout',
        )

    monkeypatch.setattr(module, 'rollout_group', fake_rollout)
    monkeypatch.setattr(module, 'GSM8KAccuracyReward', lambda: lambda rows: [1.0] * len(rows))
    monkeypatch.setattr(
        module,
        'GRPOAdvantage',
        lambda: lambda rewards, **_kwargs: [1.0, -1.0],
    )

    class FakeModel:
        def __init__(self):
            self.saved = []
            self.steps = 0
            self.forward_backward_kwargs = []

        async def save(self, name):
            self.saved.append(name)
            return {'twinkle_path': f'/checkpoints/{name}'}

        async def forward_backward_from_data_plane(self, _refs, **kwargs):
            self.forward_backward_kwargs.append(kwargs)
            events.append('train')
            first_train_started.set()

        async def clip_grad_and_step(self, **_kwargs):
            self.steps += 1

        async def calculate_metric(self, **_kwargs):
            return {'result': {'loss': 1.0 / self.steps, 'grad_norm': 0.5}}

    class FakeDataPlane:
        def __init__(self):
            self.released = []

        async def aget(self, ref, *, fields=None):
            assert fields == ['decoded']
            return [{'decoded': f'{ref.ref_id}-{index}'} for index in range(ref.size)]

        async def aappend(self, ref, rows, **_kwargs):
            return ref.model_copy(update={'fields': [*ref.fields, *rows[0]]})

        async def arelease(self, ref):
            self.released.append(ref)

    async def run():
        model = FakeModel()
        data_plane = FakeDataPlane()
        batches = [
            [{'name': 'p0-g0'}, {'name': 'p0-g1'}],
            [{'name': 'p1-g0'}, {'name': 'p1-g1'}],
            [{'name': 'p2-g0'}, {'name': 'p2-g1'}],
        ]
        await module.run_grpo(batches, model, object(), data_plane)
        return model, data_plane

    model, data_plane = asyncio.run(run())

    assert events.index('train') < events.index('rollout-done:p0-g1')
    assert model.saved == ['policy-0', 'policy-1', 'policy-2', 'policy-3']
    assert model.steps == 6
    assert all(kwargs['input_field'] == 'train_input' for kwargs in model.forward_backward_kwargs)
    assert all(kwargs['kwarg_fields'] == {
        'old_logps': 'sampled_logprobs',
        'advantages': 'advantage',
    } for kwargs in model.forward_backward_kwargs)
    assert len(data_plane.released) == 6

    snapshots = {name: (version, uri) for name, version, uri in rollout_snapshots}
    assert snapshots['p0-g0'] == (0, '/checkpoints/policy-0')
    assert snapshots['p1-g0'] == (0, '/checkpoints/policy-0')
    assert snapshots['p2-g0'][0] in (1, 2)
    assert snapshots['p2-g0'][1] == f'/checkpoints/policy-{snapshots["p2-g0"][0]}'
    output = capsys.readouterr().out
    assert 'optimizer_step=1' in output
    assert 'loss=1.0' in output
    assert 'grad_norm=0.5' in output


def test_younger_rollout_failure_stops_admission(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, 'BATCH_SIZE', 1)
    monkeypatch.setattr(module, 'NUM_GENERATIONS', 1)
    monkeypatch.setattr(module, 'TRAIN_MINI_BATCH_SIZE', 1)
    monkeypatch.setattr(module, 'MAX_STALENESS', 1)
    monkeypatch.setattr(module, 'MAX_PARTITIONS', 3)

    started = []

    async def fake_rollout(_sampler, prompt, _policy, _semaphore, _group_id):
        name = prompt['name']
        started.append(name)
        if name == 'p1':
            raise RuntimeError('rollout failed')
        return DataRef(
            ref_id=name,
            size=1,
            fields=['train_input', 'sampled_logprobs', 'decoded'],
            kind='rollout',
        )

    monkeypatch.setattr(module, 'rollout_group', fake_rollout)
    monkeypatch.setattr(module, 'GSM8KAccuracyReward', lambda: lambda rows: [1.0])
    monkeypatch.setattr(module, 'GRPOAdvantage', lambda: lambda rewards, **_kwargs: [1.0])

    class FakeModel:
        def __init__(self):
            self.saved = []

        async def save(self, name):
            self.saved.append(name)
            return {'twinkle_path': name}

        async def forward_backward(self, _ref, **_kwargs):
            return None

        async def clip_grad_and_step(self, **_kwargs):
            return None

        async def calculate_metric(self, **_kwargs):
            return {'result': {'loss': 1.0}}

    class FakeDataPlane:
        async def aget(self, ref, *, fields=None):
            assert fields == ['decoded']
            return [{'decoded': ref.ref_id}]

        async def aappend(self, ref, rows, **_kwargs):
            return ref.model_copy(update={'fields': [*ref.fields, *rows[0]]})

        async def arelease(self, _ref):
            return None

    async def run():
        model = FakeModel()
        try:
            await module.run_grpo(
                [[{'name': 'p0'}], [{'name': 'p1'}], [{'name': 'p2'}]],
                model,
                object(),
                FakeDataPlane(),
            )
        except RuntimeError as error:
            assert str(error) == 'rollout failed'
        else:
            raise AssertionError('expected the younger rollout failure')
        return model

    model = asyncio.run(run())
    assert started == ['p0', 'p1']
    assert model.saved == ['policy-0']
