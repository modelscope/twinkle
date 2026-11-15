# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU-only tests for CheckpointEngineManager mode selection and direct sync."""

import asyncio

import pytest

from twinkle.checkpoint_engine.manager import CheckpointEngineManager


class _Mesh:
    world_size = 1
    data_world_size = 1


class _Model:

    def __init__(self, weights):
        self.device_mesh = _Mesh()
        self._weights = weights
        self.generator_calls = []
        self.peft_config_calls = 0
        self._checkpoint_engine = None

    def _get_weight_generator(self, **kwargs):
        self.generator_calls.append(kwargs)
        weights = self._weights(kwargs) if callable(self._weights) else self._weights

        def _weights():
            yield from weights

        return _weights()

    def get_peft_config_dict(self):
        self.peft_config_calls += 1
        return {'r': 8, 'target_modules': ['q_proj']}


class _Sampler:

    def __init__(self, fail=None):
        self.device_mesh = _Mesh()
        self.calls = []
        self.loaded = []
        self.fail = fail
        self._checkpoint_engine = None
        # One timeline for the gate and the transfer, so "weights are swapped while generation is held"
        # is assertable as an ordering rather than implied by two separate lists happening to be
        # non-empty. The manager's contract is pause -> transfer -> resume.
        self.gate = []

    def get_state_keys(self):
        return ['q_proj.weight']

    def pause_generation(self) -> None:
        self.gate.append('pause')

    def resume_generation(self) -> None:
        self.gate.append('resume')

    def receive_weights(self, **kwargs):
        self.calls.append(kwargs)
        self.gate.append('receive')
        if self.fail is not None:
            raise self.fail
        self.loaded.append(list(kwargs['weights']))


@pytest.mark.parametrize('requested_mode', ['auto', 'naive'])
def test_local_sync_streams_generator_without_checkpoint_engine(requested_mode):
    model = _Model([('q_proj.weight', 'base')])
    sampler = _Sampler()

    manager = CheckpointEngineManager(model, sampler, platform='CPU', mode=requested_mode)
    manager.sync_weights(merge_and_sync=False)

    assert manager.requested_mode == requested_mode
    assert manager.mode == 'naive'
    assert manager.backend_cls is None
    assert manager.base_sync_done is True
    assert sampler.loaded == [[('q_proj.weight', 'base')]]
    # Generation is held across the transfer and released after, so requests in flight continue on the
    # fresh weights rather than being torn down by the swap.
    assert sampler.gate == ['pause', 'receive', 'resume']
    assert model._checkpoint_engine is None
    assert sampler._checkpoint_engine is None
    assert model.generator_calls == [{
        'base_sync_done': False,
        'merge_and_sync': False,
        'model_keys': ['q_proj.weight'],
    }]


def test_local_lora_sync_reuses_peft_config_and_sends_only_incremental_weights():
    model = _Model(lambda call: ([('q_proj.lora_A', 'adapter')]
                                 if call['base_sync_done'] else [('q_proj.weight', 'base')]))
    sampler = _Sampler()
    manager = CheckpointEngineManager(model, sampler, platform='CPU')

    manager.sync_weights(merge_and_sync=False)
    manager.sync_weights(merge_and_sync=False)

    assert sampler.loaded == [[('q_proj.weight', 'base')], [('q_proj.lora_A', 'adapter')]]
    assert model.peft_config_calls == 1
    assert sampler.calls[0]['peft_config'] is None
    assert sampler.calls[1]['peft_config'] == {'r': 8, 'target_modules': ['q_proj']}
    assert sampler.calls[0]['base_sync_done'] is False
    assert sampler.calls[1]['base_sync_done'] is True
    assert model.generator_calls[1]['base_sync_done'] is True


def test_local_merge_sync_generates_a_full_weight_set_each_time():
    model = _Model([('q_proj.weight', 'merged')])
    sampler = _Sampler()
    manager = CheckpointEngineManager(model, sampler, platform='CPU')

    manager.sync_weights(merge_and_sync=True)
    manager.sync_weights(merge_and_sync=True)

    assert sampler.loaded == [[('q_proj.weight', 'merged')], [('q_proj.weight', 'merged')]]
    assert all(call['merge_and_sync'] is True for call in model.generator_calls)
    assert [call['base_sync_done'] for call in model.generator_calls] == [False, True]


def test_local_failure_does_not_mark_base_sync_done():
    model = _Model([('q_proj.weight', 'base')])
    error = RuntimeError('sampler failed')
    sampler = _Sampler(fail=error)
    manager = CheckpointEngineManager(model, sampler, platform='CPU')

    with pytest.raises(RuntimeError, match='sampler failed') as exc_info:
        manager.sync_weights(merge_and_sync=False)

    assert exc_info.value is error
    assert manager.base_sync_done is False

    sampler.fail = None
    manager.sync_weights(merge_and_sync=False)
    assert model.generator_calls[-1]['base_sync_done'] is False


def test_local_weight_generator_failure_does_not_mark_base_sync_done():
    error = ValueError('weight generation failed')

    def broken_weights():
        yield 'q_proj.weight', 'base'
        raise error

    model = _Model(broken_weights())
    sampler = _Sampler()
    manager = CheckpointEngineManager(model, sampler, platform='CPU')

    with pytest.raises(ValueError, match='weight generation failed') as exc_info:
        manager.sync_weights(merge_and_sync=False)

    assert exc_info.value is error
    assert manager.base_sync_done is False


def test_mixed_deployment_shape_fails_at_initialization():
    model = _Model([])
    sampler = _Sampler()
    sampler._actors = [object()]

    with pytest.raises(ValueError, match='same deployment shape'):
        CheckpointEngineManager(model, sampler, platform='CPU')


@pytest.mark.parametrize(
    ('mode', 'use_actors', 'match'),
    [
        ('naive', True, "mode='naive' requires local"),
        ('colocate', False, "mode='colocate' requires"),
        ('standalone', False, "mode='standalone' requires"),
        ('unknown', False, 'Unknown checkpoint engine mode'),
    ],
)
def test_explicit_mode_validates_deployment_shape(mode, use_actors, match):
    model = _Model([])
    sampler = _Sampler()
    if use_actors:
        model._actors = [object()]
        sampler._actors = [object()]

    with pytest.raises(ValueError, match=match):
        CheckpointEngineManager(model, sampler, platform='CPU', mode=mode)


def test_backend_selection_uses_resolved_mode():
    from twinkle.checkpoint_engine import IPCCheckpointEngine, NCCLCheckpointEngine

    assert CheckpointEngineManager.decide_backend_engine('GPU', mode='naive') is None
    assert CheckpointEngineManager.decide_backend_engine('GPU', mode='colocate') is IPCCheckpointEngine
    assert CheckpointEngineManager.decide_backend_engine('GPU', mode='standalone') is NCCLCheckpointEngine


@pytest.mark.parametrize('sampler_backend', ['vllm', 'sglang'])
@pytest.mark.parametrize('provide_weights', [True, False])
def test_sampler_receive_weights_selects_direct_or_checkpoint_stream(sampler_backend, provide_weights):
    if sampler_backend == 'vllm':
        from twinkle.sampler.vllm_sampler.vllm_sampler import vLLMSampler as sampler_cls
    else:
        from twinkle.sampler.sglang_sampler.sglang_sampler import SGLangSampler as sampler_cls

    class _InferenceEngine:

        def __init__(self):
            self.loaded = None
            self.invalidated = False

        async def update_weights(self, weights, **kwargs):
            self.loaded = (list(weights), kwargs)

        def invalidate_synced_lora(self):
            self.invalidated = True

    class _CheckpointEngine:

        def receive_weights(self):
            return iter([('checkpoint.weight', 'checkpoint')])

    sampler = object.__new__(sampler_cls)
    sampler.engine = _InferenceEngine()
    sampler._run_in_loop = asyncio.run
    checkpoint_engine = _CheckpointEngine()
    checkpoint_engine_calls = []

    def get_checkpoint_engine():
        checkpoint_engine_calls.append(True)
        return checkpoint_engine

    sampler._get_or_create_checkpoint_engine = get_checkpoint_engine
    direct_weights = iter([('direct.weight', 'direct')]) if provide_weights else None

    sampler_cls.receive_weights.__wrapped__(sampler, weights=direct_weights)

    expected_weights = [('direct.weight', 'direct')] if provide_weights else [('checkpoint.weight', 'checkpoint')]
    assert sampler.engine.loaded == (expected_weights, {'peft_config': None, 'base_sync_done': False})
    assert len(checkpoint_engine_calls) == (0 if provide_weights else 1)
    assert sampler.engine.invalidated is (sampler_backend == 'vllm')


def test_auto_ray_actor_sync_keeps_standalone_checkpoint_engine_lifecycle(monkeypatch):
    events = []

    class _Backend:
        # The manager reads this off the backend class to sign which engine the channel runs on.
        backend_name = 'nccl'

        @classmethod
        def build_topology(cls, trainer_world_size, rollout_world_size, metadata):
            events.append(('build_topology', trainer_world_size, rollout_world_size))
            return ({'rank': [0], 'world_size': [2], 'master_metadata': [metadata[0]]},
                    {'rank': [1], 'world_size': [2], 'master_metadata': [metadata[0]]})

    class _ActorModel(_Model):

        def __init__(self):
            super().__init__([('q_proj.weight', 'base')])
            self._actors = [object()]

        def prepare_checkpoint_engine(self, is_master):
            events.append(('model_prepare', is_master))
            return {'zmq_ip': '127.0.0.1', 'zmq_port': 1}

        def init_checkpoint_process_group(self, **kwargs):
            events.append(('model_init_submitted', kwargs))
            return lambda: events.append(('model_init_waited', kwargs))

        def activate_checkpoint_engine(self, backend, channel, auth_key=None):
            pass

        def send_weights(self, **kwargs):
            events.append(('send_submitted', kwargs))
            return lambda: events.append(('send_waited', kwargs))

        def finalize_checkpoint_engine(self):
            events.append('model_finalize')

    class _ActorSampler(_Sampler):

        def __init__(self):
            super().__init__()
            self._actors = [object()]

        def prepare_checkpoint_engine(self, is_master):
            events.append(('sampler_prepare', is_master))

        def init_checkpoint_process_group(self, **kwargs):
            events.append(('sampler_init_submitted', kwargs))
            return lambda: events.append(('sampler_init_waited', kwargs))

        def activate_checkpoint_engine(self, backend, channel, auth_key=None):
            # The receiver half reports the replicas the channel was opened for. Not recorded in
            # ``events``: this case is about the standalone prepare/init/receive/finalize lifecycle, and
            # which engine the actors were configured with is what the colocate case asserts.
            return ['member']

        def receive_weights(self, **kwargs):
            events.append(('receive_submitted', kwargs))
            return lambda: events.append(('receive_waited', kwargs))

        def finalize_checkpoint_engine(self):
            events.append('sampler_finalize')

    model = _ActorModel()
    sampler = _ActorSampler()

    def decide_backend(platform=None, mode='standalone'):
        assert mode == 'standalone'
        return _Backend

    monkeypatch.setattr(CheckpointEngineManager, 'decide_backend_engine', staticmethod(decide_backend))

    manager = CheckpointEngineManager(model, sampler, platform='GPU')
    manager.sync_weights()

    assert manager.requested_mode == 'auto'
    assert manager.mode == 'standalone'
    assert manager.backend == 'nccl'
    # The two rendezvous pairs run on their own threads through ``_together``, which submits both
    # calls before waiting on either -- but nothing orders the two threads against each other, so one
    # may submit and finish waiting before the other submits at all. What the product guarantees is
    # which events exist, the order of the calls it makes serially, and that each half of a rendezvous
    # is submitted before it is waited on. Asserting a total order over the concurrent pairs asserted
    # one interleaving out of several legal ones, and failed whenever the pool scheduled the other.
    names = [event if isinstance(event, str) else event[0] for event in events]
    assert sorted(names) == sorted([
        'model_prepare', 'sampler_prepare', 'build_topology',
        'model_init_submitted', 'sampler_init_submitted',
        'model_init_waited', 'sampler_init_waited',
        'send_submitted', 'receive_submitted', 'send_waited', 'receive_waited',
        'model_finalize', 'sampler_finalize',
    ])
    position = {name: index for index, name in enumerate(names)}
    # Serial in the product: both sides prepare, then the topology is built, then the rendezvous begin.
    assert position['model_prepare'] < position['sampler_prepare'] < position['build_topology']
    for half in ('model_init', 'sampler_init', 'send', 'receive'):
        assert position[f'{half}_submitted'] < position[f'{half}_waited']
    assert position['build_topology'] < min(position['model_init_submitted'],
                                           position['sampler_init_submitted'])
    assert max(position['model_init_waited'], position['sampler_init_waited']) < min(
        position['send_submitted'], position['receive_submitted'])
    assert max(position['send_waited'], position['receive_waited']) < position['model_finalize']
    assert position['model_finalize'] < position['sampler_finalize']
    receive_kwargs = next(
        event[1]
        for event in events
        if isinstance(event, tuple) and event[0] == 'receive_submitted'
    )
    assert 'weights' not in receive_kwargs
    assert manager.base_sync_done is True


def test_colocate_mode_configures_actor_backends(monkeypatch):
    configured = []
    # A class rather than a bare sentinel: the manager reads ``backend_cls.backend_name`` to record
    # which engine the channel runs on, and the assertions below still need the identity preserved.
    backend = type('backend', (), {'backend_name': 'ipc'})

    class _ActorRole:
        _actors = [object()]

        def activate_checkpoint_engine(self, backend, channel, auth_key=None):
            # This is how the manager tells each actor which engine its channel runs on; the model side
            # ignores the return, the receiver side reports the replicas the channel was opened for.
            configured.append(backend)
            return ['member']

    def decide_backend(platform=None, mode='standalone'):
        assert platform == 'GPU'
        assert mode == 'colocate'
        return backend

    monkeypatch.setattr(CheckpointEngineManager, 'decide_backend_engine', staticmethod(decide_backend))

    manager = CheckpointEngineManager(_ActorRole(), _ActorRole(), platform='GPU', mode='colocate')

    assert manager.mode == 'colocate'
    assert manager.backend_cls is backend
    assert manager.backend == 'ipc'
    assert configured == ['ipc', 'ipc']
