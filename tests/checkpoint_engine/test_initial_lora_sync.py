from types import SimpleNamespace

import pytest

from twinkle.checkpoint_engine.manager import CheckpointEngineManager


def make_manager(config, fail_adapter=False):
    events = []

    def send_weights(**kwargs):
        phase = kwargs['base_sync_done']
        events.append(('send', phase))

        def wait():
            events.append(('send_done', phase))

        return wait

    def receive_weights(**kwargs):
        phase = kwargs['base_sync_done']
        events.append(('receive', phase, kwargs['peft_config']))

        def wait():
            if phase and fail_adapter:
                raise RuntimeError('adapter load failed')
            events.append(('receive_done', phase))

        return wait

    model = SimpleNamespace(
        device_mesh=SimpleNamespace(world_size=1),
        prepare_checkpoint_engine=lambda *args: {},
        init_checkpoint_process_group=lambda **kwargs: lambda: None,
        get_peft_config_dict=lambda: config,
        send_weights=send_weights,
        finalize_checkpoint_engine=lambda: events.append(('model_finalize',)),
    )
    sampler = SimpleNamespace(
        device_mesh=SimpleNamespace(data_world_size=1),
        prepare_checkpoint_engine=lambda *args: None,
        init_checkpoint_process_group=lambda **kwargs: lambda: None,
        receive_weights=receive_weights,
        finalize_checkpoint_engine=lambda: events.append(('sampler_finalize',)),
    )
    manager = object.__new__(CheckpointEngineManager)
    manager.model = model
    manager.sampler = sampler
    manager.backend_cls = SimpleNamespace(
        build_topology=lambda *args: ({}, {}),
    )
    manager.base_sync_done = False
    manager._peft_config = None
    manager._model_keys = []
    return manager, events


def test_initial_sync_loads_current_adapter_before_returning():
    config = {'r': 32}
    manager, events = make_manager(config)
    manager.sync_weights(merge_and_sync=False)

    assert events == [
        ('send', False),
        ('receive', False, None),
        ('send_done', False),
        ('receive_done', False),
        ('send', True),
        ('receive', True, config),
        ('send_done', True),
        ('receive_done', True),
        ('model_finalize',),
        ('sampler_finalize',),
    ]
    assert manager.base_sync_done

    events.clear()
    manager.sync_weights(merge_and_sync=False)
    assert [event for event in events if event[0] == 'send'] == [
        ('send', True),
    ]
    assert ('receive', True, config) in events


@pytest.mark.parametrize(
    'merge_and_sync, config',
    [(True, {'r': 32}), (False, None)],
)
def test_initial_sync_without_separate_adapter_has_one_phase(
    merge_and_sync, config
):
    manager, events = make_manager(config)
    manager.sync_weights(merge_and_sync=merge_and_sync)
    assert [event for event in events if event[0] == 'send'] == [
        ('send', False),
    ]
    assert ('receive', False, None) in events


def test_failed_adapter_load_does_not_mark_initial_sync_complete():
    manager, events = make_manager({'r': 32}, fail_adapter=True)
    with pytest.raises(RuntimeError, match='adapter load failed'):
        manager.sync_weights(merge_and_sync=False)
    assert not manager.base_sync_done
