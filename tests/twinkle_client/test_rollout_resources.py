# Copyright (c) ModelScope Contributors. All rights reserved.
from types import SimpleNamespace

import pytest

import twinkle_client.resources as resource_module
from twinkle_client.resources import RolloutResources


class Environment:
    def __init__(self, events, name='env', *, busy=False, close_error=None):
        self.events, self.name = events, name
        self.busy, self.close_error = busy, close_error

    def ensure_idle(self):
        self.events.append(f'{self.name}.idle')
        if self.busy:
            raise RuntimeError('remote command is unconfirmed')

    def close(self):
        self.events.append(f'{self.name}.close')
        if self.close_error:
            raise self.close_error


class Rollout:
    def __init__(self, events, *, idle=True, drain_error=None):
        self.events, self.is_idle, self.drain_error = events, idle, drain_error

    def drain(self):
        self.events.append('drain')
        if self.drain_error:
            raise self.drain_error


def scope_for(events):
    client = SimpleNamespace(close=lambda: events.append('client.close'))
    return RolloutResources(lambda: client)


def interrupt_waits(operations):
    original = operations._executor.submit

    class Wait:
        def __init__(self, future):
            self.future, self.remaining = future, 2

        def result(self):
            if self.remaining:
                self.remaining -= 1
                raise KeyboardInterrupt('caller interrupted')
            return self.future.result()

    operations._executor.submit = lambda *a, **k: Wait(original(*a, **k))


def test_scope_drains_before_release_and_close_is_idempotent():
    events = []
    with scope_for(events) as resources:
        resources.own_rollout(Rollout(events))
        early = resources.own_environment(Environment(events, 'collected'))
        resources.own_environment(Environment(events, 'pending'))
        resources.close_environment(early)
    resources.close()
    assert events == ['collected.idle', 'collected.close', 'drain',
                      'pending.idle', 'pending.close', 'client.close']
    assert resources.client is None and not resources._environments
    with pytest.raises(RuntimeError, match='cannot be entered again'):
        with resources:
            pass


def test_partial_environment_setup_and_primary_error_are_preserved():
    events, primary = [], RuntimeError('reset failed')
    with pytest.raises(RuntimeError) as caught:
        with scope_for(events) as resources:
            resources.own_environment(Environment(events))
            raise primary
    assert caught.value is primary
    assert events == ['env.idle', 'env.close', 'client.close']


@pytest.mark.parametrize('primary', [None, RuntimeError('training failed')])
def test_unconfirmed_rollout_retains_all_resources(primary):
    events = []
    with pytest.raises(BaseException) as caught:
        with scope_for(events) as resources:
            resources.own_rollout(Rollout(events, idle=False, drain_error=ConnectionError('lost server')))
            env = resources.own_environment(Environment(events))
            if primary is not None:
                raise primary
    assert events == ['drain']
    assert resources.client is not None and resources._environments == {id(env): env}
    if primary is not None:
        assert caught.value is primary
        assert 'environments and client retained' in '\n'.join(primary.__notes__)
    else:
        assert isinstance(caught.value, BaseExceptionGroup)


@pytest.mark.parametrize('failure', [RuntimeError('close failed'), KeyboardInterrupt('worker interrupted')])
def test_failed_close_does_not_skip_other_resources_or_replace_primary(failure):
    events, primary = [], ValueError('training failed')
    with pytest.raises(ValueError) as caught:
        with scope_for(events) as resources:
            bad = resources.own_environment(Environment(events, 'bad', close_error=failure))
            resources.own_environment(Environment(events, 'good'))
            raise primary
    assert caught.value is primary
    assert events == ['bad.idle', 'bad.close', 'good.idle', 'good.close', 'client.close']
    assert resources._environments == {id(bad): bad}
    assert str(failure) in '\n'.join(primary.__notes__)
    bad.close_error = None
    resources.close()
    assert not resources._environments


def test_unconfirmed_environment_is_not_closed_but_independent_resources_are():
    events = []
    with pytest.raises(BaseExceptionGroup, match='resource cleanup'):
        with scope_for(events) as resources:
            env = resources.own_environment(Environment(events, busy=True))
            resources.own_environment(Environment(events, 'other'))
    assert events == ['env.idle', 'other.idle', 'other.close', 'client.close']
    assert resources._environments == {id(env): env}


def test_initialization_interrupt_still_closes_returned_client():
    events = []
    resources = scope_for(events)
    interrupt_waits(resources.operations)
    with pytest.raises(KeyboardInterrupt, match='caller interrupted'):
        with resources:
            pytest.fail('interrupted initialization must not enter the body')
    assert events == ['client.close']
    assert resources.client is None


def test_repeated_cleanup_interrupts_wait_for_every_release(monkeypatch):
    events = []
    resources = scope_for(events)
    original = resource_module.SequentialOperations

    def cleanup_operations():
        operations = original()
        interrupt_waits(operations)
        return operations

    with pytest.raises(BaseExceptionGroup, match='resource cleanup'):
        with resources:
            resources.own_rollout(Rollout(events))
            resources.own_environment(Environment(events, 'a'))
            resources.own_environment(Environment(events, 'b'))
            monkeypatch.setattr(resource_module, 'SequentialOperations', cleanup_operations)
    assert events == ['drain', 'a.idle', 'a.close', 'b.idle', 'b.close', 'client.close']
    assert resources.client is None and not resources._environments


def test_client_initialization_failure_shuts_down_operations():
    failure = RuntimeError('client initialization failed')

    def fail():
        raise failure

    resources = RolloutResources(fail)
    with pytest.raises(RuntimeError) as caught:
        with resources:
            pass
    assert caught.value is failure
    assert resources.operations._stopped and resources.operations._executor._shutdown


def test_client_close_failure_retains_client_for_retry():
    events = []
    with pytest.raises(BaseExceptionGroup, match='resource cleanup'):
        with scope_for(events) as resources:
            client = resources.client
            client.close = lambda: (_ for _ in ()).throw(RuntimeError('transport close failed'))
            resources.own_environment(Environment(events))
    assert events == ['env.idle', 'env.close']
    assert resources.client is client
    client.close = lambda: events.append('client.close')
    resources.close()
    assert events[-1] == 'client.close' and resources.client is None


def test_early_release_failure_is_retained_until_scope_exit():
    events = []
    with scope_for(events) as resources:
        env = resources.own_environment(Environment(events, close_error=RuntimeError('transient failure')))
        with pytest.raises(RuntimeError, match='transient failure'):
            resources.close_environment(env)
        assert resources._environments == {id(env): env}
        env.close_error = None
    assert events == ['env.idle', 'env.close', 'env.idle', 'env.close', 'client.close']


def test_draining_error_with_confirmed_idle_still_closes_resources():
    events = []
    with pytest.raises(BaseExceptionGroup, match='resource cleanup'):
        with scope_for(events) as resources:
            resources.own_rollout(Rollout(events, drain_error=RuntimeError('cleanup diagnostic')))
            resources.own_environment(Environment(events))
    assert events == ['drain', 'env.idle', 'env.close', 'client.close']


def test_scopes_do_not_release_another_clients_resources():
    a, b = [], []
    with scope_for(b) as other:
        other.own_rollout(Rollout(b))
        other.own_environment(Environment(b))
        with pytest.raises(KeyboardInterrupt):
            with scope_for(a) as cancelled:
                cancelled.own_rollout(Rollout(a))
                cancelled.own_environment(Environment(a))
                raise KeyboardInterrupt()
        assert a == ['drain', 'env.idle', 'env.close', 'client.close']
        assert b == []
        assert other.operations.run(lambda: 'still usable') == 'still usable'
    assert b == ['drain', 'env.idle', 'env.close', 'client.close']
