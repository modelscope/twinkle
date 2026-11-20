# Copyright (c) ModelScope Contributors. All rights reserved.
import traceback
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

import twinkle_client.operations as operation_module
from twinkle_client.resources import RolloutResources


class Environment:
    def __init__(self, events, name='env', *, close_error=None):
        self.events, self.name, self.close_error = events, name, close_error

    def shutdown(self):
        self.events.append(f'{self.name}.shutdown')
        if self.close_error:
            raise self.close_error


class Rollout:
    def __init__(self, events, *, close_error=None):
        self.events, self.close_error = events, close_error

    def close(self):
        self.events.append('rollout.close')
        if self.close_error:
            raise self.close_error


def scope_for(events):
    client = SimpleNamespace(close=lambda: events.append('client.close'))
    return RolloutResources(lambda: client)


def interrupt_next_wait(monkeypatch):
    class InterruptedWait(Future):
        remaining = 2

        def result(self):
            if self.remaining:
                self.remaining -= 1
                raise KeyboardInterrupt('caller interrupted')
            return super().result()

    def next_outcome():
        monkeypatch.setattr(operation_module, 'Future', Future)
        return InterruptedWait()

    monkeypatch.setattr(operation_module, 'Future', next_outcome)


def test_scope_closes_rollout_before_environments_and_client():
    events = []
    with scope_for(events) as resources:
        resources.own_rollout(Rollout(events))
        resources.own_environment(Environment(events, 'a'))
        resources.own_environment(Environment(events, 'b'))
    resources.close()
    assert events == ['rollout.close', 'a.shutdown', 'b.shutdown', 'client.close']
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
    assert events == ['env.shutdown', 'client.close']


def note_unconfirmed(resources):
    """Drive the real settlement path: a managed remote op that sent a request then lost its outcome."""
    def operation():
        resources._operation_scope.note_request('/train/step')
        raise ConnectionError('lost server')

    with pytest.raises(ConnectionError):
        resources._operation_scope.run(operation)


def test_unconfirmed_operation_retains_environments_and_client():
    events = []
    primary = ConnectionError('lost server')
    with pytest.raises(ConnectionError) as caught:
        with scope_for(events) as resources:
            resources.own_rollout(Rollout(events))
            env = resources.own_environment(Environment(events))

            def request():
                resources._operation_scope.note_request('/forward_backward')
                raise primary

            resources._operation_scope.run(request)
    assert caught.value is primary
    assert events == ['rollout.close']
    assert resources.client is not None and resources._environments == {id(env): env}
    assert 'environments and client retained' in '\n'.join(primary.__notes__)


def test_unconfirmed_remote_settlement_retains_environments_and_client():
    events = []
    with pytest.raises(BaseExceptionGroup, match='resource cleanup') as caught:
        with scope_for(events) as resources:
            resources.own_rollout(Rollout(events))
            env = resources.own_environment(Environment(events))
            note_unconfirmed(resources)
    assert events == ['rollout.close']
    assert resources.client is not None and resources._environments == {id(env): env}
    assert 'environments and client retained' in ''.join(traceback.format_exception(caught.value))


@pytest.mark.parametrize('failure', [RuntimeError('close failed'), KeyboardInterrupt('worker interrupted')])
def test_failed_close_does_not_skip_other_resources_or_replace_primary(failure):
    events, primary = [], ValueError('training failed')
    with pytest.raises(ValueError) as caught:
        with scope_for(events) as resources:
            bad = resources.own_environment(Environment(events, 'bad', close_error=failure))
            resources.own_environment(Environment(events, 'good'))
            raise primary
    assert caught.value is primary
    assert events == ['bad.shutdown', 'good.shutdown', 'client.close']
    assert resources._environments == {id(bad): bad}
    assert str(failure) in '\n'.join(primary.__notes__)
    bad.close_error = None
    resources.close()
    assert not resources._environments


def test_initialization_interrupt_still_closes_returned_client(monkeypatch):
    events = []
    resources = scope_for(events)
    interrupt_next_wait(monkeypatch)
    with pytest.raises(KeyboardInterrupt, match='caller interrupted'):
        with resources:
            pytest.fail('interrupted initialization must not enter the body')
    assert events == ['client.close']
    assert resources.client is None


def test_repeated_cleanup_interrupts_wait_for_every_release(monkeypatch):
    events = []
    resources = scope_for(events)
    with pytest.raises(BaseExceptionGroup, match='resource cleanup'):
        with resources:
            resources.own_rollout(Rollout(events))
            resources.own_environment(Environment(events, 'a'))
            resources.own_environment(Environment(events, 'b'))
            interrupt_next_wait(monkeypatch)
    assert events == ['rollout.close', 'a.shutdown', 'b.shutdown', 'client.close']
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
    assert events == ['env.shutdown']
    assert resources.client is client
    client.close = lambda: events.append('client.close')
    resources.close()
    assert events[-1] == 'client.close' and resources.client is None


def test_failed_driver_close_retains_dependent_resources_until_retry():
    events = []
    resources = scope_for(events)
    with pytest.raises(BaseExceptionGroup, match='resource cleanup'):
        with resources:
            resources.own_rollout(Rollout(events))
            driver = resources.own_rollout(Rollout(events, close_error=RuntimeError('still running')))
            resources.own_environment(Environment(events))
    assert events == ['rollout.close']
    assert resources.client is not None
    driver.close_error = None
    resources.close()
    assert events == ['rollout.close', 'rollout.close', 'rollout.close', 'env.shutdown', 'client.close']


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
        assert a == ['rollout.close', 'env.shutdown', 'client.close']
        assert b == []
        assert other.operations.run(lambda: 'still usable') == 'still usable'
    assert b == ['rollout.close', 'env.shutdown', 'client.close']
