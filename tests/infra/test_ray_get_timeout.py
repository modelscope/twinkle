# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for the Sync_Dispatch_Path time bound.

These exercise only ``twinkle.infra`` against a plain sleeping Ray actor. They
depend on neither GPU, Megatron, nor any ``src/twinkle/server/**`` component.
"""
from __future__ import annotations

import pytest

ray = pytest.importorskip('ray')

import twinkle.infra as infra  # noqa: E402
from twinkle.infra import remote_function  # noqa: E402
from twinkle.infra._ray.ray_helper import RayHelper  # noqa: E402


@ray.remote
class _Sleeper:
    """A plain Ray actor whose only method sleeps for a caller-supplied time."""

    def slow(self, seconds: float):
        import time
        time.sleep(seconds)
        return seconds

    def slow_batch(self, seconds: list[float]):
        import time
        time.sleep(seconds[0])
        return seconds

    def _twinkle_async_slow_batch(self, seconds: list[float]):
        return self.slow_batch(seconds)


@pytest.fixture(scope='module', autouse=True)
def _ray_and_ray_mode():
    """Bring up Ray and put infra into 'ray' mode for the driver-side path."""
    ray.init(ignore_reinit_error=True, num_cpus=2, logging_level='ERROR')
    prev_mode = infra._mode
    infra._mode = 'ray'
    try:
        yield
    finally:
        infra._mode = prev_mode


def _make_driver():
    """A minimal stand-in for a remote_class handle: one actor, no concurrency."""
    driver = type('Driver', (), {})()
    driver._actors = [_Sleeper.remote()]
    driver._max_concurrency = None
    return driver


def test_execute_all_sync_times_out(_ray_and_ray_mode):
    """``execute_all_sync(timeout=)`` raises when the remote does not return in time."""
    actor = _Sleeper.remote()
    workers_and_args = [(actor, [3.0], {})]
    with pytest.raises(ray.exceptions.GetTimeoutError):
        RayHelper.execute_all_sync('slow', workers_and_args, timeout=0.5)


def test_execute_all_sync_returns_within_timeout(_ray_and_ray_mode):
    actor = _Sleeper.remote()
    workers_and_args = [(actor, [0.1], {})]
    assert RayHelper.execute_all_sync('slow', workers_and_args, timeout=10.0) == [0.1]


def test_decorator_timeout_takes_priority_over_instance():
    """A small decorator timeout wins over a large instance ``_ray_get_timeout``."""

    def slow(self, seconds):  # body runs in the worker, not here
        return seconds

    wrapped = remote_function(dispatch='all', collect='first', sync=True, timeout=0.5)(slow)
    driver = _make_driver()
    driver._ray_get_timeout = 100.0  # would allow the call if it were consulted
    with pytest.raises(ray.exceptions.GetTimeoutError):
        wrapped(driver, 3.0)


def test_decorator_timeout_wins_when_larger_than_instance():
    """The decorator value wins even when it is the *larger* of the two.

    A large decorator timeout with a tiny instance value must NOT time out --
    proving the instance value is ignored when the decorator declares one.
    """

    def slow(self, seconds):
        return seconds

    wrapped = remote_function(dispatch='all', collect='first', sync=True, timeout=100.0)(slow)
    driver = _make_driver()
    driver._ray_get_timeout = 0.3  # would time out if it were consulted
    result = wrapped(driver, 1.0)
    # sync collect may hand back a lazy-collect callable; resolving it must not time out.
    assert (result() if callable(result) else result) == 1.0


def test_instance_timeout_is_fallback_when_decorator_absent():
    """With no decorator timeout, the instance ``_ray_get_timeout`` applies."""

    def slow(self, seconds):
        return seconds

    wrapped = remote_function(dispatch='all', collect='first', sync=True)(slow)
    driver = _make_driver()
    driver._ray_get_timeout = 0.3
    with pytest.raises(ray.exceptions.GetTimeoutError):
        wrapped(driver, 2.0)


def test_continuous_work_timeout_zero_is_not_treated_as_falsy():

    def slow_batch(self, seconds):
        return seconds

    wrapped = remote_function(
        dispatch='all', collect='first', timeout=0, enable_continous_work=True)(slow_batch)
    driver = _make_driver()
    driver._ray_get_timeout = 100.0
    with pytest.raises(ray.exceptions.GetTimeoutError):
        wrapped(driver, [1.0])


def test_decorator_timeout_zero_is_not_treated_as_falsy():
    """Timeout=0 means 'time out immediately', not 'fall back to unbounded'."""

    def slow(self, seconds):
        return seconds

    wrapped = remote_function(dispatch='all', collect='first', sync=True, timeout=0)(slow)
    driver = _make_driver()
    driver._ray_get_timeout = 100.0  # the old ``or`` bug would fall back here
    with pytest.raises(ray.exceptions.GetTimeoutError):
        wrapped(driver, 1.0)
