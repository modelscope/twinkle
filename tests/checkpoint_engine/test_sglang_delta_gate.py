# Copyright (c) ModelScope Contributors. All rights reserved.
"""SGLang delta lane: KV release must drain generation first (release_memory_occupation asserts idle).

SGLang serializes ``update_weights_from_tensor`` against generation inside the engine, but
``sleep(level=1)`` goes through ``release_memory_occupation``, whose scheduler handler asserts a
fully idle server and kills the worker otherwise. These tests pin the drain-before-release
contract of ``begin_delta_sync`` against the real base-class generation gate, including a
single-point reverse control (remove the pause -> the idle assert fires with a request in flight).
"""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from twinkle.checkpoint_engine.delta.rollout.sglang_mixin import SGLangDeltaSamplerMixin
from twinkle.sampler.base import Sampler


class LocalSGLangSampler(SGLangDeltaSamplerMixin, Sampler):
    """Concrete sampler on the real base gate; coroutines run inline like the product loop driver."""

    def sample(self, *args, **kwargs):
        raise NotImplementedError

    def apply_patch(self, *args, **kwargs):
        raise NotImplementedError

    def _run_in_loop(self, coroutine):
        return asyncio.run(coroutine)


def make_sampler(events, inflight):
    sampler = LocalSGLangSampler()
    sampler._delta_config = SimpleNamespace(free_cache_engine=True, timeout_seconds=10)

    async def sleep(level=1):
        events.append(('sleep', level))
        # Mirror sglang's scheduler contract (weight_updater.release_memory_occupation).
        assert inflight['count'] == 0, \
            'release_memory_occupation should be called only when server is idle.'

    async def wake_up(tags=None):
        events.append(('wake_up', tuple(tags or ())))

    async def reset_prefix_cache():
        events.append(('reset_prefix_cache',))

    sampler.engine = SimpleNamespace(
        sleep=AsyncMock(side_effect=sleep),
        wake_up=AsyncMock(side_effect=wake_up),
        reset_prefix_cache=AsyncMock(side_effect=reset_prefix_cache))
    return sampler


def test_begin_pauses_before_sleep_and_publish_resumes_after_wake():
    events, inflight = [], {'count': 0}
    sampler = make_sampler(events, inflight)
    sampler.pause_generation = Mock(side_effect=lambda: events.append(('pause',)))
    sampler.resume_generation = Mock(side_effect=lambda: events.append(('resume',)))
    sampler.begin_delta_sync()
    sampler.publish_delta_sampler()
    assert events == [('pause',), ('sleep', 1), ('wake_up', ('kv_cache',)),
                      ('reset_prefix_cache',), ('resume',)]


def test_begin_drains_running_generation_before_kv_release():
    events, inflight = [], {'count': 1}
    sampler = make_sampler(events, inflight)
    reader_done = threading.Event()
    begin_done = threading.Event()
    errors = []

    def generation():
        # A blocking ``sample`` holds the gate's read side for its whole engine call.
        with sampler._gate().read():
            reader_done.wait(10)
            inflight['count'] = 0

    def begin():
        try:
            sampler.begin_delta_sync()
        except BaseException as error:  # noqa: BLE001 -- surfaced on the main thread
            errors.append(error)
        finally:
            begin_done.set()

    gen_thread = threading.Thread(target=generation)
    gen_thread.start()
    begin_thread = threading.Thread(target=begin)
    begin_thread.start()
    try:
        # The KV release must WAIT for the running request, not race it.
        assert not begin_done.wait(0.3), 'begin_delta_sync released KV while generation was running'
        assert events == []
        reader_done.set()
        assert begin_done.wait(10)
        assert errors == []
        assert events == [('sleep', 1)]
        sampler.publish_delta_sampler()
        assert events[-2:] == [('wake_up', ('kv_cache',)), ('reset_prefix_cache',)]
        # Gate reopened: a new generation passes without blocking.
        with sampler._gate().read():
            pass
    finally:
        reader_done.set()
        gen_thread.join(10)
        begin_thread.join(10)


def test_reverse_control_without_pause_kv_release_hits_idle_assert():
    """Single-point revert: drop ONLY the pause call (the pre-fix code) -> sglang's idle assert."""
    events, inflight = [], {'count': 1}
    sampler = make_sampler(events, inflight)
    sampler.pause_generation = Mock()  # no-op pause == the old gate-less begin_delta_sync
    with pytest.raises(AssertionError, match='release_memory_occupation should be called only when server is idle'):
        sampler.begin_delta_sync()
    assert events == [('sleep', 1)]


def test_fail_delta_sampler_reopens_the_gate():
    events, inflight = [], {'count': 0}
    sampler = make_sampler(events, inflight)
    sampler.begin_delta_sync()
    sampler.fail_delta_sampler()
    assert sampler._delta_job_failed
    # Gate must be open again so teardown / later RPCs are not stuck behind a paused sampler.
    with sampler._gate().read():
        pass


def test_without_kv_release_pause_and_resume_still_run():
    events, inflight = [], {'count': 0}
    sampler = make_sampler(events, inflight)
    sampler._delta_config = SimpleNamespace(free_cache_engine=False, timeout_seconds=10)
    sampler.pause_generation = Mock(side_effect=lambda: events.append(('pause',)))
    sampler.resume_generation = Mock(side_effect=lambda: events.append(('resume',)))
    sampler.begin_delta_sync()
    sampler.publish_delta_sampler()
    assert events == [('pause',), ('reset_prefix_cache',), ('resume',)]
    sampler.engine.sleep.assert_not_awaited()
    sampler.engine.wake_up.assert_not_awaited()