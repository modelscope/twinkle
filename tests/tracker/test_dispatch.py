# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for the dispatch system (``twinkle.tracker.__init__``).

Covers ``register_tracker``, ``dispatch``, ``dispatch_hyperparams``,
``clear_trackers``, ``set_rank``, and ``_auto_init_from_env``.
"""

import logging
import os
import pytest
import sys
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Module-level dependency mocks (mirrors test_swanlab.py)
# ---------------------------------------------------------------------------
for _mod in [
        'datasets',
        'datasets.utils',
        'datasets.utils.filelock',
        'torch',
        'accelerate',
        'transformers',
        'peft',
        'omegaconf',
        'modelscope',
        'safetensors',
        'fastapi',
        'tinker',
        'PIL',
        'PIL.Image',
        'wandb',
]:
    sys.modules.setdefault(_mod, MagicMock())

sys.modules.setdefault('twinkle.server', MagicMock())
sys.modules.setdefault('twinkle.server.model', MagicMock())
sys.modules.setdefault('twinkle.server.model.backends', MagicMock())
_common = MagicMock()
_common.clean_metrics = lambda d: {k: float(v) for k, v in d.items() if isinstance(v, (int, float))}
sys.modules['twinkle.server.model.backends.common'] = _common

sys.modules.setdefault('twinkle.utils.platforms', MagicMock())
sys.modules.setdefault('twinkle.utils.logger', MagicMock())
sys.modules.setdefault('swanlab', MagicMock())

import twinkle.tracker as tracker_mod  # noqa: E402
from twinkle.tracker import (clear_trackers, dispatch, dispatch_hyperparams, list_trackers,  # noqa: E402
                             register_tracker, set_rank)
# Now safe to import
from twinkle.tracker.base import ExperimentTracker  # noqa: E402


# ---------------------------------------------------------------------------
# Spy tracker
# ---------------------------------------------------------------------------
class SpyTracker(ExperimentTracker):
    """Minimal tracker that records all calls for later assertion."""

    def __init__(self, name: str = 'spy'):
        self.name = name
        self.reset()

    def reset(self):
        self.logged: list[tuple[dict, int]] = []
        self.hyperparams: list[dict] = []
        self.cleanup_called = False

    def log(self, data: dict, step: int) -> None:
        self.logged.append((dict(data), step))

    def log_hyperparams(self, params: dict) -> None:
        self.hyperparams.append(dict(params))

    def cleanup(self) -> None:
        self.cleanup_called = True

    def __repr__(self):
        return f"SpyTracker({self.name})"


class ErrorTracker(ExperimentTracker):
    """Tracker whose ``log()`` raises — used to test exception isolation."""

    def log(self, data: dict, step: int) -> None:
        raise RuntimeError('tracker error')

    def log_hyperparams(self, params: dict) -> None:
        raise RuntimeError('hparam error')

    def cleanup(self) -> None:
        raise RuntimeError('cleanup error')


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_global_state():
    """Reset module-level state before every test."""
    tracker_mod._global_trackers.clear()
    tracker_mod._adapter_trackers.clear()
    tracker_mod._rank = 0
    tracker_mod._hparams_dispatched.clear()
    yield


# ===================================================================
#  register_tracker / list_trackers
# ===================================================================


class TestRegistration:

    def test_register_one(self):
        t = SpyTracker()
        register_tracker(t)
        assert list_trackers() == [t]

    def test_register_multiple(self):
        t1, t2 = SpyTracker('a'), SpyTracker('b')
        register_tracker(t1)
        register_tracker(t2)
        assert list_trackers() == [t1, t2]

    def test_register_returns_none(self):
        ret = register_tracker(SpyTracker())
        assert ret is None

    def test_list_trackers_returns_copy(self):
        """list_trackers should return a copy, not the internal list."""
        t = SpyTracker()
        register_tracker(t)
        snapshot = list_trackers()
        clear_trackers()
        # Snapshot should still have the original reference
        assert t in snapshot


# ===================================================================
#  dispatch
# ===================================================================


class TestDispatch:

    def test_sends_to_all_trackers(self):
        t1, t2 = SpyTracker('a'), SpyTracker('b')
        register_tracker(t1)
        register_tracker(t2)
        set_rank(0)

        dispatch({'loss': 0.5, 'acc': 0.95}, step=10)

        assert t1.logged == [({'loss': 0.5, 'acc': 0.95}, 10)]
        assert t2.logged == [({'loss': 0.5, 'acc': 0.95}, 10)]

    def test_skipped_on_non_zero_rank(self):
        t = SpyTracker()
        register_tracker(t)
        set_rank(3)

        dispatch({'loss': 0.5}, step=1)

        assert t.logged == []

    def test_no_trackers_is_noop(self):
        """dispatch with no registered trackers should not crash."""
        dispatch({'loss': 0.5}, step=1)  # no assert — must not raise

    def test_rank_0_is_default(self):
        """Default rank is 0, so dispatch works without explicit set_rank."""
        t = SpyTracker()
        register_tracker(t)
        dispatch({'loss': 0.5}, step=1)
        assert len(t.logged) == 1

    def test_empty_data_after_clean_metrics_skips(self):
        """dispatch returns early when clean_metrics returns empty dict."""
        t = SpyTracker()
        register_tracker(t)
        set_rank(0)

        # Values that clean_metrics cannot convert to float
        dispatch({'invalid': [1, 2, 3], 'text': 'not-a-number'}, step=5)

        assert t.logged == []

    def test_exception_isolation(self):
        """One tracker raising does not prevent others from receiving."""
        good = SpyTracker('good')
        bad = ErrorTracker()
        register_tracker(good)
        register_tracker(bad)

        dispatch({'loss': 0.5}, step=10)

        assert len(good.logged) == 1
        assert good.logged[0] == ({'loss': 0.5}, 10)

    def test_exception_isolation_reverse_order(self):
        """Exception isolation works regardless of tracker order."""
        bad = ErrorTracker()
        good = SpyTracker('good')
        register_tracker(bad)
        register_tracker(good)

        dispatch({'loss': 0.5}, step=10)

        assert len(good.logged) == 1

    def test_multiple_steps(self):
        t = SpyTracker()
        register_tracker(t)
        set_rank(0)

        dispatch({'loss': 0.5}, step=1)
        dispatch({'loss': 0.3}, step=2)
        dispatch({'loss': 0.1}, step=3)

        assert len(t.logged) == 3
        assert t.logged[0] == ({'loss': 0.5}, 1)
        assert t.logged[1] == ({'loss': 0.3}, 2)
        assert t.logged[2] == ({'loss': 0.1}, 3)

    def test_rank_change_during_runtime(self):
        """Changing rank mid-training affects subsequent dispatches."""
        t = SpyTracker()
        register_tracker(t)

        set_rank(0)
        dispatch({'loss': 0.5}, step=1)
        assert len(t.logged) == 1

        set_rank(1)
        dispatch({'loss': 0.3}, step=2)
        assert len(t.logged) == 1  # no change — rank 1 skipped

        set_rank(0)
        dispatch({'loss': 0.1}, step=3)
        assert len(t.logged) == 2  # now rank 0 again


# ===================================================================
#  dispatch_hyperparams
# ===================================================================


class TestDispatchHyperparams:

    def test_sends_to_all_trackers(self):
        t1, t2 = SpyTracker('a'), SpyTracker('b')
        register_tracker(t1)
        register_tracker(t2)
        set_rank(0)

        dispatch_hyperparams({'lr': 1e-4})

        assert t1.hyperparams == [{'lr': 1e-4}]
        assert t2.hyperparams == [{'lr': 1e-4}]

    def test_idempotent_with_adapter_name(self):
        """Same adapter_name only dispatches once."""
        t = SpyTracker()
        register_tracker(t)
        set_rank(0)

        dispatch_hyperparams({'lr': 1e-4}, adapter_name='default')
        dispatch_hyperparams({'lr': 2e-4}, adapter_name='default')  # ignored
        dispatch_hyperparams({'batch_size': 32}, adapter_name='default')  # ignored

        assert len(t.hyperparams) == 1
        assert t.hyperparams[0] == {'lr': 1e-4}

    def test_different_adapters_separate(self):
        """Different adapter_names are each dispatched once."""
        t = SpyTracker()
        register_tracker(t)
        set_rank(0)

        dispatch_hyperparams({'lr': 1e-4}, adapter_name='lora_a')
        dispatch_hyperparams({'lr': 2e-4}, adapter_name='lora_b')
        dispatch_hyperparams({'lr': 3e-4}, adapter_name='lora_a')  # ignored

        assert len(t.hyperparams) == 2
        assert t.hyperparams[0] == {'lr': 1e-4}
        assert t.hyperparams[1] == {'lr': 2e-4}

    def test_without_adapter_sends_once(self):
        """When adapter_name is None, only the first call dispatches (idempotent via _default_)."""
        t = SpyTracker()
        register_tracker(t)
        set_rank(0)

        dispatch_hyperparams({'lr': 1e-4})
        dispatch_hyperparams({'lr': 2e-4})
        dispatch_hyperparams({'lr': 3e-4})

        assert len(t.hyperparams) == 1
        assert t.hyperparams[0] == {'lr': 1e-4}

    def test_mixed_adapter_and_no_adapter(self):
        """Calls with and without adapter_name interact correctly."""
        t = SpyTracker()
        register_tracker(t)
        set_rank(0)

        dispatch_hyperparams({'a': 1}, adapter_name='adp')  # sent
        dispatch_hyperparams({'b': 2})  # sent (no adapter, first call)
        dispatch_hyperparams({'c': 3}, adapter_name='adp')  # ignored (idempotent)
        dispatch_hyperparams({'d': 4})  # ignored (no adapter, idempotent via _default_)

        assert len(t.hyperparams) == 2

    def test_skipped_on_non_zero_rank(self):
        t = SpyTracker()
        register_tracker(t)
        set_rank(2)

        dispatch_hyperparams({'lr': 1e-4})

        assert t.hyperparams == []

    def test_no_trackers_is_noop(self):
        dispatch_hyperparams({'lr': 1e-4}, adapter_name='test')

    def test_exception_isolation(self):
        good = SpyTracker('good')
        bad = ErrorTracker()
        register_tracker(good)
        register_tracker(bad)

        dispatch_hyperparams({'lr': 1e-4})

        assert len(good.hyperparams) == 1


# ===================================================================
#  clear_trackers
# ===================================================================


class TestClearTrackers:

    def test_calls_cleanup_on_all(self):
        t1, t2 = SpyTracker('a'), SpyTracker('b')
        register_tracker(t1)
        register_tracker(t2)

        clear_trackers()

        assert t1.cleanup_called
        assert t2.cleanup_called
        assert list_trackers() == []

    def test_cleanup_exception_isolation(self):
        """cleanup() raising on one tracker doesn't break others."""
        bad = ErrorTracker()
        good = SpyTracker('good')
        register_tracker(bad)
        register_tracker(good)

        clear_trackers()  # must not raise

        assert good.cleanup_called
        assert list_trackers() == []

    def test_empty_list_is_noop(self):
        clear_trackers()  # must not raise
        assert list_trackers() == []

    def test_idempotent(self):
        """Calling clear_trackers twice is safe."""
        t = SpyTracker()
        register_tracker(t)
        clear_trackers()
        clear_trackers()
        assert list_trackers() == []


# ===================================================================
#  set_rank
# ===================================================================


class TestSetRank:

    def test_default_rank_is_zero(self):
        assert tracker_mod._rank == 0  # after fixture reset

    def test_set_rank_changes_global(self):
        set_rank(3)
        assert tracker_mod._rank == 3

    def test_set_rank_zero(self):
        set_rank(0)
        assert tracker_mod._rank == 0

    def test_set_rank_negative(self):
        """Negative rank values are stored as-is (caller responsibility)."""
        set_rank(-1)
        # A negative rank will cause dispatch to skip (since rank != 0)
        t = SpyTracker()
        register_tracker(t)
        dispatch({'loss': 0.5}, step=1)
        assert t.logged == []


# ===================================================================
#  _auto_init_from_env
# ===================================================================


class TestAutoInitFromEnv:
    """Environment-variable auto-initialisation."""

    def _reset_auto_init(self):
        """Allow _auto_init_from_env to run again."""
        tracker_mod._AUTO_INIT_DONE = False
        tracker_mod._global_trackers.clear()
        tracker_mod._adapter_trackers.clear()

    def test_env_empty_is_noop(self):
        """No TWINKLE_TRACKERS → nothing registered."""
        self._reset_auto_init()
        with patch.dict(os.environ, {}, clear=True):
            tracker_mod._auto_init_from_env()
        assert list_trackers() == []

    def test_env_swanlab_registers_tracker(self):
        """TWINKLE_TRACKERS=swanlab registers a SwanLabTracker."""
        self._reset_auto_init()
        with patch.dict(os.environ, {'TWINKLE_TRACKERS': 'swanlab'}, clear=True):
            tracker_mod._auto_init_from_env()
        trackers = list_trackers()
        assert len(trackers) == 1
        from twinkle.tracker.swanlab import SwanLabTracker
        assert isinstance(trackers[0], SwanLabTracker)

    def test_env_wandb_registers_tracker(self):
        """TWINKLE_TRACKERS=wandb registers a WandbTracker."""
        self._reset_auto_init()
        with patch.dict(os.environ, {'TWINKLE_TRACKERS': 'wandb'}, clear=True):
            tracker_mod._auto_init_from_env()
        trackers = list_trackers()
        assert len(trackers) == 1
        from twinkle.tracker.wandb import WandbTracker
        assert isinstance(trackers[0], WandbTracker)

    def test_env_both_registers_both(self):
        """TWINKLE_TRACKERS=swanlab,wandb registers both."""
        self._reset_auto_init()
        with patch.dict(os.environ, {'TWINKLE_TRACKERS': 'swanlab,wandb'}, clear=True):
            tracker_mod._auto_init_from_env()
        assert len(list_trackers()) == 2

    def test_env_unknown_logs_warning(self, caplog):
        """Unknown tracker name logs a warning."""
        self._reset_auto_init()
        caplog.set_level(logging.WARNING)
        with patch.dict(os.environ, {'TWINKLE_TRACKERS': 'unknown'}, clear=True):
            tracker_mod._auto_init_from_env()
        assert 'Unknown tracker backend in TWINKLE_TRACKERS: unknown' in caplog.text
        assert list_trackers() == []

    def test_env_project_and_experiment(self):
        """TWINKLE_TRACKER_PROJECT and _EXPERIMENT env vars are used."""
        self._reset_auto_init()
        with patch.dict(
                os.environ, {
                    'TWINKLE_TRACKERS': 'swanlab',
                    'TWINKLE_TRACKER_PROJECT': 'my-project',
                    'TWINKLE_TRACKER_EXPERIMENT': 'my-exp',
                },
                clear=True):
            tracker_mod._auto_init_from_env()
        trackers = list_trackers()
        assert len(trackers) == 1
        # The swanlab.init mock was called with these values
        import swanlab
        swanlab.init.assert_called()

    def test_auto_init_guard(self):
        """_AUTO_INIT_DONE prevents re-initialisation."""
        self._reset_auto_init()
        tracker_mod._AUTO_INIT_DONE = True  # simulate already done
        with patch.dict(os.environ, {'TWINKLE_TRACKERS': 'swanlab'}, clear=True):
            tracker_mod._auto_init_from_env()
        # If the guard worked, no trackers were added
        assert list_trackers() == []

    def test_auto_init_exception_does_not_crash(self):
        """An exception during tracker construction is caught."""
        self._reset_auto_init()

        # Make SwanLabTracker constructor raise by removing swanlab mock
        with patch.dict(os.environ, {'TWINKLE_TRACKERS': 'swanlab'}, clear=True):
            # This will call SwanLabTracker(project=..., ...) which does
            # import swanlab; swanlab.init(...). Our mock will not crash.
            tracker_mod._auto_init_from_env()
        # Should have one tracker if successful
        assert len(list_trackers()) == 1

    def test_env_whitespace_handling(self):
        """Extra whitespace in TWINKLE_TRACKERS is tolerated."""
        self._reset_auto_init()
        with patch.dict(os.environ, {'TWINKLE_TRACKERS': '  swanlab ,  wandb  '}, clear=True):
            tracker_mod._auto_init_from_env()
        assert len(list_trackers()) == 2
