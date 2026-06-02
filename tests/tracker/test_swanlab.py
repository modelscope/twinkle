# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for SwanLabTracker.

These tests mock the ``swanlab`` package so they can run without a real
SwanLab installation or API key.  Each test verifies that the tracker
delegates correctly to the underlying ``swanlab`` SDK.
"""

import json
import logging
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Module-level dependency mocks.
#
# Importing ``twinkle.tracker.swanlab`` triggers a package-init chain that
# pulls in heavyweight third-party libraries (datasets, torch, …).  We mock
# them here so the tests can run without the full dependency tree installed.
# ---------------------------------------------------------------------------
for _mod in [
    "datasets",
    "datasets.utils",
    "datasets.utils.filelock",
    "torch",
    "accelerate",
    "transformers",
    "peft",
    "omegaconf",
    "modelscope",
    "safetensors",
    "fastapi",
    "tinker",
    "PIL",
    "PIL.Image",
]:
    sys.modules.setdefault(_mod, MagicMock())

# twinkle.server.model.backends.common is imported by twinkle.tracker itself
sys.modules.setdefault("twinkle.server", MagicMock())
sys.modules.setdefault("twinkle.server.model", MagicMock())
sys.modules.setdefault("twinkle.server.model.backends", MagicMock())
_common = MagicMock()
_common.clean_metrics = lambda d, **kw: {k: float(v) for k, v in d.items() if isinstance(v, (int, float))}
sys.modules["twinkle.server.model.backends.common"] = _common

# Intermediate twinkle sub-packages that the init chain touches
sys.modules.setdefault("twinkle.utils.platforms", MagicMock())
sys.modules.setdefault("twinkle.utils.logger", MagicMock())

# Mock swanlab itself so that ``import swanlab`` inside SwanLabTracker
# resolves to a mock rather than trying to import the real package.
sys.modules.setdefault("swanlab", MagicMock())

# Now that all heavy deps are mocked, the import should succeed.
from twinkle.tracker.swanlab import SwanLabTracker


# ===================================================================
#  Helpers
# ===================================================================

@pytest.fixture(autouse=True)
def _reset_swanlab_mock():
    """Reset the swanlab mock before each test so call counts are clean."""
    swanlab_mock = sys.modules["swanlab"]
    swanlab_mock.reset_mock()
    swanlab_mock.init.return_value = MagicMock()
    yield


def _mock_swanlab():
    """Shortcut to access the module-level swanlab mock."""
    return sys.modules["swanlab"]


def _mock_run():
    """Shortcut to access the run mock returned by swanlab.init()."""
    return _mock_swanlab().init.return_value


# ===================================================================
#  __init__ — construction & parameter routing
# ===================================================================

class TestInit:
    """SwanLabTracker.__init__ parameter handling."""

    def test_defaults(self):
        """Default logdir and mode when neither kwarg nor env var is set."""
        SwanLabTracker(project="test-project")
        _mock_swanlab().init.assert_called_once_with(
            project="test-project",
            experiment_name=None,
            config={"framework": "\u2728Twinkle"},
            logdir="swanlog",
            mode="cloud",
        )
        _mock_swanlab().login.assert_not_called()

    def test_with_api_key_kwarg(self):
        """api_key kwarg triggers swanlab.login() before init."""
        SwanLabTracker(project="test-project", api_key="key-123")
        _mock_swanlab().login.assert_called_once_with("key-123")

    def test_with_api_key_from_env(self):
        """SWANLAB_API_KEY env var triggers login when api_key kwarg absent."""
        with patch.dict(os.environ, {"SWANLAB_API_KEY": "env-key"}):
            SwanLabTracker(project="test-project")
        _mock_swanlab().login.assert_called_once_with("env-key")

    def test_api_key_kwarg_precedence(self):
        """api_key kwarg takes precedence over SWANLAB_API_KEY env var."""
        with patch.dict(os.environ, {"SWANLAB_API_KEY": "env-key"}):
            SwanLabTracker(project="test-project", api_key="kwarg-key")
        _mock_swanlab().login.assert_called_once_with("kwarg-key")

    def test_experiment_name_and_config(self):
        """experiment_name and config are forwarded to swanlab.init."""
        SwanLabTracker(
            project="test-project",
            experiment_name="my-exp",
            config={"lr": 1e-4, "batch_size": 32},
        )
        _mock_swanlab().init.assert_called_once_with(
            project="test-project",
            experiment_name="my-exp",
            config={"framework": "\u2728Twinkle", "lr": 1e-4, "batch_size": 32},
            logdir="swanlog",
            mode="cloud",
        )

    def test_logdir_and_mode_kwargs(self):
        """Explicit logdir/mode override both defaults and env vars."""
        with patch.dict(os.environ, {"SWANLAB_LOG_DIR": "env_logs", "SWANLAB_MODE": "cloud"}):
            SwanLabTracker(project="test-project", logdir="my_logs", mode="local")
        _mock_swanlab().init.assert_called_once_with(
            project="test-project",
            experiment_name=None,
            config={"framework": "\u2728Twinkle"},
            logdir="my_logs",
            mode="local",
        )

    def test_logdir_from_env(self):
        """SWANLAB_LOG_DIR env var is used when no logdir kwarg."""
        with patch.dict(os.environ, {"SWANLAB_LOG_DIR": "env_logs"}):
            SwanLabTracker(project="test-project")
        _mock_swanlab().init.assert_called_once_with(
            project="test-project",
            experiment_name=None,
            config={"framework": "\u2728Twinkle"},
            logdir="env_logs",
            mode="cloud",
        )

    def test_mode_from_env(self):
        """SWANLAB_MODE env var is used when no mode kwarg."""
        with patch.dict(os.environ, {"SWANLAB_MODE": "local"}):
            SwanLabTracker(project="test-project")
        _mock_swanlab().init.assert_called_once_with(
            project="test-project",
            experiment_name=None,
            config={"framework": "\u2728Twinkle"},
            logdir="swanlog",
            mode="local",
        )

    def test_output_dir_writes_info_file(self, tmp_path):
        """output_dir causes experiment URL to be saved as JSON."""
        _mock_run().get_run.return_value.url = "https://swanlab.cn/foo/bar"
        SwanLabTracker(project="test", output_dir=str(tmp_path))

        info_file = tmp_path / "swanlab_config.json"
        assert info_file.exists()
        data = json.loads(info_file.read_text())
        assert data == {"swanlab_experiment_url": "https://swanlab.cn/foo/bar"}

    def test_additional_kwargs_passthrough(self):
        """Arbitrary kwargs reach swanlab.init after api_key/api_key is consumed."""
        SwanLabTracker(project="test-project", workspace="my-ws", tags=["t1"])
        kwargs = _mock_swanlab().init.call_args[1]
        # workspace and tags are forwarded via **kwargs passthrough
        assert kwargs["workspace"] == "my-ws"
        assert kwargs["tags"] == ["t1"]
        # api_key is consumed by swanlab.login() and must NOT leak into init
        assert "api_key" not in kwargs
        # logdir and mode are explicit named args (not passthrough), always present


# ===================================================================
#  log
# ===================================================================

class TestLog:
    """SwanLabTracker.log() delegates to swanlab.Run.log()."""

    def test_log_basic(self):
        tracker = SwanLabTracker(project="test")
        tracker.log({"loss": 0.5}, step=10)
        _mock_run().log.assert_called_once_with({"loss": 0.5}, step=10)

    def test_log_multiple_steps(self):
        tracker = SwanLabTracker(project="test")
        tracker.log({"loss": 0.5}, step=1)
        tracker.log({"loss": 0.3}, step=2)
        tracker.log({"loss": 0.1}, step=3)

        assert _mock_run().log.call_count == 3
        _mock_run().log.assert_any_call({"loss": 0.5}, step=1)
        _mock_run().log.assert_any_call({"loss": 0.3}, step=2)
        _mock_run().log.assert_any_call({"loss": 0.1}, step=3)

    def test_log_empty_dict(self):
        """Empty dict is forwarded (dispatch layer normally filters it earlier)."""
        tracker = SwanLabTracker(project="test")
        tracker.log({}, step=5)
        _mock_run().log.assert_called_once_with({}, step=5)


# ===================================================================
#  log_hyperparams
# ===================================================================

class TestLogHyperparams:
    """SwanLabTracker.log_hyperparams() updates run config."""

    def test_log_hyperparams_updates_config(self):
        tracker = SwanLabTracker(project="test")
        tracker.log_hyperparams({"lr": 1e-4, "batch_size": 32})
        _mock_run().config.update.assert_called_once_with({"lr": 1e-4, "batch_size": 32})

    def test_log_hyperparams_multiple_calls(self):
        tracker = SwanLabTracker(project="test")
        tracker.log_hyperparams({"lr": 1e-4})
        tracker.log_hyperparams({"batch_size": 32})
        assert _mock_run().config.update.call_count == 2

    def test_log_hyperparams_empty(self):
        tracker = SwanLabTracker(project="test")
        tracker.log_hyperparams({})
        _mock_run().config.update.assert_called_once_with({})


# ===================================================================
#  cleanup
# ===================================================================

class TestCleanup:
    """SwanLabTracker.cleanup() finalises the run."""

    def test_cleanup_calls_finish(self):
        tracker = SwanLabTracker(project="test")
        tracker.cleanup()
        _mock_run().finish.assert_called_once()

    def test_cleanup_exception_logged(self, caplog):
        """Exception in finish() is logged as warning, not propagated."""
        _mock_run().finish.side_effect = RuntimeError("connection lost")
        tracker = SwanLabTracker(project="test")

        caplog.set_level(logging.WARNING)
        tracker.cleanup()

        assert "SwanLab finish() failed" in caplog.text
        assert "connection lost" in caplog.text
        _mock_run().finish.assert_called_once()


# ===================================================================
#  _save_experiment_info
# ===================================================================

class TestSaveExperimentInfo:
    """_save_experiment_info writes the experiment URL to disk."""

    def test_saves_url(self, tmp_path):
        _mock_run().get_run.return_value.url = "https://swanlab.cn/exp/abc"
        SwanLabTracker(project="test", output_dir=str(tmp_path))

        info = json.loads((tmp_path / "swanlab_config.json").read_text())
        assert info == {"swanlab_experiment_url": "https://swanlab.cn/exp/abc"}

    def test_idempotent_overwrite(self, tmp_path):
        """Multiple trackers with the same output_dir overwrite the file."""
        run_a = _mock_run()
        run_a.get_run.return_value.url = "https://swanlab.cn/exp/a"
        SwanLabTracker(project="test", output_dir=str(tmp_path))

        # Reset mock so the second tracker creates a new run mock
        _mock_swanlab().reset_mock()
        run_b = MagicMock()
        run_b.get_run.return_value.url = "https://swanlab.cn/exp/b"
        _mock_swanlab().init.return_value = run_b
        SwanLabTracker(project="test", output_dir=str(tmp_path))

        info = json.loads((tmp_path / "swanlab_config.json").read_text())
        assert info["swanlab_experiment_url"] == "https://swanlab.cn/exp/b"


# ===================================================================
#  Edge cases
# ===================================================================

class TestEdgeCases:
    """Unusual / error scenarios."""

    def test_empty_project_name(self):
        """An empty project string is forwarded (swanlab may reject it)."""
        SwanLabTracker(project="")
        assert _mock_swanlab().init.call_args[1]["project"] == ""

    def test_none_experiment_name(self):
        """None experiment_name is passed as None (swanlab uses default)."""
        SwanLabTracker(project="test", experiment_name=None)
        assert _mock_swanlab().init.call_args[1]["experiment_name"] is None

    def test_config_overrides_framework_key(self):
        """User-provided 'framework' in config overrides the default."""
        SwanLabTracker(project="test", config={"framework": "MyFramework"})
        cfg = _mock_swanlab().init.call_args[1]["config"]
        # The tracker does: {"framework": "✨Twinkle", **(config or {})},
        # so user's framework wins via dict unpacking.
        assert cfg["framework"] == "MyFramework"
