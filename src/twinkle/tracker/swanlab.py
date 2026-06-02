# Copyright (c) ModelScope Contributors. All rights reserved.
"""SwanLab experiment tracker."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .base import ExperimentTracker

logger = logging.getLogger(__name__)


class SwanLabTracker(ExperimentTracker):
    """Experiment tracker backed by `SwanLab <https://swanlab.cn>`_.

    Args:
        project: SwanLab project name.
        experiment_name: Optional run / experiment name.
        config: Optional dict of hyperparameters to record.
        output_dir: If set, the SwanLab experiment URL is written to
            ``{output_dir}/swanlab_config.json`` so users can easily
            find the online dashboard.
        **kwargs: Passed through to ``swanlab.init()``.
    """

    def __init__(
        self,
        project: str,
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        **kwargs,
    ):
        import swanlab

        api_key = kwargs.pop('api_key', None) or os.environ.get('SWANLAB_API_KEY')
        logdir = kwargs.pop('logdir', None) or os.environ.get('SWANLAB_LOG_DIR', 'swanlog')
        mode = kwargs.pop('mode', None) or os.environ.get('SWANLAB_MODE', 'cloud')

        if api_key:
            swanlab.login(api_key)

        self._run = swanlab.init(
            project=project,
            experiment_name=experiment_name,
            config={
                'framework': '\u2728Twinkle',
                **(config or {})
            },
            logdir=logdir,
            mode=mode,
            **kwargs,
        )

        if output_dir:
            self._save_experiment_info(output_dir)

    def log(self, data: Dict[str, float], step: int) -> None:
        self._run.log(data, step=step)

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        self._run.config.update(params)

    def cleanup(self) -> None:
        try:
            self._run.finish()
        except Exception:
            logger.warning('SwanLab finish() failed', exc_info=True)

    def _save_experiment_info(self, output_dir: str) -> None:
        try:
            info = {'swanlab_experiment_url': self._run.get_run().url}
            out = Path(output_dir) / 'swanlab_config.json'
            out.write_text(json.dumps(info, indent=2))
        except Exception:
            pass
