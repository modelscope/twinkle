# Copyright (c) ModelScope Contributors. All rights reserved.
"""Weights & Biases experiment tracker."""

import logging
import os
from typing import Any, Dict, Optional

from .base import ExperimentTracker

logger = logging.getLogger(__name__)


class WandbTracker(ExperimentTracker):
    """Experiment tracker backed by `Weights & Biases <https://wandb.ai>`_.

    Args:
        project: W&B project name.
        experiment_name: Optional run name.
        config: Optional dict of hyperparameters.
        **kwargs: Passed through to ``wandb.init()``.
    """

    def __init__(
        self,
        project: str,
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        import wandb

        entity = kwargs.pop('entity', None) or os.environ.get('WANDB_ENTITY')
        settings = kwargs.pop('settings', None)
        proxy = kwargs.pop('wandb_proxy', None) or os.environ.get('WANDB_PROXY')
        if proxy:
            settings = wandb.Settings(https_proxy=proxy)

        self._run = wandb.init(
            project=project,
            name=experiment_name,
            entity=entity,
            config={
                'framework': '\u2728Twinkle',
                **(config or {})
            },
            settings=settings,
            **kwargs,
        )

    def log(self, data: Dict[str, float], step: int) -> None:
        self._run.log(data, step=step)

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        self._run.config.update(params)

    def cleanup(self) -> None:
        try:
            self._run.finish(exit_code=0)
        except Exception:
            logger.warning('WandB finish() failed', exc_info=True)
