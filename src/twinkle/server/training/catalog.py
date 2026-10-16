# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pure lookup and request validation, separate from runtime allocation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from twinkle.server.config.training import LoraTrainingConfig, TrainingConfig, TrainingModeConfig, TrainingModelConfig
from twinkle.server.exceptions import RequestRejectedError


@dataclass(frozen=True)
class TrainingSpec:
    model: TrainingModelConfig
    mode: Literal['lora', 'full']
    config: TrainingModeConfig

    @property
    def key(self) -> tuple[str, str]:
        return self.model.name, self.mode

    @property
    def capacity(self) -> int:
        return self.config.max_loras if isinstance(self.config, LoraTrainingConfig) else 1


class TrainingCatalog:

    def __init__(self, config: TrainingConfig):
        self._models = {model.name: model for model in config.models}

    def get(self, name: str, mode: Literal['lora', 'full']) -> TrainingSpec:
        model = self._models.get(name)
        if model is None:
            raise RequestRejectedError(f'Unknown training model: {name}')
        config = getattr(model.modes, mode)
        if config is None:
            raise RequestRejectedError(f'Training model {name} does not provide {mode} mode')
        return TrainingSpec(model, mode, config)

    def specs(self) -> tuple[TrainingSpec, ...]:
        return tuple(
            TrainingSpec(model, mode, config) for model in self._models.values() for mode in ('lora', 'full')
            if (config := getattr(model.modes, mode)) is not None)
