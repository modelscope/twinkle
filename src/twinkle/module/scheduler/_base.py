# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations


class ScheduleShapeGuardMixin:
    # Keep the schedule horizon out of state_dict so a resume rebuilt with a new max_steps keeps its
    # own construction spec instead of being clobbered by the checkpoint's old horizon (defect69);
    # HF's closure-bound lambdas already behave this way. last_epoch/base_lrs still round-trip.
    _SHAPE_KEYS = ('num_training_steps', 'num_warmup_steps', 'num_cycles')

    def state_dict(self):
        sd = super().state_dict()
        for k in self._SHAPE_KEYS:
            sd.pop(k, None)
        return sd

    def load_state_dict(self, state_dict):
        super().load_state_dict({k: v for k, v in state_dict.items() if k not in self._SHAPE_KEYS})