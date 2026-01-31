# Copyright (c) ModelScope Contributors. All rights reserved.
import time

from .base import Metric


class TrainMetric(Metric):
    """The training metric.

    Args:
        device_mesh: The device mesh
        process_group: The process group to collect data from
    """

    def __init__(self, device_mesh, process_group, **kwargs):
        super().__init__(device_mesh, process_group, **kwargs)
        self.lr = ''
        self.step = 0
        self.last_step = 0
        self.gradient_accumulation_steps = 1
        self.start_time = time.time()
        self.time = time.time()

    def accumulate(self, inputs, outputs):
        lr = outputs.get('lr')
        if isinstance(lr, list):
            lr = [f'{x:.2e}' for x in lr]
            lr = ','.join(lr)
        else:
            lr = f'{lr:.2e}'
        self.lr = lr
        self.step = outputs.get('step')
        self.gradient_accumulation_steps = outputs.get('gradient_accumulation_steps', 1)

    def reset(self):
        self.time = time.time()
        self.last_step = self.step

    def calculate(self):
        results = {}
        if self.lr is not None:
            results['last lr(by param_groups)'] = self.lr
        if self.step is not None:
            results['iters'] = (self.step - 1) // self.gradient_accumulation_steps
            interval = time.time() - self.time
            speed = (self.step - self.last_step) / interval / self.gradient_accumulation_steps
            if interval < 60:
                results['total time elapse'] = f'{(time.time() - self.start_time):.0f} seconds'
            else:
                results['total time elapse'] = f'{(time.time() - self.start_time)/60:.1f} minutes'
            results['speed'] = f'{speed:.2f} iters/s'
        self.reset()
        return results