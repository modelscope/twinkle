# Copyright (c) ModelScope Contributors. All rights reserved.
import time
from typing import List, Union

from twinkle.data_format import InputFeature, ModelOutput
from .base import Metric


class TrainMetric(Metric):
    """The training metric.

    Args:
        device_mesh: The device mesh
        process_group: The process group to collect data from
    """

    def __init__(self, device_mesh=None, process_group=None, **kwargs):
        super().__init__(device_mesh, process_group, **kwargs)
        self.lr = None
        self.step = 0
        self.last_step = 0
        self.gradient_accumulation_steps = 1
        self.start_time = time.time()
        self.time = time.time()
        self.lrs = []

    def accumulate(self, inputs: Union[InputFeature, List[InputFeature]], outputs: ModelOutput, **kwargs):
        lr = kwargs.get('lr')
        if isinstance(lr, list):
            lr = [f'{x:.6e}' for x in lr]
        else:
            lr = f'{lr:.6e}'
        self.lr = lr
        self.lrs.append(lr)
        self.step = kwargs.get('step')
        self.gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', self.gradient_accumulation_steps)

    def reset(self):
        self.time = time.time()
        self.last_step = self.step

    def calculate(self):
        results = {}
        if self.lr is not None:
            if isinstance(self.lr, list) and len(self.lr) == 1:
                self.lr = self.lr[0]
            if isinstance(self.lr, list):
                for idx, lr in enumerate(self.lr):
                    results[f'learning rate(param group {idx + 1})'] = lr
            else:
                results['learning rate'] = self.lr
        if self.step is not None:
            results['iters'] = self.step // self.gradient_accumulation_steps
            interval = time.time() - self.time
            speed = (self.step - self.last_step) / interval / self.gradient_accumulation_steps
            if interval < 60:
                results['total time elapse'] = f'{(time.time() - self.start_time):.0f} seconds'
            else:
                results['total time elapse'] = f'{(time.time() - self.start_time) / 60:.1f} minutes'
            results['speed'] = f'{speed:.2f} iters/s'
        self.reset()
        return results

def memory_metrics(rows):
    """Convert per-worker byte counters and reduce peaks by max, host usage by mean."""
    rows = [rows] if isinstance(rows, dict) else rows
    fields = {
        "max_memory_allocated_bytes": "max_memory_allocated_gb",
        "max_memory_reserved_bytes": "max_memory_reserved_gb",
        "cpu_memory_used_bytes": "cpu_memory_used_gb",
    }
    result = {}
    for raw, name in fields.items():
        values = [row[raw] / 1024**3 for row in rows if raw in row]
        if values:
            result[f"perf/{name}"] = max(values) if raw.startswith("max_") else sum(values) / len(values)
    return result


class ThroughputMetric(Metric):
    """Accumulate explicit token counts and phase seconds; calculate ratios of sums.

    The producer defines window boundaries and supplies the physical device budget.
    No timing, loop state, task waiting or file output occurs inside this metric.
    """

    def __init__(self, device_mesh=None, process_group=None, **kwargs):
        super().__init__(device_mesh, process_group, **kwargs)
        self.reset()

    def reset(self):
        from collections import Counter

        self.prompts = Counter()
        self.responses = Counter()
        self.timings = {}
        self.num_devices = None

    def accumulate(
        self, inputs=None, outputs=None, *, prompt_lengths, response_lengths, timings, num_devices, **kwargs
    ):
        import math

        prompts, responses = list(prompt_lengths), list(response_lengths)
        if not prompts or len(prompts) != len(responses):
            raise ValueError("expected a nonempty batch of paired prompt/response lengths")
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in prompts + responses):
            raise ValueError("token lengths must be nonnegative integers")
        if not isinstance(num_devices, int) or isinstance(num_devices, bool) or num_devices < 1:
            raise ValueError("num_devices must count allocated physical devices")
        if self.num_devices is not None and self.num_devices != num_devices:
            raise ValueError("reset the metric when the physical allocation changes")
        if timings.get("step", 0) <= 0 or any(not math.isfinite(t) or t < 0 for t in timings.values()):
            raise ValueError("timings must be finite and nonnegative, with positive step time")
        self.num_devices = num_devices
        self.prompts.update(prompts)
        self.responses.update(responses)
        for name, seconds in timings.items():
            self.timings[name] = self.timings.get(name, 0.0) + seconds

    def calculate(self, *, prompt_width=None, response_width=None):
        if not self.prompts:
            return {}
        prompts, responses = self.prompts, self.responses
        count = sum(prompts.values())
        p = sum(length * frequency for length, frequency in prompts.items())
        r = sum(length * frequency for length, frequency in responses.items())
        total, elapsed = p + r, self.timings["step"]
        result = {
            "perf/total_num_tokens": total,
            "perf/num_prompt_tokens": p,
            "perf/num_response_tokens": r,
            "perf/num_samples": count,
            "perf/num_gpus": self.num_devices,
            "perf/time_per_step": elapsed,
            "perf/throughput": total / elapsed / self.num_devices,
            "perf/throughput_total": total / elapsed,
            "perf/response_tokens_per_second": r / elapsed,
            "perf/samples_per_second": count / elapsed,
            **{f"timing_s/{name}": seconds for name, seconds in self.timings.items()},
        }
        denominators = {
            "gen": r,
            **dict.fromkeys(("ref", "values", "adv", "adv_pipeline", "update_critic", "update_actor"), total),
        }
        for name, tokens in denominators.items():
            if name in self.timings:
                result[f"timing_per_token_ms/{name}"] = self.timings[name] * 1000 / tokens if tokens else 0.0
        for name, lengths, width in (
            ("prompt_length", prompts, prompt_width),
            ("response_length", responses, response_width),
        ):
            width = max(lengths) if width is None else width
            if width < max(lengths):
                raise ValueError("padded width cannot be smaller than an actual sequence")
            result.update(
                {
                    f"{name}/mean": sum(n * frequency for n, frequency in lengths.items()) / count,
                    f"{name}/min": min(lengths),
                    f"{name}/max": max(lengths),
                    f"{name}/clip_ratio": lengths.get(width, 0) / count,
                }
            )
        active = {n: frequency for n, frequency in responses.items() if n > 0}
        active_count = sum(active.values())
        width = max(responses) if response_width is None else response_width
        result.update(
            {
                "response/aborted_ratio": 1 - active_count / count,
                "response_length_non_aborted/mean": r / active_count if active_count else float("nan"),
                "response_length_non_aborted/min": min(active) if active else float("nan"),
                "response_length_non_aborted/max": max(active) if active else float("nan"),
                "response_length_non_aborted/clip_ratio": active.get(width, 0) / active_count
                if active_count
                else float("nan"),
            }
        )
        return result
