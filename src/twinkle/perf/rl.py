# Copyright (c) ModelScope Contributors. All rights reserved.
"""Reusable RL iteration accounting, phase timing and persistent metrics.

The iteration axis is independent of optimizer updates. Never average per-step
throughputs to compare a run: sum tokens / sum seconds / physical allocation.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

from .rl_metrics import JsonlMetricWriter, compute_rl_performance_metrics


def sample_lengths(sample):
    """Unpadded initial prompt and full response, including tool observations.

    Twinkle stores shifted labels/masks. Policy tokens identify the first response
    position; subsequent masked tool tokens belong to the full response span too.
    """
    encoded = sample.encoded
    ids = list(encoded["input_ids"])
    labels = list(encoded["labels"])
    if len(labels) != len(ids):
        raise ValueError("performance metrics require one shifted label per input token")
    labels = labels[-1:] + labels[:-1]
    mask = encoded.get("completion_mask")
    if mask is not None:
        mask = list(mask)
        if len(mask) != len(ids):
            raise ValueError("completion mask and input_ids must have equal lengths")
        mask = mask[-1:] + mask[:-1]
    attention = encoded.get("attention_mask")
    attention = list(attention) if attention is not None else [1] * len(ids)
    if len(attention) != len(ids):
        raise ValueError("performance metrics require a one-dimensional attention mask")
    positions = [i for i, label in enumerate(labels) if label != -100 and (mask is None or mask[i]) and attention[i]]
    first = positions[0] if positions else len(ids)
    prompt = sum(bool(x) for x in attention[:first])
    response = sum(bool(x) for x in attention[first:])
    return prompt, response


def reduce_memory(rows):
    """Reduce allocator peaks by rank maximum and other memory observations by mean."""
    rows = [rows] if isinstance(rows, dict) else rows
    keys = set().union(*(row.keys() for row in rows))
    result = {}
    for key in keys:
        values = [row[key] for row in rows if key in row]
        result[key] = max(values) if "max" in key else sum(values) / len(values)
    return result


class RLPerformanceRecorder:
    def __init__(
        self,
        output_dir,
        device_groups,
        *,
        mode="synchronous",
        run_id=None,
        parameter_sync_step=1,
        clock=perf_counter,
        logger=None,
        metadata=None,
        max_forward_workers=3,
        track_batch_wait=False,
        memory_prefix=None,
    ):
        self.writer = JsonlMetricWriter(Path(output_dir) / "performance.jsonl", run_id=run_id)
        self.summary_writer = JsonlMetricWriter(
            Path(output_dir) / "performance_summary.jsonl", run_id=self.writer.run_id
        )
        self.wall_clock = clock
        self.excluded_seconds = 0.0
        self.clock = lambda: self.wall_clock() - self.excluded_seconds
        self.groups = device_groups
        self.n_gpus = len({rank for _, ranks in device_groups for rank in ranks})
        if not self.n_gpus:
            raise ValueError("missing physical device allocation for RL performance metrics")
        self.mode = mode
        self.logger = logger or logging.getLogger(__name__)
        self.metadata = dict(metadata or {})
        self.max_forward_workers = max_forward_workers
        self.track_batch_wait = track_batch_wait
        self.memory_prefix = memory_prefix
        self.iteration = 0
        self.active = False
        self.prompt_lengths = None
        self.response_lengths = None
        self.truncated = None
        self.assistant_turns = None
        self.memory = None
        self._publish_time = 0.0
        self._publishes = 0
        self.parameter_sync_step = parameter_sync_step
        self.total_tokens = 0
        self.total_seconds = 0.0
        self._forward_executor = None
        self._forward_observers = []

    def begin(self, optimizer_step):
        self.timings = {}
        self.start_step = optimizer_step
        self.prompt_lengths = None
        self.response_lengths = None
        self.truncated = None
        self.assistant_turns = None
        self.memory = None
        self.sample_visits = 0
        self.trained_indices = set()
        self.started = self.clock()
        self.wall_started = self.wall_clock()
        self.excluded_at_start = self.excluded_seconds
        self.active = True

    @contextmanager
    def exclude_logging(self):
        """Exclude reporting I/O from logical time while retaining full wall time."""
        started = self.wall_clock()
        try:
            yield
        finally:
            self.excluded_seconds += self.wall_clock() - started

    @contextmanager
    def stage(self, name):
        started = self.clock()
        try:
            yield
        finally:
            self.timings[name] = self.timings.get(name, 0.0) + self.clock() - started

    def call(self, name, fn, *args, collect_result=False, **kwargs):
        with self.stage(name):
            result = fn(*args, **kwargs)
            # Complete the actual remote work; do not time only RPC submission.
            if getattr(result, "_is_lazy_collect", False):
                if collect_result:
                    # These small side-effect results have no downstream tensor
                    # consumer to surface failures. Waiting alone is insufficient.
                    result = result.result()
                else:
                    result.wait()
            return result

    def launch_forward(self, name, fn, *args, **kwargs):
        """Observe independent RPCs without serializing auxiliary model lanes.

        Each span is submission-to-completion wall time, including queueing. They
        may overlap and must not be added to reconstruct iteration time.
        """
        started = self.clock()
        result = fn(*args, **kwargs)
        if getattr(result, "_is_lazy_collect", False):
            if self._forward_executor is None:
                self._forward_executor = ThreadPoolExecutor(
                    max_workers=self.max_forward_workers, thread_name_prefix="rl-metric"
                )

            def observe():
                result.wait()  # no payload transfer; consumer propagates task errors
                return self.clock() - started

            self._forward_observers.append((name, self._forward_executor.submit(observe)))
        else:
            self.timings[name] = self.timings.get(name, 0.0) + self.clock() - started
        return result

    def join_forwards(self):
        """The advantage consumer already depends on every auxiliary lane."""
        pending, self._forward_observers = self._forward_observers, []
        for name, observer in pending:
            self.timings[name] = self.timings.get(name, 0.0) + observer.result()

    def received(self, samples):
        """Adapt encoded rollout samples; callers may instead supply lengths directly."""
        lengths = [sample_lengths(sample) for sample in samples]
        turns = [getattr(sample, "rollout_infos", {}).get("turns") for sample in samples]
        self.received_lengths(
            [prompt for prompt, _ in lengths],
            [response for _, response in lengths],
            truncated=[bool(getattr(sample, "truncated", False)) for sample in samples],
            assistant_turns=turns if turns and all(isinstance(n, (int, float)) for n in turns) else None,
        )

    def received_lengths(self, prompt_lengths, response_lengths, *, truncated=None, assistant_turns=None):
        """Register a rollout batch without depending on any training/sample class."""
        prompts, responses = list(prompt_lengths), list(response_lengths)
        if not prompts or len(prompts) != len(responses):
            raise ValueError("expected a nonempty batch of paired prompt/response lengths")
        flags = [False] * len(prompts) if truncated is None else list(truncated)
        turns = None if assistant_turns is None else list(assistant_turns)
        if len(flags) != len(prompts) or (turns is not None and len(turns) != len(prompts)):
            raise ValueError("per-sample observations must match the batch length")
        self.prompt_lengths, self.response_lengths = prompts, responses
        self.truncated, self.assistant_turns = flags, turns
        if self.track_batch_wait:
            self.timings["gen"] = self.clock() - self.started

    def record_training(self, sample_indices):
        """Count visits across replay while retaining unique original-batch indices."""
        indices = list(sample_indices)
        if self.prompt_lengths is None or any(not 0 <= index < len(self.prompt_lengths) for index in indices):
            raise ValueError("training indices must refer to the registered rollout batch")
        self.sample_visits += len(indices)
        self.trained_indices.update(indices)

    def finish(
        self, optimizer_step, *, stream_stats=None, resource_metrics=None, memory=None, terminal=False, publications=0
    ):
        if not self.active or self.prompt_lengths is None:
            return
        ended = self.clock()
        wall_elapsed = self.wall_clock() - self.wall_started
        self.active = False
        timings = dict(self.timings, step=ended - self.started)
        extra = {}
        if stream_stats is not None:
            stats = dict(stream_stats)
            if "publish_time" in stats:
                publish_time = stats["publish_time"]
                timings["update_weights"] = publish_time - self._publish_time
                self._publish_time = publish_time
            if "publishes" in stats:
                publications = stats["publishes"] - self._publishes
                self._publishes = stats["publishes"]
            extra.update(
                {
                    f"stream/{key}": value
                    for key, value in stats.items()
                    if isinstance(value, (int, float))
                    and key not in ("total_time", "final_version", "dropped_incomplete")
                }
            )
        extra.update(resource_metrics or {})
        metrics = compute_rl_performance_metrics(self.prompt_lengths, self.response_lengths, timings, self.n_gpus)
        metrics.update(extra)
        turns = self.assistant_turns
        if turns is not None and all(isinstance(n, (int, float)) for n in turns):
            # Count assistant generations, independently of user/tool message counts.
            metrics.update(
                {
                    "rollout/assistant_turns/mean": sum(turns) / len(turns),
                    "rollout/assistant_turns/min": min(turns),
                    "rollout/assistant_turns/max": max(turns),
                }
            )
        metrics.update(memory or {})
        if self.memory_prefix:
            metrics.update({f"{self.memory_prefix}/{key}": value for key, value in (memory or {}).items()})
        metrics.update(
            {
                "training/optimizer_step_start": self.start_step,
                "training/weight_publications_this_iteration": publications,
                "training/optimizer_step": optimizer_step,
                "training/optimizer_steps_this_iteration": optimizer_step - self.start_step,
                "training/sample_visits": self.sample_visits,
                "training/unique_samples_trained": len(self.trained_indices),
                "training/samples_not_trained": len(self.prompt_lengths) - len(self.trained_indices),
                "response/truncated_ratio": sum(bool(flag) for flag in self.truncated) / len(self.truncated),
                "perf/gpu_seconds": timings["step"] * self.n_gpus,
                "observability/iteration_wall_seconds": wall_elapsed,
                "observability/excluded_logging_seconds": self.excluded_seconds - self.excluded_at_start,
            }
        )
        self.iteration += 1
        self.writer.write(
            self.iteration,
            metrics,
            schema_version=1,
            step_unit="rl_iteration",
            iteration_scope="one_consumed_rollout_batch",
            parameter_sync_step=self.parameter_sync_step,
            parameter_sync_step_unit="optimizer_update",
            mode=self.mode,
            device_groups=self.groups,
            token_scope="initial_prompt_plus_full_response_including_tool_observations",
            terminal=terminal,
            **self.metadata,
            unavailable_metrics=["perf/mfu/actor"],
            unsupported_mfu_reason="no validated model FLOPs and device peak FLOPs estimator",
        )
        self.total_tokens += metrics["perf/total_num_tokens"]
        self.total_seconds += timings["step"]
        self.logger.info(
            "[perf] iteration=%s step=%s seconds=%.3f tokens=%s tok/s/GPU=%.3f",
            self.iteration,
            optimizer_step,
            timings["step"],
            metrics["perf/total_num_tokens"],
            metrics["perf/throughput"],
        )

    def write_summary(self, optimizer_step, *, stream_stats=None, status="completed"):
        """Call after driver teardown: terminal drops/final_version now exist."""
        data = {
            "completed_iterations": self.iteration,
            "optimizer_step": optimizer_step,
            "total_num_tokens": self.total_tokens,
            "sum_iteration_seconds": self.total_seconds,
            "allocated_gpus": self.n_gpus,
            "throughput": (self.total_tokens / (self.total_seconds * self.n_gpus) if self.total_seconds else None),
        }
        if stream_stats is not None:
            data.update({f"stream/{k}": v for k, v in stream_stats.items() if isinstance(v, (int, float))})
        self.summary_writer.write(self.iteration, data, step_unit="run_summary", status=status)

    def close(self):
        self.active = False
        self.writer.close()
        self.summary_writer.close()
        if self._forward_executor is not None:
            self._forward_executor.shutdown(wait=False, cancel_futures=True)
