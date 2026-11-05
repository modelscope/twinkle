"""Numerical coverage of the loss families used by backend-owned fixed micro-batches."""

from types import SimpleNamespace

import pytest
import torch

from twinkle.loss.dpo import DPOLoss
from twinkle.loss.cross_entropy import CrossEntropyLoss
from twinkle.loss.chunked_cross_entropy import ChunkedCrossEntropyLoss
from twinkle.loss.gkd import GKDLoss
from twinkle.loss.liger_fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
from twinkle.loss.mopd import MOPDLoss
from twinkle.loss.reward import RewardLoss
from twinkle.loss.seq_cls import SeqClsLoss
from .test_grpo_micro_batch import TrainingHarness, data


class Harness(TrainingHarness):
    def forward(self, *, inputs, **kwargs):
        super().forward(inputs=inputs, **kwargs)
        status = self.optimizer_group["default"].train_status
        scores = status.outputs["logps"]
        loss = self.optimizer_group["default"].loss_instance
        if isinstance(loss, RewardLoss):
            status.outputs["logits"] = scores[:, :1]
        elif isinstance(loss, SeqClsLoss):
            status.inputs["labels"] = torch.tensor([row["input_ids"][0] % 2 for row in inputs])
            status.outputs["logits"] = torch.cat([scores[:, :1], -scores[:, :1]], dim=-1)
        else:
            status.outputs["logits"] = torch.stack([scores, -scores], dim=-1)
        status.outputs["values"] = scores
        return dict(status.outputs)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DPOLoss(reference_free=True),
        RewardLoss,
        lambda: SeqClsLoss(problem_type="single_label_classification", num_labels=2),
    ],
)
def test_sample_mean_losses_match_whole_batch_with_a_short_micro_batch(factory):
    baseline, internal = Harness(factory()), Harness(factory())
    rows = data()[:6]
    baseline.run(rows)
    internal.run(rows, micro_batch_size=4)
    torch.testing.assert_close(internal.normalized_gradient(), baseline.normalized_gradient())


@pytest.mark.parametrize("factory", [GKDLoss, LigerFusedLinearCrossEntropyLoss])
def test_token_losses_match_explicit_fixed_micro_batches(factory):
    baseline, internal = Harness(factory()), Harness(factory())
    rows = data()[:6]
    for start in range(0, len(rows), 2):
        micro = rows[start : start + 2]
        kwargs = {
            "old_values": [[0.0] * len(row["labels"]) for row in micro],
            "returns": [[0.2] * len(row["labels"]) for row in micro],
            "teacher_logits": torch.zeros(len(micro), max(len(row["labels"]) for row in micro), 2),
        }
        baseline.run(micro, **kwargs)
    internal.run(
        rows,
        micro_batch_size=2,
        old_values=[[0.0] * len(row["labels"]) for row in rows],
        returns=[[0.2] * len(row["labels"]) for row in rows],
        teacher_logits=torch.zeros(len(rows), max(len(row["labels"]) for row in rows), 2),
    )
    torch.testing.assert_close(internal.normalized_gradient(), baseline.normalized_gradient())


def test_multi_teacher_slicing_preserves_teacher_axis_and_global_weights():
    # Teacher count deliberately equals batch size: the outer teacher axis must never be sliced.
    kwargs = {
        "teacher_logps": [[[1.0], [2.0]], [[3.0], [4.0]]],
        "teacher_weights": [0.25, 0.75],
        "advantages": [5.0, 6.0],
    }
    micro = MOPDLoss().micro_batch_kwargs(kwargs, [1], 2)
    assert micro["teacher_logps"] == [[[2.0]], [[4.0]]]
    assert micro["teacher_weights"] == [0.25, 0.75]
    assert micro["advantages"] == [6.0]


def test_multi_teacher_signals_are_sliced_by_samples_across_dp_then_micro_batches():
    from twinkle.infra.dispatch import _dispatch_args

    mesh = SimpleNamespace(
        world_size=2,
        get_data_rank_from_global_rank=lambda rank: rank,
        get_slice=lambda length, rank: slice(rank * length // 2, (rank + 1) * length // 2),
    )
    kwargs = {
        "inputs": list(range(8)),
        "teacher_logps": {0: [[i] for i in range(8)], 1: [[i + 10] for i in range(8)]},
        "teacher_weights": (0.25, 0.75),
    }
    dispatched = _dispatch_args([None, None], "slice_dp", "all", mesh, (), kwargs)
    rank_one = dispatched[1][2]
    assert rank_one["inputs"] == [4, 5, 6, 7]
    assert rank_one["teacher_weights"] == (0.25, 0.75)
    micro = MOPDLoss().micro_batch_kwargs(rank_one, [2, 3], 4)
    assert micro["teacher_logps"] == {0: [[6], [7]], 1: [[16], [17]]}


@pytest.mark.parametrize("factory", [
    CrossEntropyLoss,
    lambda **kwargs: ChunkedCrossEntropyLoss(chunk_size=4, **kwargs),
    LigerFusedLinearCrossEntropyLoss,
])
def test_cross_entropy_counts_tokens_inside_packing_units(factory):
    rows = [[{"labels": [-100, 1]}, {"labels": [1, 2]}], [{"labels": [1]}]]
    assert factory(reduction="mean").micro_batch_scale(rows, [0]) == pytest.approx(3 / 4)
    assert factory(reduction="sum").batch_normalizer(rows) == 4


@pytest.mark.parametrize('task, num_samples, micro_batch_size', [
    ('seq_cls', 5, 2),
    ('seq_cls', 4, 2),
    ('causal_lm', 6, None),
    ('causal_lm', 8, 4),
    ('causal_lm', 6, 4),
])
def test_native_megatron_mean_losses_keep_existing_micro_batch_reduction(
    monkeypatch, task, num_samples, micro_batch_size,
):
    """Keep native micro-batch means equally weighted, including a short final micro-batch."""
    import inspect
    import sys
    from contextlib import nullcontext
    from types import ModuleType

    from twinkle import Platform
    from twinkle.model.megatron.megatron import MegatronModel
    from twinkle.model.optimizer_group import BaseOptimizerGroup, TrainStatus
    from twinkle.processor import InputProcessor

    class Processor(InputProcessor):
        def __call__(self, rows, *, micro_batch_size, **kwargs):
            return [
                {
                    "input_ids": torch.tensor([row["input_ids"] for row in rows[start : start + micro_batch_size]]),
                    "labels": torch.tensor([row["labels"] for row in rows[start : start + micro_batch_size]]),
                }
                for start in range(0, len(rows), micro_batch_size)
            ]

        def postprocess_tensor_cp(self, value, **kwargs):
            return value

        def unpack_inputs(self, inputs, **kwargs):
            return inputs

        def unpack_packed_sequences(self, inputs, outputs):
            return inputs, outputs

    class Chunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.12, dtype=torch.float64))
            self.config, self.vp_stage = SimpleNamespace(), None

        def forward(self, input_ids):
            scores = input_ids * self.weight
            return torch.stack([scores, -scores], dim=-1)

    def schedule(*, forward_step_func, data_iterator, model, num_microbatches, **kwargs):
        losses, reports, counts = [], [], []
        for _ in range(num_microbatches):
            output, calculate = forward_step_func(data_iterator, model[0])
            loss, count, report = calculate(output)
            losses.append(loss)
            counts.append(count)
            reports.append(report)
        (sum(losses) / sum(counts)).backward()
        return reports

    core = ModuleType("megatron.core")
    core.parallel_state = SimpleNamespace(
        is_pipeline_last_stage=lambda *args: True,
        get_pipeline_model_parallel_world_size=lambda: 1,
        get_tensor_model_parallel_world_size=lambda: 1,
    )
    pipeline = ModuleType("megatron.core.pipeline_parallel")
    pipeline.get_forward_backward_func = lambda: schedule
    monkeypatch.setitem(sys.modules, "megatron", ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.pipeline_parallel", pipeline)
    monkeypatch.setattr(Platform, "get_local_device", lambda: torch.device("cpu"))
    loss_fn = (SeqClsLoss(problem_type="single_label_classification", num_labels=2)
               if task == 'seq_cls' else DPOLoss(reference_free=True))
    baseline, internal = Chunk(), Chunk()
    rows = [{"input_ids": [index + 1], "labels": index % 2 if task == 'seq_cls' else [1]}
            for index in range(num_samples)]
    effective_micro_batch_size = micro_batch_size or 2
    losses = []
    for start in range(0, len(rows), effective_micro_batch_size):
        micro = rows[start:start + effective_micro_batch_size]
        ids = torch.tensor([row["input_ids"] for row in micro])
        logits = baseline(ids)
        if task == 'seq_cls':
            logits = logits[:, -1]
        losses.append(loss_fn(
            {"labels": torch.tensor([row["labels"] for row in micro])}, {"logits": logits})["loss"])
    expected = torch.stack(losses).mean()
    expected.backward()
    group = BaseOptimizerGroup(loss_instance=loss_fn, processor=Processor(), train_status=TrainStatus())
    runtime = SimpleNamespace(
        _lazy_wrap_model=lambda: None,
        _lazy_finish_param_config=lambda: None,
        _get_default_group=lambda: "default",
        _not_encoded=lambda inputs: False,
        _router_replay_enabled=False,
        _maybe_apply_router_replay=lambda: None,
        _mtp=SimpleNamespace(note_num_microbatches=lambda count: None),
        _mtp_training_enabled=lambda **kwargs: False,
        optimizer_group={"default": group},
        model=[internal],
        device_mesh=SimpleNamespace(cp_world_size=1, tp_world_size=1, vpp_size=1),
        variable_seq_lengths=False,
        strategy=SimpleNamespace(
            sequence_parallel=False,
            unwrap_model=lambda model: model,
            transactional_grad_schedule=lambda *args, **kwargs: nullcontext(),
            reduce_loss=lambda loss, count, logits, logps: (
                loss, count, {"loss": loss.detach(), "num_tokens": count}
            ),
        ),
    )
    result = inspect.unwrap(MegatronModel.forward_backward)(
        runtime, inputs=rows, micro_batch_size=micro_batch_size, task=task)
    torch.testing.assert_close(internal.weight.grad, baseline.weight.grad)
    assert result["loss"] == pytest.approx(expected.item())
