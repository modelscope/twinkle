"""Backend layout estimates, replica dispatch and remote-reference alignment."""

from types import SimpleNamespace

import pytest
import torch

from twinkle import remote_function
from twinkle.infra.collectors import tensor_dict
from twinkle.model.base import TrainableModel
from twinkle.model.dp_balance import _estimate_megatron_ranks, estimate_workload, plan_call
from twinkle.processor import InputProcessor
from twinkle.utils import DeviceMesh
from .test_dp_balance import _call_inputs, _FakeModel


@pytest.mark.parametrize("padding_free", [False, True])
@pytest.mark.parametrize("variable", [False, True])
@pytest.mark.parametrize("multiple", [1, 4])
def test_megatron_estimate_matches_actual_processor_layout(padding_free, variable, multiple):
    rank_lengths = [[9, 2, 7, 3], [6, 5, 4, 2]]
    estimate = _estimate_megatron_ranks(
        rank_lengths,
        micro_batch_size=2,
        padding_free=padding_free,
        variable_seq_lengths=variable,
        sequence_multiple=multiple,
        workload_fn=estimate_workload,
    )
    processor = InputProcessor(framework="megatron", padding_free=padding_free)
    expected = []
    computed = []
    for lengths in rank_lengths:
        rounded = [((length + multiple - 1) // multiple) * multiple for length in lengths]
        rows = [{"input_ids": torch.ones(length, dtype=torch.long)} for length in rounded]
        batches = processor.collate_fn(rows, micro_batch_size=2, variable_seq_lengths=variable)
        computed.append(sum(batch["input_ids"].numel() for batch in batches))
        expected.append(
            sum(estimate_workload(length) for length in rounded)
            if padding_free
            else sum(batch["input_ids"].shape[0] * estimate_workload(batch["input_ids"].shape[-1]) for batch in batches)
        )
    assert estimate.workloads == expected
    assert estimate.computed_tokens == computed
    assert estimate.micro_batches == 2


def mega_spec(**extra):
    return dict(
        enabled=True,
        backend="megatron",
        hidden_size=1024,
        padding_free=True,
        group_size=1,
        variable_seq_lengths=True,
        sequence_multiple=1,
        **extra,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"micro_batch_size": 3},
        {"micro_batch_size": 2, "dynamic_batching": True, "max_tokens_per_micro_batch": 4096},
    ],
)
def test_megatron_skips_unsupported_or_unequally_weighted_schedules(kwargs):
    assert plan_call([900, 800, 700, 600, 50, 40, 30, 20], 2, mega_spec(), kwargs) is None


def test_megatron_keeps_all_replica_peers_on_the_same_shard():
    from twinkle.infra.dispatch import _dispatch_args

    mesh = DeviceMesh.from_sizes(world_size=16, dp_size=2, tp_size=2, pp_size=2, cp_size=2, device_type="cpu")
    model = _FakeModel()
    model.spec = mega_spec()
    model.device_mesh = mesh
    inputs = _call_inputs([900, 800, 700, 600, 50, 40, 30, 20])
    plan, args, kwargs = model._dp_balance_call(
        "forward_backward", (), {"inputs": inputs, "advantages": list(range(8))}
    )
    assert plan.reorders
    for worker, _, shard in _dispatch_args(list(range(16)), "slice_dp", "all", mesh, args, kwargs):
        rank = mesh.get_data_rank_from_global_rank(worker)
        assert shard["advantages"] == plan.order[rank * 4 : (rank + 1) * 4]
        assert len(shard["inputs"]) == 4


def test_backend_spec_and_switch_are_owned_by_trainable_model():
    from twinkle.loss import GRPOLoss
    from twinkle.model.megatron.megatron import MegatronModel
    from twinkle.model.transformers.transformers import TransformersModel

    assert MegatronModel._dp_balance_spec is TrainableModel._dp_balance_spec
    assert TransformersModel._dp_balance_spec is TrainableModel._dp_balance_spec
    mesh = DeviceMesh.from_sizes(world_size=8, dp_size=2, tp_size=2, cp_size=2, device_type="cpu")
    model = SimpleNamespace(
        optimizer_group={"": SimpleNamespace(get_loss=lambda key: GRPOLoss(), processor=InputProcessor())},
        _get_default_group=lambda: "",
        hf_config=SimpleNamespace(hidden_size=256),
        device_mesh=mesh,
        _dp_balance_backend="megatron",
        variable_seq_lengths=True,
    )
    TrainableModel._init_dp_balance(model, False)
    spec = TrainableModel._dp_balance_spec.__wrapped__(model)
    assert not spec["enabled"]
    assert spec["backend"] == "megatron"
    assert spec["sequence_multiple"] == 4


class _RefWorker:
    def __init__(self):
        from twinkle import infra

        infra._mode = "ray"

    @remote_function(dispatch="slice_dp", collect=tensor_dict)
    def forward_backward(self, *, inputs, advantages, _dp_balance=None):
        assert len(inputs) == 4  # driver already sliced; worker must not slice twice
        assert [row["sample_id"] for row in inputs] == advantages
        assert _dp_balance is not None
        return {"sample_id": torch.tensor(advantages)}


def test_ray_reference_batch_balances_without_fetching_payloads_on_driver(monkeypatch):
    import ray

    from twinkle import infra

    ray.init(ignore_reinit_error=True, num_cpus=2, logging_level="ERROR")
    actors = [ray.remote(_RefWorker).remote() for _ in range(2)]
    try:
        model = _FakeModel()
        model.device_mesh = DeviceMesh.from_sizes(world_size=2, dp_size=2, device_type="cpu")
        model._actors, model._max_concurrency = actors, None
        refs = [
            ray.put({"sample_id": i, "input_ids": [1] * length})
            for i, length in enumerate([900, 800, 700, 600, 50, 40, 30, 20])
        ]
        original_get = ray.get

        def checked_get(ref, *args, **kwargs):
            assert ref not in refs, "feature payload fetched on the driver"
            return original_get(ref, *args, **kwargs)

        monkeypatch.setattr(ray, "get", checked_get)
        monkeypatch.setattr(infra, "_mode", "ray")
        result = _RefWorker.forward_backward(
            model, inputs=refs, advantages=list(range(8)), input_lengths=[900, 800, 700, 600, 50, 40, 30, 20]
        )
        assert result()["sample_id"].tolist() == list(range(8))
    finally:
        for actor in actors:
            ray.kill(actor)


def test_local_subclass_delegation_is_balanced_only_once(monkeypatch):
    from twinkle import infra

    monkeypatch.setattr(infra, "_mode", "local")

    class Nested:
        calls = 0

        def _dp_balance_local(self, run, args, kwargs):
            self.calls += 1
            return run(*args, **kwargs)

        @remote_function(dispatch="slice_dp")
        def forward_only(self, *, inputs):
            return self.forward_backward(inputs=inputs)

        @remote_function(dispatch="slice_dp")
        def forward_backward(self, *, inputs):
            return inputs

    model = Nested()
    assert model.forward_only(inputs=[1, 2]) == [1, 2]
    assert model.calls == 1
    assert not model._dp_balance_running


def test_disabled_local_model_consumes_only_compute_metadata(monkeypatch):
    from twinkle import infra

    monkeypatch.setattr(infra, "_mode", "local")

    class Disabled:
        _dp_balance_local = TrainableModel._dp_balance_local
        _dp_balance_off = True

        @remote_function(dispatch="slice_dp")
        def forward_backward(self, *, inputs):
            return inputs

        @remote_function()
        def other(self, *, input_lengths):
            return input_lengths

    model = Disabled()
    assert model.forward_backward(inputs=[1], input_lengths=[10]) == [1]
    assert model.other(input_lengths=[10]) == [10]


def test_opaque_whole_batch_reference_skips_instead_of_misaligning(monkeypatch):
    import ray

    monkeypatch.setattr(ray, "get", lambda *args, **kwargs: pytest.fail("feature payload fetched"))
    model = _FakeModel()
    assert (
        model._dp_balance_call(
            "forward_backward",
            (),
            {"inputs": _call_inputs([900, 800, 50, 40]), "advantages": ray.ObjectRef.from_random()},
        )
        is None
    )
    refs = [ray.ObjectRef.from_random() for _ in range(4)]
    assert model._dp_balance_call("forward_backward", (), {"inputs": refs}) is None


class _LocalMegatron:
    _dp_balance_local = TrainableModel._dp_balance_local

    def __init__(self, mesh):
        self.device_mesh = mesh
        self.calls = 0

    def _dp_balance_spec(self, **kwargs):
        return mega_spec()

    @remote_function(dispatch="slice_dp")
    def forward_only(self, *, inputs, **kwargs):
        return self.forward_backward(inputs=inputs, **kwargs)

    @remote_function(dispatch="slice_dp")
    def forward_backward(self, *, inputs, advantages, _dp_balance=None):
        self.calls += 1
        lengths = [len(row["input_ids"]) for row in inputs]
        assert lengths == advantages.tolist()
        assert _dp_balance["reordered"]
        return {
            "logps": [torch.tensor(lengths[start : start + 2])[:, None].expand(-1, 3) for start in (0, 2)],
            "loss": sum(lengths),
        }


def _local_megatron_worker(rank, rendezvous, result_dir):
    import os

    import torch.distributed as dist

    from twinkle import infra

    os.environ.update(RANK=str(rank), WORLD_SIZE="2")
    infra._mode = "local"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        lengths = [900, 800, 700, 600, 50, 40, 30, 20][rank * 4 : (rank + 1) * 4]
        mesh = DeviceMesh.from_sizes(world_size=2, dp_size=2, device_type="cpu")
        model = _LocalMegatron(mesh)
        outputs = model.forward_only(inputs=_call_inputs(lengths), advantages=torch.tensor(lengths))
        assert outputs["logps"][:, 0].tolist() == lengths
        assert model.calls == 1
        torch.save(True, f"{result_dir}/{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_local_megatron_layout_restores_micro_batch_outputs_and_delegates_once(tmp_path):
    torch.multiprocessing.spawn(
        _local_megatron_worker, args=(str(tmp_path / "gloo"), str(tmp_path)), nprocs=2, join=True
    )
    assert all(torch.load(tmp_path / f"{rank}.pt") for rank in range(2))


def test_undeclared_backend_skips_instead_of_assuming_transformers_layout():
    spec = mega_spec()
    spec["backend"] = None
    assert plan_call([9, 8, 1, 1], 2, spec, {}) is None
