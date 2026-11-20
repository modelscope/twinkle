# Copyright (c) ModelScope Contributors. All rights reserved.
"""Adapter publication must respect offloaded FSDP shards and preserve the trainer."""

import os
import time
from contextlib import ExitStack
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from peft import LoraConfig, get_peft_model

from twinkle.model.transformers.strategy.native_fsdp import NativeFSDPStrategy
from twinkle.model.transformers.transformers import TransformersModel
from twinkle.utils.framework import Torch


def _wrapper(model):
    strategy = NativeFSDPStrategy(enable_ep=False)
    strategy.unwrap_model = lambda module: module
    return SimpleNamespace(
        model=model,
        strategy=strategy,
        hf_config=SimpleNamespace(tie_word_embeddings=False),
        _get_default_group=lambda: "default",
        _lazy_wrap_model=lambda: None,
    )


def _lora_model():
    base = torch.nn.Sequential()
    base.add_module("linear", torch.nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        base.linear.weight.zero_()
    model = get_peft_model(base, LoraConfig(target_modules=["linear"], r=1, lora_alpha=1))
    layer = model.base_model.model.linear
    with torch.no_grad():
        layer.lora_A["default"].weight.fill_(1)
        layer.lora_B["default"].weight.fill_(2)
    return model, layer


@pytest.mark.parametrize("close_early", [False, True])
def test_lending_export_preserves_merge_until_reclaim(close_early):
    model, layer = _lora_model()
    wrapper = _wrapper(model)
    TransformersModel.prepare_for_lending(wrapper)
    # Repeated preparation must not add the adapter delta twice.
    TransformersModel.prepare_for_lending(wrapper)
    torch.testing.assert_close(layer.base_layer.weight, torch.full((2, 2), 2.0))
    with (
        patch.object(model, "merge_adapter", side_effect=AssertionError("merge during export")),
        patch.object(model, "unmerge_adapter", side_effect=AssertionError("unmerge before reclaim")),
    ):
        weights = TransformersModel._get_weight_generator(wrapper, base_sync_done=True, merge_and_sync=True)
        name, tensor = next(weights)
        assert name == "linear.weight"
        torch.testing.assert_close(tensor, torch.full((2, 2), 2.0))
        if close_early:
            weights.close()
        else:
            assert list(weights) == []
    assert layer.merged
    TransformersModel.restore_after_lending(wrapper)
    TransformersModel.restore_after_lending(wrapper)
    assert not layer.merged
    torch.testing.assert_close(layer.base_layer.weight, torch.zeros(2, 2))


def test_partial_prepare_failure_can_be_rolled_back():
    model, layer = _lora_model()
    wrapper = _wrapper(model)
    original = model.merge_adapter

    def merge_then_fail():
        original()
        raise RuntimeError("merge failed")

    with patch.object(model, "merge_adapter", side_effect=merge_then_fail):
        with pytest.raises(RuntimeError, match="merge failed"):
            TransformersModel.prepare_for_lending(wrapper)
    TransformersModel.restore_after_lending(wrapper)
    assert not layer.merged
    torch.testing.assert_close(layer.base_layer.weight, torch.zeros(2, 2))


def test_full_model_needs_no_adapter_preparation():
    model = torch.nn.Linear(2, 2)
    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    wrapper = _wrapper(model)
    TransformersModel.prepare_for_lending(wrapper)
    TransformersModel.restore_after_lending(wrapper)
    assert getattr(wrapper, "_lent_merged_model", None) is None
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, original[name])


@pytest.mark.parametrize("close_early", [False, True])
def test_ordinary_export_still_unmerges_when_finished(close_early):
    model, layer = _lora_model()
    wrapper = _wrapper(model)
    weights = TransformersModel._get_weight_generator(wrapper, base_sync_done=True, merge_and_sync=True)
    _, tensor = next(weights)
    torch.testing.assert_close(tensor, torch.full((2, 2), 2.0))
    assert layer.merged
    if close_early:
        weights.close()
    else:
        assert list(weights) == []
    assert not layer.merged
    torch.testing.assert_close(layer.base_layer.weight, torch.zeros(2, 2))


def _distributed_export(rank, directory, device_type):
    import torch.distributed as dist
    from packaging.version import Version
    from peft import LoraConfig, get_peft_model
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import Shard, distribute_tensor

    from twinkle.model.transformers.transformers import TransformersModel

    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
    if device_type == "npu":
        import torch_npu  # noqa: F401

        torch.npu.set_device(rank)
    elif device_type == "cuda":
        torch.cuda.set_device(rank)
    backend = {"cpu": "gloo", "npu": "hccl", "cuda": "nccl"}[device_type]
    device = torch.device(device_type, rank) if device_type != "cpu" else torch.device("cpu")
    dist.init_process_group(
        backend, init_method=f"file://{directory}/pg", rank=rank, world_size=2, timeout=timedelta(seconds=45)
    )
    device_context = ExitStack()
    device_context.enter_context(patch.object(Torch, "get_device", return_value=str(device)))
    if device_type == "cpu":
        device_context.enter_context(patch.object(Torch, "_device_module", return_value=None))
    try:
        mesh = init_device_mesh(device_type, (2,))
        base = torch.nn.Sequential()
        base.add_module("first", torch.nn.Linear(8, 8, bias=False))
        base.add_module("second", torch.nn.Linear(8, 8, bias=False))
        with torch.no_grad():
            for parameter in base.parameters():
                parameter.zero_()
        model = get_peft_model(base, LoraConfig(target_modules=["first", "second"], r=4, lora_alpha=4))
        layers = [model.base_model.model.first, model.base_model.model.second]
        with torch.no_grad():
            for index, layer in enumerate(layers):
                layer.lora_A["default"].weight.fill_(1)
                layer.lora_B["default"].weight.fill_(2 + index)
        model.to(device)
        use_fsdp = device_type != "cpu" or Version(torch.__version__.split("+")[0]) >= Version("2.11")
        if use_fsdp:
            for layer in layers:
                fully_shard(layer, mesh=mesh)
            fully_shard(model, mesh=mesh)
        else:
            # CPU FSDP2 execution needs torch >= 2.11; older versions still exercise
            # real distributed shards and merging through the same export path.
            for module in model.modules():
                for name, parameter in list(module.named_parameters(recurse=False)):
                    module._parameters[name] = torch.nn.Parameter(
                        distribute_tensor(parameter.detach(), mesh, [Shard(0)]),
                        requires_grad=parameter.requires_grad,
                    )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        if use_fsdp:
            model(torch.ones(2, 8, device=device)).sum().backward()
            optimizer.zero_grad(set_to_none=True)
        original_params = dict(model.named_parameters())
        original_shards = {name: parameter.to_local().cpu().clone() for name, parameter in original_params.items()}
        wrapper = _wrapper(model)

        # Merge while resident, then export several times without moving any model layer.
        for _ in range(2):
            TransformersModel.prepare_for_lending(wrapper)
            model.to("cpu")
            with (
                patch.object(model, "merge_adapter", side_effect=AssertionError("merge after offload")),
                patch.object(model, "unmerge_adapter", side_effect=AssertionError("unmerge before reload")),
                patch.object(model, "to", side_effect=AssertionError("model move during export")),
            ):
                weights = TransformersModel._get_weight_generator(wrapper, base_sync_done=True, merge_and_sync=True)
                published = {name: tensor.detach().cpu().clone() for name, tensor in weights}
                assert set(published) == {"first.weight", "second.weight"}
                torch.testing.assert_close(published["first.weight"], torch.full((8, 8), 8.0))
                torch.testing.assert_close(published["second.weight"], torch.full((8, 8), 12.0))
                # An interrupted export also leaves the lending-owned merge intact.
                weights = TransformersModel._get_weight_generator(wrapper, base_sync_done=True, merge_and_sync=True)
                next(weights)
                weights.close()
            assert all(layer.merged for layer in layers)
            assert all(parameter.device.type == "cpu" for parameter in model.parameters())
            model.to(device)
            TransformersModel.restore_after_lending(wrapper)
            assert all(not layer.merged for layer in layers)
            for name, parameter in model.named_parameters():
                assert parameter is original_params[name]
                torch.testing.assert_close(parameter.to_local().cpu(), original_shards[name], rtol=0, atol=0)
            assert {id(parameter) for group in optimizer.param_groups for parameter in group["params"]} == {
                id(parameter) for parameter in model.parameters()
            }
        if use_fsdp:
            # Reclaiming the device must leave FSDP's next forward/backward usable.
            model(torch.ones(2, 8, device=device)).sum().backward()
            optimizer.step()
            assert any(
                parameter.requires_grad and not torch.equal(parameter.to_local().cpu(), original_shards[name])
                for name, parameter in model.named_parameters()
            )
    finally:
        device_context.close()
        dist.destroy_process_group()


def _run_distributed_export(tmp_path, device_type):
    context = torch.multiprocessing.spawn(_distributed_export, args=(str(tmp_path), device_type), nprocs=2, join=False)
    try:
        deadline = time.monotonic() + 100
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("Offloaded adapter export did not finish within 100 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def test_distributed_offloaded_lora_export_preserves_shards_and_optimizer(tmp_path):
    _run_distributed_export(tmp_path, "cpu")


@pytest.mark.accel(2)
def test_accelerator_offloaded_lora_export_uses_the_original_mesh(tmp_path):
    from twinkle import Platform

    device_type = Platform.device_prefix()
    device_module = Platform.get_platform().device_module()
    if device_type not in ("npu", "cuda") or device_module is None or device_module.device_count() < 2:
        pytest.skip("Two visible CUDA or Ascend devices are required")
    _run_distributed_export(tmp_path, device_type)
