# Copyright (c) ModelScope Contributors. All rights reserved.
"""New replicas on lent trainer devices must not join the standalone HCCL/NCCL group."""

import threading
from types import SimpleNamespace as NS

import pytest

from twinkle.checkpoint_engine.manager import CheckpointEngineManager


def _manager(monkeypatch, *, shared=True, prefix="npu", fail=False):
    events = []
    manager = CheckpointEngineManager.__new__(CheckpointEngineManager)
    manager.mode = "standalone"
    manager.backend = "hccl" if prefix == "npu" else "nccl"
    manager.backend_cls = NS(backend_name=manager.backend)
    manager.sync_backend = "full"
    manager.channel = "main"
    manager._sync_lock = threading.RLock()
    manager._auth_key = "auth"
    manager._members = ("dedicated",)
    manager._merge_and_sync = True
    manager.base_sync_done = True
    manager._peft_config = {"r": 8}
    manager._model_keys = ["weight"]
    manager._version = 3

    def record(name, **kwargs):
        events.append((name, kwargs))

    def send(*args, **kwargs):
        record("send", **kwargs)
        if fail:
            raise RuntimeError("sender failed")

    def activate(backend, channel, auth_key):
        record("activate", backend=backend, channel=channel)
        return ["new-replica"]

    manager.model = NS(
        direct_weight_sync_device=lambda: [f"{prefix}0", f"{prefix}1"],
        send_weights_to_vllm=send,
        activate_checkpoint_engine=activate,
        reset_checkpoint_engine=lambda channel, backend: record("reset_model", channel=channel, backend=backend),
    )
    # Only the new replicas should be touched, never the existing sampler handle.
    manager.sampler = NS()
    ranks = (0, 1) if shared else (2, 3)
    sub = NS(
        direct_weight_sync_metadata=lambda: [
            [{"device_uuid": f"{prefix}{rank}", "endpoint": str(rank)} for rank in ranks]
        ],
        receive_weights_direct=lambda *args, **kwargs: record("receive", **kwargs),
        activate_checkpoint_engine=activate,
        reset_checkpoint_engine=lambda channel, backend: record("reset_receiver", channel=channel, backend=backend),
    )

    def transfer(session, receiver, count, merge_and_sync, base_sync_done):
        assert receiver is sub
        record(
            "transfer",
            backend=session.backend,
            channel=session.channel,
            count=count,
            base_sync_done=base_sync_done,
            merge_and_sync=merge_and_sync,
        )

    monkeypatch.setattr(CheckpointEngineManager, "_transfer", transfer)
    monkeypatch.delenv("TWINKLE_DISABLE_DIRECT_VLLM_SYNC", raising=False)
    return manager, sub, events


def _assert_main_unchanged(manager, sampler, backend):
    assert manager.sampler is sampler
    assert manager.mode == "standalone"
    assert manager.backend == backend
    assert manager.backend_cls.backend_name == backend
    assert manager.channel == "main"
    assert manager._members == ("dedicated",)
    assert manager.base_sync_done
    assert manager._version == 3


@pytest.mark.parametrize("prefix", ["npu", "cuda"])
def test_lent_replica_warm_uses_device_paired_direct_transfer(monkeypatch, prefix):
    manager, sub, events = _manager(monkeypatch, prefix=prefix)
    sampler, backend = manager.sampler, manager.backend

    manager.warm(sub)

    assert sorted(name for name, _ in events) == ["receive", "reset_model", "reset_receiver", "send"]
    send = next(kwargs for name, kwargs in events if name == "send")
    assert send["channel"].startswith("main-warm-")
    # True selects the exporter's merged LoRA path after training has begun.
    assert send["base_sync_done"] is True
    assert send["merge_and_sync"] is True
    resets = [kwargs for name, kwargs in events if name.startswith("reset_")]
    assert resets == [{"channel": send["channel"], "backend": "ipc"}] * 2
    _assert_main_unchanged(manager, sampler, backend)


@pytest.mark.parametrize("shared", [False, True])
def test_warm_checkpoint_transport_matches_physical_placement(monkeypatch, shared):
    manager, sub, events = _manager(monkeypatch, shared=shared)
    sampler, backend = manager.sampler, manager.backend
    monkeypatch.setenv("TWINKLE_DISABLE_DIRECT_VLLM_SYNC", "1")

    manager.warm(sub)

    expected_backend = "ipc" if shared else "hccl"
    transfers = [kwargs for name, kwargs in events if name == "transfer"]
    assert len(transfers) == 1
    assert transfers[0]["backend"] == expected_backend
    assert transfers[0]["count"] == 1
    assert transfers[0]["base_sync_done"] is True
    assert all(kwargs["backend"] == expected_backend for name, kwargs in events if name.startswith("reset_"))
    _assert_main_unchanged(manager, sampler, backend)


@pytest.mark.parametrize("shared", [False, True])
def test_warm_lora_only_seeds_base_then_adapter(monkeypatch, shared):
    manager, sub, events = _manager(monkeypatch, shared=shared)
    sampler, backend = manager.sampler, manager.backend
    manager._merge_and_sync = False

    manager.warm(sub)

    sends = [kwargs for name, kwargs in events if name == ("send" if shared else "transfer")]
    assert [kwargs["base_sync_done"] for kwargs in sends] == [False, True]
    assert all(kwargs["merge_and_sync"] is False for kwargs in sends)
    if shared:
        received = [kwargs["peft_config"] for name, kwargs in events if name == "receive"]
        assert received == [None, {"r": 8}]
    _assert_main_unchanged(manager, sampler, backend)


@pytest.mark.parametrize("shared", [False, True])
def test_warm_before_the_first_sync_still_seeds_the_base(monkeypatch, shared):
    manager, sub, events = _manager(monkeypatch, shared=shared)
    manager.base_sync_done = False

    manager.warm(sub)

    sends = [kwargs for name, kwargs in events if name == ("send" if shared else "transfer")]
    assert len(sends) == 1
    assert sends[0]["base_sync_done"] is False
    assert manager.base_sync_done is False


def test_warm_publishes_learned_lora_weights(monkeypatch):
    import torch
    from peft import LoraConfig, get_peft_model

    from twinkle.model.transformers.transformers import TransformersModel

    base = torch.nn.Sequential()
    base.add_module("linear", torch.nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        base.linear.weight.zero_()
    model = get_peft_model(base, LoraConfig(target_modules=["linear"], r=1, lora_alpha=1))
    with torch.no_grad():
        model.base_model.model.linear.lora_A["default"].weight.fill_(1)
        model.base_model.model.linear.lora_B["default"].weight.fill_(2)
    wrapper = NS(
        model=model,
        strategy=NS(unwrap_model=lambda module: module),
        hf_config=NS(tie_word_embeddings=False),
        _get_default_group=lambda: "default",
    )
    manager, sub, _ = _manager(monkeypatch)
    published = {}

    def send(*args, **kwargs):
        weights = TransformersModel._get_weight_generator(
            wrapper, base_sync_done=kwargs["base_sync_done"], merge_and_sync=kwargs["merge_and_sync"]
        )
        published.update((name, tensor.detach().clone()) for name, tensor in weights)

    manager.model.send_weights_to_vllm = send

    manager.warm(sub)

    torch.testing.assert_close(published["linear.weight"], torch.full((2, 2), 2.0))
    # Publishing the merged adapter must leave the trainer's unmerged base intact.
    torch.testing.assert_close(model.base_model.model.linear.base_layer.weight, torch.zeros(2, 2))


def test_warm_failure_closes_only_the_temporary_ipc_channel(monkeypatch):
    manager, sub, events = _manager(monkeypatch, fail=True)
    sampler, backend = manager.sampler, manager.backend

    with pytest.raises(RuntimeError, match="sender failed"):
        manager.warm(sub)

    resets = [kwargs for name, kwargs in events if name.startswith("reset_")]
    assert len(resets) == 2
    assert all(kwargs["channel"].startswith("main-warm-") and kwargs["backend"] == "ipc" for kwargs in resets)
    _assert_main_unchanged(manager, sampler, backend)
