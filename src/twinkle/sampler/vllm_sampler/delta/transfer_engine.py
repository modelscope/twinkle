# Copyright (c) ModelScope Contributors. All rights reserved.
"""vLLM delta weight transfer based on verl."""

from __future__ import annotations

import inspect
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import vllm

try:
    from vllm.config.weight_transfer import WeightTransferConfig
    from vllm.distributed.weight_transfer.base import (
        WeightTransferEngine,
        WeightTransferInitInfo,
        WeightTransferUpdateInfo,
    )
except ImportError as exc:
    raise RuntimeError(
        "vLLM delta_sharded support requires the vllm.config.weight_transfer "
        "and vllm.distributed.weight_transfer modules"
    ) from exc

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.checkpoint_weight_patch import (
        CheckpointWeightPatch,
    )

TWINKLE_DELTA_WEIGHT_TRANSFER_BACKEND = "twinkle_delta_ipc"
TWINKLE_DELTA_HCCL_WEIGHT_TRANSFER_BACKEND = "twinkle_delta_hccl"
_REGISTERED = False
_HCCL_REGISTERED = False


def _checkpoint_patch_api():
    try:
        from vllm.model_executor.model_loader.checkpoint_weight_patch import (
            CheckpointWeightPatch,
            load_checkpoint_weight_patches,
        )
    except ImportError as exc:
        if exc.name != "vllm.model_executor.model_loader.checkpoint_weight_patch":
            raise
        raise RuntimeError(
            "sync_backend='delta_sharded' with vLLM requires the checkpoint patch API from "
            f"vLLM #50723; installed vLLM {getattr(vllm, '__version__', 'unknown')} does not provide it. "
            "Install a compatible vLLM build or select a different checkpoint engine backend."
        ) from exc
    return CheckpointWeightPatch, load_checkpoint_weight_patches


def require_vllm_delta_support() -> None:
    """Verify that vLLM provides the required weight transfer and checkpoint patch APIs."""

    init_parameters = inspect.signature(WeightTransferEngine.__init__).parameters
    if not {"config", "vllm_config", "device", "model"}.issubset(init_parameters):
        raise RuntimeError(
            "sync_backend='delta_sharded' with vLLM requires the engine-owned WeightTransferEngine "
            "API from vLLM #44353 and the checkpoint patch API from vLLM #50723; installed vLLM "
            f"{getattr(vllm, '__version__', 'unknown')} does not provide the required WeightTransferEngine API. "
            "Install a compatible vLLM build or select a different checkpoint engine backend."
        )
    _checkpoint_patch_api()


def is_moe_model(hf_config) -> bool:
    """Whether vLLM will treat this HF config as a MoE model.

    Delta updates require the Triton MoE backend, so this check must agree with
    the engine's own MoE detection: expert counts may live under a different
    attribute (Dbrx ``moe_num_experts``, DeepSeek ``n_routed_experts``, Mixtral
    ``num_local_experts``) or on a nested text config (multimodal MoE). The
    fallback mirrors vLLM's attribute list for configs its convertor cannot
    process.
    """
    try:
        from vllm.transformers_utils.config import get_hf_text_config
        from vllm.transformers_utils.model_arch_config_convertor import (
            MODEL_ARCH_CONFIG_CONVERTORS,
            ModelArchConfigConvertorBase,
        )

        convertor_cls = MODEL_ARCH_CONFIG_CONVERTORS.get(
            getattr(hf_config, "model_type", ""), ModelArchConfigConvertorBase
        )
        return int(convertor_cls(hf_config, get_hf_text_config(hf_config)).get_num_experts()) > 0
    except Exception:
        text_config = hf_config.get_text_config() if hasattr(hf_config, "get_text_config") else hf_config
        for name in ("num_experts", "moe_num_experts", "n_routed_experts", "num_local_experts"):
            num_experts = getattr(text_config, name, 0)
            if isinstance(num_experts, list):
                num_experts = num_experts[0] if num_experts else 0
            if num_experts and int(num_experts) > 0:
                return True
        return False


@dataclass
class TwinkleDeltaIPCInitInfo(WeightTransferInitInfo):
    """No initialization data is needed for the local IPC transport."""


@dataclass
class TwinkleDeltaIPCUpdateInfo(WeightTransferUpdateInfo):
    """Connection details for one flush."""

    zmq_handle: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.zmq_handle, str) or not self.zmq_handle:
            raise ValueError("Twinkle delta IPC updates require a worker-local zmq_handle")


def decode_delta_payload(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
) -> tuple[str, list[CheckpointWeightPatch]]:
    """Validate and decode one DeltaFlush into vLLM checkpoint patches."""

    from twinkle.checkpoint_engine.delta.protocol.decode import decode_delta_wire

    CheckpointWeightPatch, _ = _checkpoint_patch_api()
    encoding, patches = decode_delta_wire(named_tensors)
    return encoding, [
        CheckpointWeightPatch(
            name=p.name,
            shape=p.shape,
            dtype=p.dtype,
            values=p.values,
            indices=p.positions,
        )
        for p in patches
    ]


class TwinkleDeltaIPCWeightTransferEngine(WeightTransferEngine):
    """Consume Twinkle DeltaFlush payloads over the colocated CUDA-IPC channel."""

    init_info_cls = TwinkleDeltaIPCInitInfo
    update_info_cls = TwinkleDeltaIPCUpdateInfo
    supports_draft_weight_update = False
    required_device_type = "cuda"

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config,
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        self._session_encoding: str | None = None
        self._update_failed = False

    def _device_error_message(self) -> str:
        return "Twinkle delta weight transfer requires CUDA"

    def _validate_configuration(self) -> None:
        require_vllm_delta_support()
        if self.device.type != self.required_device_type:
            raise NotImplementedError(self._device_error_message())
        if self.parallel_config.data_parallel_size != 1:
            raise NotImplementedError("Twinkle delta weight transfer requires data_parallel_size=1")
        if self.parallel_config.pipeline_parallel_size != 1:
            raise NotImplementedError("Twinkle delta weight transfer requires pipeline_parallel_size=1")
        if self.model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Twinkle delta weight transfer supports only BF16 vLLM models")
        if getattr(self.vllm_config, "quant_config", None) is not None:
            raise NotImplementedError("Twinkle delta weight transfer does not support quantized vLLM models")
        if getattr(self.vllm_config, "speculative_config", None) is not None:
            raise NotImplementedError("Twinkle delta weight transfer does not support speculative decoding")

    def init_transfer_engine(self, init_info: TwinkleDeltaIPCInitInfo) -> None:
        self._validate_configuration()

    def start_weight_update(self) -> None:
        if self._update_failed:
            raise RuntimeError(
                "A previous delta update did not complete; restart the full job so every rollout worker receives a "
                "fresh dense seed. Restarting only the vLLM workers is unsafe because the producer snapshot may "
                "already have advanced."
            )
        self._session_encoding = None

    def _receive_payload(self, *, zmq_handle: str) -> list[tuple[str, torch.Tensor]]:
        from twinkle.sampler.vllm_sampler.delta.cuda_ipc import (
            BucketedWeightReceiver,
        )

        receiver = BucketedWeightReceiver(
            zmq_handle=zmq_handle,
            device=self.device,
            use_shm=False,
        )
        payload: list[tuple[str, torch.Tensor]] = []
        callback_error: BaseException | None = None

        def retain_bucket(weights: list[tuple[str, torch.Tensor]], _is_last: bool) -> None:
            nonlocal callback_error
            if callback_error is not None:
                return
            try:
                # The receiver reuses its IPC buffer for the next bucket, so
                # retain a private copy until this flush is decoded and applied.
                payload.extend((name, tensor.clone()) for name, tensor in weights)
            except BaseException as exc:
                # Finish the receiver handshake before reporting a callback error.
                callback_error = exc

        receiver.receive_weights(on_bucket_received=retain_bucket)
        if callback_error is not None:
            raise callback_error
        return payload

    def _decode_and_apply(self, payload: list[tuple[str, torch.Tensor]]) -> None:
        """Decode one flush's named tensors into checkpoint patches and apply them."""
        from twinkle.sampler.vllm_sampler.delta.weight_update_utils import (
            drop_tied_alias_patches,
            prepare_sparse_weight_loaders,
        )

        encoding, patches = decode_delta_payload(payload)
        patches = drop_tied_alias_patches(self.model, patches)

        if self._session_encoding is None:
            self._session_encoding = encoding
            if encoding == "dense":
                from vllm.model_executor.model_loader.reload import (
                    initialize_layerwise_reload,
                )

                initialize_layerwise_reload(self.model)
        elif encoding != self._session_encoding:
            raise ValueError(
                "One weight update cannot mix dense and sparse DeltaFlush payloads "
                f"({self._session_encoding!r} then {encoding!r})"
            )

        if encoding == "indices":
            prepare_sparse_weight_loaders(self.model)
        _, load_checkpoint_weight_patches = _checkpoint_patch_api()
        # The trusted producer emits every changed position at most once;
        # skip vLLM's sort-based duplicate check on the rollout device.
        load_checkpoint_weight_patches(
            self.model,
            patches,
            validate_unique_indices=False,
        )

    def receive_weights(self, update_info: TwinkleDeltaIPCUpdateInfo) -> None:
        assert update_info.zmq_handle is not None
        try:
            payload = self._receive_payload(zmq_handle=update_info.zmq_handle)
            self._decode_and_apply(payload)
        except BaseException:
            # An earlier flush may already have changed runtime weights, so a
            # later update would start from an unknown partial state.
            self._update_failed = True
            raise

    def finish_weight_update(self) -> None:
        if self._session_encoding == "dense":
            # Dense updates use vLLM's layerwise reload lifecycle. Finalize it
            # once after the last flush to run model-specific post-load work.
            from vllm.model_executor.model_loader.reload import (
                finalize_layerwise_reload,
            )

            try:
                finalize_layerwise_reload(self.model, self.model_config)
            except BaseException:
                self._update_failed = True
                raise
        elif self._session_encoding == "indices":
            # Sparse patches only write source parameters; re-derive runtime copies that
            # process_weights_after_loading would have rebuilt (e.g. Ascend packed conv1d).
            from twinkle.sampler.vllm_sampler.delta.weight_update_utils import (
                refresh_derived_weights,
            )

            try:
                refresh_derived_weights(self.model)
            except BaseException:
                self._update_failed = True
                raise
        self._session_encoding = None

    def shutdown(self) -> None:
        self._session_encoding = None

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[Any],
        trainer_args: dict[str, Any] | Any,
    ) -> None:
        raise NotImplementedError("Twinkle sends DeltaFlush payloads; this adapter only receives them")


def register_twinkle_delta_weight_transfer_engine() -> None:
    """Register the adapter before vLLM creates its worker-side engine."""

    global _REGISTERED
    if _REGISTERED:
        return
    from vllm.distributed.weight_transfer import WeightTransferEngineFactory

    WeightTransferEngineFactory.register_engine(
        TWINKLE_DELTA_WEIGHT_TRANSFER_BACKEND,
        TwinkleDeltaIPCWeightTransferEngine,
    )
    _REGISTERED = True


@dataclass
class TwinkleDeltaHCCLInitInfo(WeightTransferInitInfo):
    """Rendezvous endpoints of the trainer's HCCL delta broadcast group."""

    dist_ip: str
    dist_port: int
    zmq_ip: str
    zmq_port: int
    topic: str = "bucket_metadata"
    world_size: int = 1
    # Trainer rank 0's HCCL socket env (HCCL_IF_BASE_PORT etc.); applied before joining the group.
    hccl_env: dict[str, str] = field(default_factory=dict)


@dataclass
class TwinkleDeltaHCCLUpdateInfo(WeightTransferUpdateInfo):
    """This worker's rank in the trainer's HCCL delta broadcast group."""

    group_rank: int


class TwinkleDeltaHCCLWeightTransferEngine(TwinkleDeltaIPCWeightTransferEngine):
    """Consume Twinkle DeltaFlush payloads straight from the trainer's HCCL broadcasts (Ascend NPU).

    The vLLM worker joins the trainer's broadcast group itself: one manifest subscription plus
    one or two broadcasts per flush. The SUB is connected before the communicator is built, and
    both are created once and reused across sessions.
    """

    init_info_cls = TwinkleDeltaHCCLInitInfo
    update_info_cls = TwinkleDeltaHCCLUpdateInfo
    required_device_type = "npu"

    def __init__(self, config, vllm_config, device, model) -> None:
        super().__init__(config, vllm_config, device, model)
        self._rendezvous: TwinkleDeltaHCCLInitInfo | None = None
        self._manifest_sub = None
        self._comm = None
        self._group_rank: int | None = None

    def _device_error_message(self) -> str:
        return "Twinkle delta HCCL weight transfer requires Ascend NPU ('npu' device)"

    def init_transfer_engine(self, init_info: TwinkleDeltaHCCLInitInfo) -> None:
        self._validate_configuration()
        # A re-seed always builds a new trainer group: drop any channel from an earlier rendezvous
        # (even an identical one) so _ensure_channel rebuilds against the new endpoint.
        self.shutdown()
        self._rendezvous = init_info

    def _ensure_channel(self, group_rank: int):
        """Create (once) and return the (manifest SUB, communicator) pair, SUB first."""
        from twinkle.checkpoint_engine.delta.rollout.hccl_receiver import build_delta_hccl_communicator
        from twinkle.checkpoint_engine.zmq_pubsub import connect_subscriber

        if self._comm is None:
            if self._rendezvous is None:
                raise RuntimeError("init_transfer_engine must run before receive_weights")
            rv = self._rendezvous
            self._manifest_sub = connect_subscriber(rv.zmq_ip, rv.zmq_port, rv.topic)
            try:
                self._comm = build_delta_hccl_communicator(
                    rv.dist_ip, rv.dist_port, group_rank, rv.world_size, self.device, hccl_env=rv.hccl_env
                )
            except BaseException:
                self._close_sub()
                raise
            self._group_rank = group_rank
        elif group_rank != self._group_rank:
            raise RuntimeError(
                f"HCCL delta group_rank changed between sessions ({self._group_rank} -> {group_rank})"
            )
        return self._manifest_sub, self._comm

    def receive_weights(self, update_info: TwinkleDeltaHCCLUpdateInfo) -> None:
        from twinkle.checkpoint_engine.delta.rollout.hccl_receiver import pump_hccl_delta_session

        try:
            sub, comm = self._ensure_channel(update_info.group_rank)
            pump_hccl_delta_session(sub, comm, self.device, on_flush=self._decode_and_apply)
        except BaseException:
            self._update_failed = True
            raise

    def _close_sub(self) -> None:
        sub, self._manifest_sub = self._manifest_sub, None
        if sub is None:
            return
        context = getattr(sub, "context", None)
        try:
            sub.close(linger=0)
        except Exception:
            pass
        if context is not None:
            try:
                context.term()
            except Exception:
                pass

    def shutdown(self) -> None:
        comm, self._comm = self._comm, None
        if comm is not None:
            try:
                comm.destroyComm(comm.comm)
            except Exception:
                pass
        self._close_sub()
        self._group_rank = None
        self._rendezvous = None
        super().shutdown()


def register_twinkle_delta_hccl_weight_transfer_engine() -> None:
    """Register the HCCL adapter before vLLM creates its worker-side engine."""

    global _HCCL_REGISTERED
    if _HCCL_REGISTERED:
        return
    from vllm.distributed.weight_transfer import WeightTransferEngineFactory

    WeightTransferEngineFactory.register_engine(
        TWINKLE_DELTA_HCCL_WEIGHT_TRANSFER_BACKEND,
        TwinkleDeltaHCCLWeightTransferEngine,
    )
    _HCCL_REGISTERED = True
