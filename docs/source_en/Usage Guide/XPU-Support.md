# XPU (Kunlunxin) Quick Start Guide

This document describes how to install and use the Twinkle framework on Kunlunxin XPU (P800 series).

## How XPU Support Works

Unlike Ascend NPU, Kunlunxin XPU does **not** introduce a separate device type in Twinkle. It relies on `torch_xmlir`, which rewrites `torch.cuda` symbols at import time so that:

- `torch.cuda.is_available()` returns `True` and devices report as `cuda:N`
- `libbkcl.so` (BKCL) is mounted as PyTorch's `nccl` backend (XCCL)
- `CUDA_VISIBLE_DEVICES` is taken over by the XPU runtime

Because of this "cuda-alike" route, most of Twinkle's GPU code path works unchanged. Only a few CUDA-specific assumptions that the symbol rewrite does not cover need adaptation. The `XPU` platform therefore **inherits from `GPU`** (`src/twinkle/utils/platforms/xpu.py`) rather than being a fully independent device backend like NPU.

## Environment Requirements

Adaptation was validated inside the vendor container image (`kylin_v11-vllm_021_swift:*`). The verified component matrix:

| Component | Version | Notes |
|---|---|---|
| OS / Python | Kylin V11 / 3.10.14 | Meets Twinkle `>=3.10` |
| torch | 2.9.0 | Symbol-rewritten by `torch_xmlir` |
| vllm / vllm_kunlun | 0.21.0 / 0.21.0.dev0 (also revalidated on 0.25.1) | `KunlunPlatform: device_type="cuda", dist_backend="nccl"` |
| megatron-core / transformers / triton | 0.16.1 / 5.9.0 / 3.5.0 | triton runs on the xmlir backend |
| XPU kernels | xpu_flash_attn, xpu_fla, kunlun_ops, xspeedgate_ops, xformers | Provided by the vendor container |

**Notes**:
- The Kunlunxin driver, `torch_xmlir`, and the XPU-enabled `torch` / `vllm_kunlun` builds are provided by Kunlunxin (typically via the vendor container image). Twinkle does not install them.
- `vllm_kunlun` and its dependencies come from Kunlunxin; some kernels currently have dtype restrictions (see [Known Limitations](#known-limitations)).

## Supported Hardware

- Kunlunxin P800 (OAM), verified on an 8-card host

## Installation Steps

### 1. Prepare the XPU environment

Use the Kunlunxin-provided container image (or a host with the Kunlunxin driver, `torch_xmlir`, and the XPU builds of `torch` / `vllm` / `vllm_kunlun` installed). Twinkle does not manage these components.

### 2. Install Twinkle

Install from source with `--no-deps` so pip does not attempt to replace the vendor's `torch` / `vllm` builds:

```bash
git clone https://github.com/modelscope/twinkle.git
cd twinkle
pip install -e . --no-deps
pip install pyzmq
# If the container is missing debug/runtime helpers:
pip install h5py prettytable func_timeout ray redis
```

### 3. Verify Installation

Create test script `verify_xpu.py`:

```python
import torch
import twinkle  # triggers ensure_xpu_compat()
from twinkle.utils.platforms import Platform

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA (XPU) available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Detected platform: {Platform.get_platform().__name__}")

if torch.cuda.is_available():
    x = torch.randn(3, 3).cuda()
    y = torch.randn(3, 3).cuda()
    print(f"XPU computation test passed: {(x + y).shape}")
```

Run verification:

```bash
python verify_xpu.py
```

A successful run reports `Detected platform: XPU`, `CUDA (XPU) available: True`, and the correct device count. Platform detection keys off the presence of `xpu-smi` on `PATH`.

## Verified Capabilities

The following have been validated end-to-end on Kunlunxin P800 (single-host, 8-card):

| Capability | Backend | Status | Notes |
|---|---|---|---|
| FSDP2 LoRA SFT | transformers | ✅ Verified | Single-card, loss converges as on GPU |
| vLLM sampling (TP=1) | vLLM | ✅ Verified | Requires `enforce_eager=True`; text models |
| vLLM sampling (TP=2) | vLLM | ✅ Verified | Tensor parallel |
| GRPO (colocate) | native_fsdp | ✅ Verified | Full-weight sync (`merge_and_sync=True`); relay ~0.7 GB/s |
| Megatron LoRA (TP=2) | Megatron | ✅ Verified | Zero Twinkle code change; cuda-alike path |
| Weight sync (train↔sample) | XCCL | ✅ Verified | Via `XCCLCheckpointEngine`, single-host relay |

## Known Limitations

These are constraints of the current Kunlunxin XPU stack (vendor kernels / `vllm_kunlun`), not of Twinkle logic:

- **LoRA sampling is not usable**: `vLLMSampler` with `enable_lora=True` cannot start. The underlying LoRA kernels (`bgmv_shrink_cluster`, `sgmv_expand_sdnn`, `sgmv_expand_slice`) are fp16-only on the 0.21 stack. Use full-weight sync (`merge_and_sync=True`) for RL. (On the 0.25.1 stack the vendor added a bf16 adapter layer, but other LoRA-path issues remain.)
- **fp16 required for some models**: GDN FLA kernels are fp16-only; Qwen3.5 inference must run in float16.
- **CUDA graph disabled**: Use `enforce_eager=True` for vLLM; graph capture on XPU is not yet validated.
- **Multimodal (Qwen-VL) blocked**: The ViT SDPA kernel triggers a hardware-level `kl3ChannelCheckErrors` / `noc idle timeout` that requires a physical card reset. Use text-only models.
- **Single-host only**: `XCCLCheckpointEngine` multi-host (inter/intra-node) broadcast is implemented but not yet validated; only single-host relay mode is verified.
- **`logprobs` on vllm_kunlun 0.25.1**: The logprobs path can return NaN/uninitialized top-logprob token ids, causing tokenizer `OverflowError`. Reported to the vendor.

## Platform Internals (for reference)

The XPU adaptation is concentrated in a few files:

- `src/twinkle/utils/platforms/xpu.py` — `XPU` platform (inherits `GPU`); `ensure_xpu_compat()` neutralizes the native Intel-XPU stub in `torch.xpu`; a device-UUID fallback chain (`current_platform` → `xpu-smi -q` Bus Id → `xpu-smi -L` UUID → sha1) for vLLM.
- `src/twinkle/utils/platforms/base.py` — detects XPU via `xpu-smi` on `PATH`.
- `src/twinkle/infra/_ray/ray_helper.py` — registers the cuda-alike GPU count through `ray.init(num_gpus=N)` (Ray cannot autodetect XPUs). Remote workers started via `ray start` must pass `--num-gpus`.
- `src/twinkle/checkpoint_engine/xpu_checkpoint_engine.py` — `XCCLCheckpointEngine` (see the Checkpoint Engine docs).

## Reference Resources

- [vLLM-Kunlun](https://github.com/baidu/vLLM-Kunlun)
- [Twinkle GitHub](https://github.com/modelscope/twinkle)
- [Twinkle Documentation](https://twinkle.readthedocs.io/)

## Getting Help

If you encounter issues during use:

1. **Submit an Issue**: [Twinkle GitHub Issues](https://github.com/modelscope/twinkle/issues)
2. **Vendor stack issues** (LoRA kernels, logprobs, multimodal): report to Kunlunxin / vLLM-Kunlun.
