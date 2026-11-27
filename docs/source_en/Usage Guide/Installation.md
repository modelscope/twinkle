# Twinkle Installation

| What you want to do | Install |
| --- | --- |
| Train or run inference on your own GPU | `pip install twinkle-kit` |
| Use the Python SDK against a remote service | see the source installation below |
| Run the HTTP service yourself | see the source installation below |

## Installation from Source

```shell
git clone https://github.com/modelscope/twinkle.git
cd twinkle
pip install -e .                                                        # local training and inference
pip install -e . -e ./twinkle_cs/client                                 # Python SDK, provides twinkle-auto
pip install -e . -e ./twinkle_cs/server                                 # HTTP service, provides twinkle-server
pip install -e '.[test]' -e ./twinkle_cs/client -e ./twinkle_cs/server  # work on all three
```

Install a matching `ms-swift` when you use Swift templates or model loaders.



## Docker Image

You can also use our pre-built Docker image:

```text
modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:twinkle-0.3.0
```

## Client Installation

If you need to use Twinkle's Client for remote training, you can use our one-click installation script:

```shell
# Mac or Linux
sh INSTALL_CLIENT.sh
# Windows, Open with PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\INSTALL_CLIENT.ps1
```

This script will download or utilize conda to create a virtual environment called `twinkle-client`, which can be directly used for remote training.

## Megatron Dependencies

If you need to install Megatron-related dependencies, you can use the following script:

```shell
sh INSTALL_MEGATRON.sh
```

## Supported Hardware

| Hardware Environment            | Notes                                    |
|-----------------------------|----------------------------------------|
| GPU A10/A100/H100/RTX series |                                        |
| GPU T4/V100                 | Does not support bfloat16, Flash-Attention |
| Ascend NPU                  | Some operators not supported            |
| Kunlunxin XPU (P800)        | Supported via the cuda-alike route; some operators fp16-only, see [XPU Support](XPU-Support.md) |
| PPU                         | Supported                              |
| CPU                         | Supports partial components like dataset, dataloader |
