# Twinkle Installation

## Wheel Package Installation

You can install using pip:

```shell
pip install 'twinkle-kit'
```

## Installation from Source

```shell
git clone https://github.com/modelscope/twinkle.git
cd twinkle
pip install -e .
```

## Docker Image

You can also use our pre-built Docker image:

```text
modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:twinkle-0.2.0
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
| PPU                         | Supported                              |
| CPU                         | Supports partial components like dataset, dataloader |
