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
pip install -e . --no-build-isolation
```

## Supported Hardware

| Hardware Environment            | Notes                                    |
|-----------------------------|----------------------------------------|
| GPU A10/A100/H100/RTX series |                                        |
| GPU T4/V100                 | Does not support bfloat16, Flash-Attention |
| Ascend NPU                  | Some operators not supported            |
| PPU                         | Supported                              |
| CPU                         | Supports partial components like dataset, dataloader |
