"""Launch native-TQ async multi-LoRA GRPO from one YAML configuration."""

from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from twinkle_agentic.async_rl import AsyncMultiLoraGRPOPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='cookbook/rl/async_rl/async_multi_lora_grpo.yaml')
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    print(AsyncMultiLoraGRPOPipeline.from_config(config).run())


if __name__ == '__main__':
    main()
