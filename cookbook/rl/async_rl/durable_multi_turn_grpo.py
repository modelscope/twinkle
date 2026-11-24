# Copyright (c) ModelScope Contributors. All rights reserved.
"""Crash-safe GSM8K multi-turn GRPO using model component state.

The StreamingDriver persists only original prompt queries. The trainer writes that component
state in the same checkpoint as the adapter, optimizer, scheduler and RNG state:

    component = driver.state_dict()
    model.save(..., component_states={"streaming_driver": component})

On resume, the model checkpoint is loaded first and the saved queries are submitted as fresh
rollout attempts. No environment snapshot, turn journal or second commit protocol is involved.
"""

from __future__ import annotations
import argparse
import os
import re
import time
from pathlib import Path
from typing import Any

import json
from peft import LoraConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.metric import CompletionRewardMetric
from twinkle.model import TransformersModel
from twinkle.preprocessor.llm import GSM8KProcessor
from twinkle.processor import InputProcessor
from twinkle.reward import GSM8KAccuracyReward, GSM8KBrevityReward
from twinkle.reward.base import Reward
from twinkle.reward.worker import FunctionWorker
from twinkle.sampler import vLLMSampler
from twinkle.template import Qwen3_5Template
from twinkle.utils import Platform
from twinkle_agentic.async_rl import StreamingDriver
from twinkle_agentic.rollout.multi_turn import MultiTurnRollout

logger = get_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="ms://Qwen/Qwen3.5-4B")
    parser.add_argument("--strategy", choices=("native_fsdp", "accelerate"), default="native_fsdp")
    parser.add_argument("--model-gpus", type=int, default=4)
    parser.add_argument("--sampler-gpus", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--adapter-name", default="durable-grpo")
    parser.add_argument("--resume-checkpoint", default="")
    return parser.parse_args()


def _followup(trajectory: dict[str, Any], followups: int) -> str | None:
    if followups:
        return None
    replies = [
        str(message.get("content") or "")
        for message in trajectory.get("messages") or ()
        if message.get("role") == "assistant"
    ]
    if replies and re.search(r"\\boxed\{[^}]+\}", replies[-1]):
        return None
    return "Please restate the result with the final answer inside \\boxed{}."


def _reward_future(reward: Reward):
    return FunctionWorker(reward)


def _score_vector(facts: dict[str, Any]) -> list[float]:
    rewards = []
    for item in facts["scores"]:
        values = (item.get("rewards") or {}).get("orm") or {}
        rewards.append(float(values.get("accuracy") or 0.0) + float(values.get("brevity") or 0.0))
    return rewards


def _latest_checkpoint(root: Path) -> str:
    checkpoints = []
    for path in root.glob("policy-step-*"):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((step, path))
    return str(max(checkpoints)[1]) if checkpoints else ""


def _write_history(path: Path, rows: dict[int, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(rows[step], ensure_ascii=False, default=str) + "\n" for step in sorted(rows)),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    model_gpus = args.model_gpus
    sampler_gpus = args.sampler_gpus
    total_gpus = model_gpus + sampler_gpus
    global_batch_size = args.batch_size * args.num_generations
    output_root = Path(os.environ.get("TWINKLE_DURABLE_OUTPUT", "output/durable-gsm8k"))
    checkpoint_root = output_root / "model-checkpoints"
    history_path = output_root / "training-metrics.jsonl"
    system_prompt = "You are a careful math assistant. Solve the problem and put only the final answer in \\boxed{}."
    accelerator = Platform.get_platform().__name__

    device_groups = [
        DeviceGroup(name="model", ranks=list(range(model_gpus)), device_type=accelerator),
        DeviceGroup(name="sampler", ranks=list(range(model_gpus, total_gpus)), device_type=accelerator),
        DeviceGroup(name="rollout", ranks=1, device_type="CPU"),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=model_gpus, dp_size=model_gpus)
    sampler_mesh = DeviceMesh.from_sizes(world_size=sampler_gpus, dp_size=sampler_gpus)
    twinkle.initialize(mode="ray", nproc_per_node=total_gpus, groups=device_groups, lazy_collect=False)

    model = TransformersModel(
        model_id=args.model_id,
        device_mesh=model_mesh,
        remote_group="model",
        mixed_precision="bf16",
    )
    model.add_adapter_to_model(
        args.adapter_name,
        LoraConfig(target_modules="all-linear", r=args.lora_r, lora_alpha=args.lora_r * 2),
        gradient_accumulation_steps=1,
    )
    model.set_optimizer("AdamW", lr=args.lr)
    model.set_lr_scheduler("CosineAnnealingLR", T_max=args.max_steps, eta_min=0)
    model.set_loss("GRPOLoss", epsilon=0.2)
    model.set_processor(InputProcessor, padding_free=True)
    model.set_template("Qwen3_5Template", model_id=args.model_id, enable_thinking=False)

    sampler = vLLMSampler(
        model_id=args.model_id,
        engine_args={
            "gpu_memory_utilization": 0.8,
            "max_model_len": 8192,
            "max_lora_rank": max(32, args.lora_r),
            "enable_lora": True,
            "enable_tower_connector_lora": True,
        },
        device_mesh=sampler_mesh,
        remote_group="sampler",
    )
    sampler.set_template("Qwen3_5Template", model_id=args.model_id, enable_thinking=False)
    checkpoint_manager = CheckpointEngineManager(model=model, sampler=sampler, merge_and_sync=False)

    checkpoint = args.resume_checkpoint or _latest_checkpoint(checkpoint_root)
    component_states: dict[str, Any] = {}
    optim_step = 0
    if checkpoint:
        progress = model.resume_from_checkpoint(checkpoint, adapter_name=args.adapter_name)
        component_states = dict(progress.get("component_states") or {})
        optim_step = int(progress.get("cur_step") or 0)
        checkpoint_manager.sync_weights()
        logger.info("resumed %s at optimizer step %s", checkpoint, optim_step)

    dataset = Dataset()
    dataset.add_dataset(DatasetMeta("ms://modelscope/gsm8k", subset_name="main", split="train"))
    dataset.set_template(
        "Qwen3_5Template",
        model_id=args.model_id,
        max_length=4096,
        truncation_strategy="delete",
        enable_thinking=False,
    )
    dataset.map(GSM8KProcessor(system=system_prompt))
    dataset.encode(add_generation_prompt=True)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        min_batch_size=args.batch_size,
        device_mesh=model_mesh,
        remote_group="model",
    )

    template = Qwen3_5Template(args.model_id, max_length=4096, enable_thinking=False)
    template.truncation_strategy = "delete"
    rollout = MultiTurnRollout(
        sampler=sampler,
        template=template,
        sampling_params=SamplingParams(
            max_tokens=args.max_tokens,
            num_samples=1,
            logprobs=1,
            temperature=1.0,
            top_p=0.95,
        ),
        max_turns=2,
        max_trajectory_tokens=4096,
        followup_fn=_followup,
        remote_group="rollout",
    )
    driver = StreamingDriver(
        dataloader,
        rollout,
        batch_size=global_batch_size,
        num_generations=args.num_generations,
        max_steps=args.max_steps,
        max_staleness=1,
        checkpoint_manager=checkpoint_manager,
        orms=[_reward_future(GSM8KAccuracyReward()), _reward_future(GSM8KBrevityReward())],
        orm_names=["accuracy", "brevity"],
    )
    driver.load_state_dict(component_states.get("streaming_driver"))

    advantage_fn = GRPOAdvantage(num_generations=args.num_generations)
    metrics = CompletionRewardMetric()
    history = {}
    if history_path.is_file():
        history = {
            int(row["optimizer_step"]): row
            for row in (json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line)
        }
    logger.info(get_device_placement())

    started = time.monotonic()
    try:
        for rows, facts in driver:
            if optim_step >= args.max_steps:
                break
            rewards = _score_vector(facts)
            rows, _ = advantage_fn(rows, scores=facts["scores"], global_step=optim_step)
            metrics.accumulate(completion_lengths=facts["response_lengths"], rewards={"total": rewards})
            model.forward_backward(inputs=rows, micro_batch_size=args.micro_batch_size)
            model.clip_grad_and_step()
            optim_step += 1

            component = driver.state_dict()
            checkpoint = model.save(
                f"policy-step-{optim_step}",
                output_dir=str(checkpoint_root),
                adapter_name=args.adapter_name,
                save_optimizer=True,
                async_save=False,
                component_states={"streaming_driver": component},
            )
            checkpoint_manager.sync_weights()
            sampler.reset_prefix_cache()

            log_dict = metrics.calculate()
            log_dict.update(model.calculate_metric(is_training=True))
            history[optim_step] = {
                "optimizer_step": optim_step,
                "checkpoint": str(checkpoint),
                "elapsed_s": time.monotonic() - started,
                "metrics": log_dict,
                "driver": driver.stats(),
            }
            _write_history(history_path, history)
            logger.info("durable step %s/%s: %s", optim_step, args.max_steps, log_dict)
            metrics.reset()
    finally:
        driver.close()

    logger.info("durable GRPO completed: optimizer_step=%s", optim_step)


if __name__ == "__main__":
    main()
