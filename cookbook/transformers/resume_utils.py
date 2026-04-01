from pathlib import Path
from typing import Any, Optional

from twinkle import get_logger


logger = get_logger()


def _build_model_kwargs(adapter_name: str) -> dict:
    if not adapter_name:
        return {}
    return {'adapter_name': adapter_name}


def resume_from_checkpoint(
        model: Any,
        dataloader: Any,
        checkpoint_path: Path,
        *,
        resume_only_model: bool,
        ignore_data_skip: bool,
        adapter_name: Optional[str] = None) -> int:
    adapter_name = adapter_name or ''
    checkpoint_dir = str(checkpoint_path)
    model_kwargs = _build_model_kwargs(adapter_name)
    if model_kwargs:
        # Load adapter checkpoint.
        model.load(
            name=checkpoint_path.name,
            output_dir=str(checkpoint_path.parent),
            **model_kwargs,
        )

    if resume_only_model:
        # Only load model weights, optionally skip data.
        if ignore_data_skip:
            logger.info('Resumed weights only and restarted progress from step 0.')
            return 0
        progress = model.read_training_progress(checkpoint_dir, **model_kwargs)
        # Skip consumed samples in dataloader and move optimizer to the right step.
        consumed_train_samples = int(progress['consumed_train_samples'])
        dataloader.skip_consumed_samples(consumed_train_samples)
        optimizer_group = model.optimizer_group[adapter_name]
        optimizer_group.cur_step = progress['cur_step']
        optimizer_group.gradient_accumulation_steps = progress['gradient_accumulation_steps']
        logger.info(f'Skipped {consumed_train_samples} consumed samples.')
        return consumed_train_samples

    # Load full training state, including model weights, optimizer states, and training progress.
    trainer_state = model.load_training_state(checkpoint_dir, **model_kwargs)
    consumed_train_samples = int(trainer_state['consumed_train_samples'])
    dataloader.skip_consumed_samples(consumed_train_samples)
    logger.info(f'Restored full training state from step {trainer_state["cur_step"]}.')
    return consumed_train_samples
