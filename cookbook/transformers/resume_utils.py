from pathlib import Path
from typing import Any, Optional
from twinkle import get_logger


logger = get_logger()


def resume_from_checkpoint(
        model: Any,
        dataloader: Any,
        checkpoint_path: Path,
        *,
        resume_only_model: bool,
        ignore_data_skip: bool,
        adapter_name: Optional[str] = None) -> int:
    checkpoint_dir = str(checkpoint_path)
    adapter_name = adapter_name or ''
    model_kwargs = {}
    if adapter_name != '':
        # Load adapter checkpoint.
        model_kwargs['adapter_name'] = adapter_name
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
        dataloader.skip_consumed_samples(progress['consumed_train_samples'])
        model.optimizer_group[adapter_name].cur_step = progress['cur_step']
        model.optimizer_group[adapter_name].gradient_accumulation_steps = progress['gradient_accumulation_steps']

        consumed_train_samples = int(progress['consumed_train_samples'])
        logger.info(f'Skipped {consumed_train_samples} consumed samples.')
        return consumed_train_samples

    # Load full training state, including model weights, optimizer states, and training progress.
    trainer_state = model.load_training_state(checkpoint_dir, **model_kwargs)
    dataloader.skip_consumed_samples(trainer_state['consumed_train_samples'])
    consumed_train_samples = int(trainer_state['consumed_train_samples'])
    logger.info(f'Restored full training state from step {trainer_state["cur_step"]}.')
    return consumed_train_samples
