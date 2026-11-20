'Single-card accumulation and update-boundary checkpoints. The caller owns epoch shuffling.'
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass
class Progress:
    accumulation_steps: int = 2
    micro_steps: int = 0
    optimizer_steps: int = 0
    consumed_samples: int = 0
    pending_micro_steps: int = 0

    def __post_init__(self):
        if self.accumulation_steps < 1:
            raise ValueError('accumulation_steps must be positive')


def windows(batches, size):
    if size < 1:
        raise ValueError('window size must be positive')
    window = []
    for batch in batches:
        window.append(batch)
        if len(window) == size:
            yield window
            window = []
    if window:
        yield window


def train_window(model, batches, progress, adapter_name, before_micro=None):
    """Exactly one optimizer/scheduler update for a full or final short window."""
    batches = list(batches)
    if not 1 <= len(batches) <= progress.accumulation_steps:
        raise ValueError('Expected one nonempty accumulation window')
    group = model.optimizer_group[adapter_name]
    if group.gradient_accumulation_steps != 1:
        raise ValueError('Twinkle accumulation must be 1; cookbook owns boundaries')
    if progress.pending_micro_steps or group.train_status.num_tokens:
        raise ValueError('Previous accumulation window is incomplete')
    if group.cur_step != progress.micro_steps:
        raise ValueError('Twinkle and cookbook micro-step counts differ')
    # Validate the entire window before executing any backward operation.
    for batch in batches:
        if batch['input_ids'].shape[0] != 1:
            raise ValueError('Only single-sample micro-batches are supported')
    losses = []
    for batch in batches:
        if before_micro is not None:
            before_micro(progress.micro_steps)
        result = model.forward_backward(inputs=batch, adapter_name=adapter_name)
        if not math.isfinite(float(result['loss'])):
            raise FloatingPointError('Nonfinite micro-batch loss; abort this run')
        losses.append(float(result['loss']))
        progress.micro_steps += 1
        progress.consumed_samples += 1
        progress.pending_micro_steps += 1
    # Each Qwen loss is a mean and reports num_tokens=0. Twinkle counts each
    # as 1, so clipping normalizes by the ACTUAL window size, including tails.
    if group.train_status.num_tokens != len(batches):
        raise AssertionError('Unexpected loss normalization count')
    model.clip_grad_and_step(adapter_name=adapter_name)
    progress.optimizer_steps += 1
    progress.pending_micro_steps = 0
    return losses


def save_boundary(model, progress, output_dir, name, adapter_name, sample_fingerprint):
    group = model.optimizer_group[adapter_name]
    if progress.pending_micro_steps or group.train_status.num_tokens:
        raise ValueError('Cannot save with unstepped gradients')
    if group.cur_step != progress.micro_steps:
        raise ValueError('Micro-step count mismatch')
    if any(p.grad is not None for p in model.model.parameters()):
        raise ValueError('Checkpoint requires cleared gradients')
    checkpoint = Path(model.save(name, output_dir=str(output_dir), adapter_name=adapter_name,
                                 save_optimizer=True, consumed_train_samples=progress.consumed_samples))
    (checkpoint / 'cookbook_progress.json').write_text(json.dumps({
        'format_version': 1, 'progress': asdict(progress),
        'sample_fingerprint': sample_fingerprint}, indent=2), encoding='utf-8')
    return checkpoint


def resume_boundary(model, checkpoint, adapter_name, sample_fingerprint, accumulation_steps):
    """Restore only complete locally produced checkpoints, never silently restart."""
    checkpoint = Path(checkpoint)
    for name in ('adapter_config.json', 'adapter_model.safetensors', 'optimizer.pt',
                 'scheduler.pt', 'rng_state.pt', 'trainer_state.json', 'cookbook_progress.json'):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    state = json.loads((checkpoint / 'cookbook_progress.json').read_text(encoding='utf-8'))
    if state['format_version'] != 1 or state['sample_fingerprint'] != sample_fingerprint:
        raise ValueError('Checkpoint format/sample mismatch')
    progress = Progress(**state['progress'])
    if progress.pending_micro_steps or progress.accumulation_steps != accumulation_steps:
        raise ValueError('Checkpoint boundary/accumulation mismatch')
    trainer = json.loads((checkpoint / 'trainer_state.json').read_text(encoding='utf-8'))
    if (trainer['cur_step'] != progress.micro_steps
            or trainer['consumed_train_samples'] != progress.consumed_samples
            or trainer['gradient_accumulation_steps'] != 1):
        raise ValueError('Twinkle checkpoint progress disagrees with cookbook')
    group = model.optimizer_group[adapter_name]
    if getattr(group, 'optimizer', None) is None or getattr(group, 'lr_scheduler', None) is None:
        raise ValueError('Create optimizer and scheduler before resuming')
    model.resume_from_checkpoint(str(checkpoint), adapter_name=adapter_name)
    group = model.optimizer_group[adapter_name]
    if group.cur_step != progress.micro_steps or group.gradient_accumulation_steps != 1:
        raise AssertionError('Twinkle did not restore micro-step state')
    return progress
