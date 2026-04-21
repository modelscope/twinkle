from pathlib import Path
from typing import Any, Optional

from twinkle import get_logger

logger = get_logger()


def resume_from_checkpoint(model: Any,
                           dataloader: Any,
                           checkpoint_path: Path,
                           *,
                           resume_only_model: bool,
                           ignore_data_skip: bool,
                           adapter_name: Optional[str] = None) -> int:
    kwargs = {}
    if adapter_name:
        kwargs['adapter_name'] = adapter_name

    progress = model.resume_from_checkpoint(
        str(checkpoint_path), resume_only_model=resume_only_model, **kwargs)

    consumed_train_samples = int(progress.get('consumed_train_samples', 0))
    if not ignore_data_skip and consumed_train_samples > 0:
        dataloader.resume_from_checkpoint(consumed_train_samples)

    return consumed_train_samples
