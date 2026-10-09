# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Server-specific Pydantic models for checkpoint path resolution.
"""
from pydantic import BaseModel
from typing import Optional


class ResolvedLoadPath(BaseModel):
    """Result of resolving a load path.

    Attributes:
        checkpoint_name: The name of the checkpoint (e.g., 'step-8' or hub model id)
        checkpoint_dir: The directory containing the checkpoint, or None if loading from hub
        is_twinkle_path: Whether the path was a twinkle:// path
        training_run_id: The training run ID (only set for twinkle:// paths)
        checkpoint_id: The checkpoint ID (only set for twinkle:// paths)
    """
    checkpoint_name: str
    checkpoint_dir: Optional[str] = None
    is_twinkle_path: bool = False
    training_run_id: Optional[str] = None
    checkpoint_id: Optional[str] = None
