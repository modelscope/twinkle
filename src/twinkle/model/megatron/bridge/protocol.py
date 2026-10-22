# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class BridgeBackend(Protocol):
    backend_name: str

    def build_model_config(self, hf_config: Any, parallel_kwargs: Dict[str, Any], strategy: Any, **kwargs) -> Any:
        """HF config -> backend-specific model config (opaque; stored as ``strategy.config``)."""
        ...

    def create_model(self, config: Any, model_dir: str, *, load_weights: bool, move_to_gpu) -> List[Any]:
        """Build the model chunks from ``config``; ``move_to_gpu`` is the strategy's placement
        callable so device policy stays in one place."""
        ...

    @property
    def is_multimodal(self) -> bool:
        """Whether the constructed model is multimodal (drives VL module discovery)."""
        ...
