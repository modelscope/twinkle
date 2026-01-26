from typing import Any, Dict, Optional
import torch
from twinkle import remote_class
from twinkle.template import Template


@remote_class()
class Qwen3VLTemplate(Template):
    """
    Processor for Qwen VL series.

    Note: Qwen3-VL handles embedding merge internally in forward(),
    so post_encode just passes through.
    """

    # _build_messages: Uses base class implementation.
    # Qwen's HF processor accepts the standard format:
    # [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': '...'}]}]

    def post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Qwen3-VL handles embedding merge internally."""
        return inputs

    def _get_vision_token_id(self) -> Optional[int]:
        if self.config is not None:
            return getattr(self.config, 'image_token_id', None)
        return None

    def _get_position_ids(self, inputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Get 3D RoPE position_ids for Qwen VL."""
        if self.model is None:
            return None

        input_ids = inputs.get('input_ids')
        if input_ids is None:
            return None

        # Find get_rope_index
        base_model = self.model
        if hasattr(base_model, 'base_model'):
            base_model = base_model.base_model
        if hasattr(base_model, 'model'):
            base_model = base_model.model

        get_rope_index = getattr(base_model, 'get_rope_index', None)
        if get_rope_index is None and hasattr(base_model, 'model'):
            get_rope_index = getattr(base_model.model, 'get_rope_index', None)

        if get_rope_index is None:
            return None

        try:
            position_ids, _ = get_rope_index(
                input_ids,
                inputs.get('image_grid_thw'),
                inputs.get('video_grid_thw'),
                inputs.get('attention_mask')
            )
            return position_ids
        except Exception as e:
            return None