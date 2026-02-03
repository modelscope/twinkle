from typing import Any, Dict, List, Optional, Union
from PIL import Image
import torch

from twinkle import remote_class
from twinkle.template import Template
from twinkle.template.base import ImageInput, VideoInput


@remote_class()
class Qwen3VLTemplate(Template):
    """
    Processor for Qwen VL series.

    Note: Qwen3-VL handles embedding merge internally in forward(),
    so post_encode just passes through inputs unchanged.
    """

    def __init__(self, *args, **kwargs):
        # TODO untested code
        super().__init__(*args, **kwargs)
        # Cache processor config for preprocessing
        self._patch_size: Optional[int] = None
        self._merge_size: Optional[int] = None
        self._init_vision_config()

    def _init_vision_config(self):
        """Initialize vision config from processor."""
        if hasattr(self.processor, 'image_processor'):
            ip = self.processor.image_processor
            self._patch_size = getattr(ip, 'patch_size', 16)
            self._merge_size = getattr(ip, 'merge_size', 2)

    @property
    def patch_size(self) -> int:
        """Vision transformer patch size."""
        return self._patch_size or 16

    @property
    def merge_size(self) -> int:
        """Spatial merge size for vision tokens."""
        return self._merge_size or 2

    def preprocess_image(self, image: ImageInput) -> Image.Image:
        try:
            from qwen_vl_utils.vision_process import fetch_image
            if isinstance(image, str):
                image_input = {'image': image}
            elif isinstance(image, Image.Image):
                image_input = {'image': image}
            else:
                # Fallback to base class for tensor inputs
                return super().preprocess_image(image)
            
            # Use qwen_vl_utils with correct patch_size
            return fetch_image(image_input, image_patch_size=self.patch_size)
            
        except ImportError:
            return super().preprocess_image(image)

    def preprocess_video(self, video: VideoInput) -> Union[List[Image.Image], torch.Tensor]:
        try:
            from qwen_vl_utils.vision_process import fetch_video
            
            if isinstance(video, str):
                # Use qwen_vl_utils for video loading
                video_input = {'video': video}
                result = fetch_video(
                    video_input,
                    image_patch_size=self.patch_size,
                    return_video_sample_fps=False
                )
                return result
            elif isinstance(video, list):
                # List of images - preprocess each frame
                return [self.preprocess_image(frame) for frame in video]
            else:
                return super().preprocess_video(video)
                
        except ImportError:
            return super().preprocess_video(video)

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
        except Exception:
            return None
