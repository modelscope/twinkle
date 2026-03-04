import torch
from PIL import Image
from typing import Any, Dict, List, Optional, Union
from twinkle import remote_class, requires
from twinkle.template import Template
from twinkle.template.base import ImageInput, VideoInput
from twinkle.template.utils import get_inputs_embeds_hf
from twinkle.utils import load_image


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
            requires('qwen_vl_utils')
            from qwen_vl_utils.vision_process import fetch_image
            image = load_image(image)
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
                result = fetch_video(video_input, image_patch_size=self.patch_size, return_video_sample_fps=False)
                return result
            elif isinstance(video, list):
                # List of images - preprocess each frame
                return [self.preprocess_image(frame) for frame in video]
            else:
                return super().preprocess_video(video)

        except ImportError:
            return super().preprocess_video(video)

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        input_ids = inputs['input_ids']
        from peft import PeftModel
        if isinstance(model, PeftModel):
            base_model = model.model
        else:
            base_model = model
        if hasattr(base_model.model, 'embed_tokens'):
            inputs_embeds = base_model.model.embed_tokens(input_ids)
        else:
            inputs_embeds = base_model.model.language_model.embed_tokens(input_ids)
        inputs_embeds = get_inputs_embeds_hf(inputs_embeds, inputs, model.visual, self.processor, model.config)
        return {'inputs_embeds': inputs_embeds}

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
            position_ids, _ = get_rope_index(input_ids, inputs.get('image_grid_thw'), inputs.get('video_grid_thw'),
                                             inputs.get('attention_mask'))
            return position_ids
        except Exception:
            return None
