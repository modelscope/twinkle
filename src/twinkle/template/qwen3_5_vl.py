import torch
from PIL import Image
from typing import Any, Dict, List, Optional, Union

from twinkle import remote_class, requires
from twinkle.template import Template
from twinkle.template.base import ImageInput, VideoInput
from twinkle.template.utils import get_inputs_embeds_hf


@remote_class()
class Qwen3_5Template(Template):
    """
    Processor for Qwen VL series.

    Note: Qwen3-VL handles embedding merge internally in forward(),
    so post_encode just passes through inputs unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        requires('qwen_vl_utils')
        from qwen_vl_utils.vision_process import fetch_image
        image = super().preprocess_image(image)
        if isinstance(image, str):
            image_input = {'image': image}
        elif isinstance(image, Image.Image):
            image_input = {'image': image}
        else:
            # Fallback to base class for tensor inputs
            return super().preprocess_image(image)

        # Use qwen_vl_utils with correct patch_size
        return fetch_image(image_input, image_patch_size=self.patch_size)

    def preprocess_video(self, video: VideoInput) -> Union[List[Image.Image], torch.Tensor]:
        requires('qwen_vl_utils')
        from qwen_vl_utils.vision_process import fetch_video

        if isinstance(video, str):
            video_input = {'video': video}
            result = fetch_video(video_input, image_patch_size=self.patch_size, return_video_sample_fps=False)
            return result
        elif isinstance(video, list):
            return [self.preprocess_image(frame) for frame in video]
        else:
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
        inputs_embeds = get_inputs_embeds_hf(inputs_embeds, inputs, base_model.model.visual, self.processor,
                                             model.config)
        return {'inputs_embeds': inputs_embeds}
