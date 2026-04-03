import inspect
import numpy as np
import torch
from copy import copy
from PIL import Image
from typing import Any, Dict, List, Optional, Union

from twinkle import remote_class, requires
from twinkle.data_format import InputFeature
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
        from transformers.models.qwen3_vl import Qwen3VLModel
        with torch.device('meta'):
            self.dummy_model = Qwen3VLModel(self.config)
            self.rope_index_func = self.get_rope_index()

    def get_rope_index(self):
        for _, sub_module in self.dummy_model.named_modules():
            if hasattr(sub_module, 'get_rope_index'):
                return sub_module.get_rope_index
        raise NotImplementedError(f'Module {self.dummy_model.__class__.__name__} has no get_rope_index method!')

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

    @staticmethod
    def to_tensor(_input):
        import torch
        for key in list(_input.keys()):
            value = _input[key]
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            elif isinstance(value, list) and isinstance(value[0], (int, float, np.number)):
                value = torch.tensor(value)
            _input[key] = value
        return _input

    def set_mm_position_ids(self, input_feature: InputFeature):
        kwargs = {}
        input_feature = copy(input_feature)
        input_feature = self.to_tensor(input_feature)
        attention_mask = input_feature.get('attention_mask').unsqueeze(0)
        input_ids = input_feature['input_ids'].unsqueeze(0)
        if 'mm_token_type_ids' in inspect.signature(self.rope_index_func).parameters:
            mm_token_type_ids = torch.zeros_like(input_ids)
            mm_token_type_ids[input_ids == self.processor.image_token_id] = 1
            mm_token_type_ids[input_ids == self.processor.video_token_id] = 2
            kwargs['mm_token_type_ids'] = mm_token_type_ids
        position_ids, _ = self.rope_index_func(
            input_ids,
            image_grid_thw=input_feature.get('image_grid_thw'),
            video_grid_thw=input_feature.get('video_grid_thw'),
            attention_mask=attention_mask,
            **kwargs)
        return self._concat_text_position_ids(position_ids)

    @staticmethod
    def _concat_text_position_ids(position_ids):
        seq_len = position_ids.shape[-1]
        text_position_ids = torch.arange(seq_len, device=position_ids.device).expand(1, *position_ids.shape[1:])
        return torch.concat([text_position_ids, position_ids], dim=0)
