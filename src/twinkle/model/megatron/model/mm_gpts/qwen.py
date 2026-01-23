# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from PIL import Image

from ..constant import ModelType, MegatronModelType
from ..gpt_bridge import MultimodalGPTBridge
from ..register import MegatronModelMeta, register_megatron_model
from .utils import HuggingFaceModule
from twinkle.utils.torch_utils import to_device


class Qwen2_5VL_Vit(HuggingFaceModule):
    module_mapping = {'model.visual': 'visual'}
    _vision_tower = ['visual']
    _aligner = ['visual.merger']
    version = 'v2_5'

    def __init__(self, config):
        if self.version == 'v2_5':
            try:
                from transformers.models.qwen2_5_vl import Qwen2_5_VLTextModel
            except ImportError:
                from transformers.models.qwen2_5_vl import Qwen2_5_VLModel as Qwen2_5_VLTextModel
            ignore_init_model_cls = Qwen2_5_VLTextModel
        elif self.version == 'v2':
            try:
                from transformers.models.qwen2_vl import Qwen2VLTextModel
            except ImportError:
                from transformers.models.qwen2_vl import Qwen2VLModel as Qwen2VLTextModel
            ignore_init_model_cls = Qwen2VLTextModel
        super().__init__(config, ignore_init_model_cls)

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        return self._get_inputs_embeds_hf(inputs_embeds, kwargs, self.visual, self.processor, self.model_config)

    def _get_inputs_embeds_hf(self, inputs_embeds, inputs, visual, processor, config):
        # mimic the behavior of Template._get_inputs_embeds_hf in swift
        input_ids = inputs['input_ids']
        pixel_values = inputs.get('pixel_values')
        pixel_values_videos = inputs.get('pixel_values_videos')
        image_grid_thw = inputs.get('image_grid_thw')
        video_grid_thw = inputs.get('video_grid_thw')
        dtype = visual.dtype
        if pixel_values is None and pixel_values_videos is None:  # plain-text
            images = [Image.new('RGB', (32, 32), (0, 0, 0))]
            media_inputs = processor.image_processor(images=images, return_tensors='pt')
            media_inputs = to_device(media_inputs, input_ids.device)
            pixel_values = media_inputs['pixel_values'].type(dtype)
            image_embeds = visual(pixel_values, grid_thw=media_inputs['image_grid_thw'])
            inputs_embeds = inputs_embeds + image_embeds.mean().to(device=inputs_embeds.device) * 0.
        else:
            if pixel_values is None:
                pixel_values_mixed = pixel_values_videos
                grid_thw = video_grid_thw
            elif pixel_values_videos is None:
                pixel_values_mixed = pixel_values
                grid_thw = image_grid_thw
            else:
                pixel_values_mixed = torch.concat([pixel_values, pixel_values_videos], dim=0)
                grid_thw = torch.concat([image_grid_thw, video_grid_thw], dim=0)
            pixel_values_mixed = pixel_values_mixed.type(dtype)
            mixed_embeds = visual(pixel_values_mixed, grid_thw=grid_thw)
            if pixel_values is None:
                image_embeds = None
                video_embeds = mixed_embeds
            elif pixel_values_videos is None:
                image_embeds = mixed_embeds
                video_embeds = None
            else:
                merge_length = processor.image_processor.merge_size**2
                image_tokens = (image_grid_thw.prod(dim=-1) // merge_length).sum()
                image_embeds = mixed_embeds[:image_tokens]
                video_embeds = mixed_embeds[image_tokens:]

            if image_embeds is not None:
                image_mask = (input_ids == config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask = image_mask.to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if video_embeds is not None:
                video_mask = (input_ids == config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                video_mask = video_mask.to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        return inputs_embeds



class Qwen2_5VLBridge(MultimodalGPTBridge):
    # Compatible with older versions of transformers
    hf_state_dict_mapping = {
        'model.layers': 'model.language_model.layers',
        'model.embed_tokens': 'model.language_model.embed_tokens',
        'model.norm': 'model.language_model.norm',
        'visual': 'model.visual',
    }


register_megatron_model(
    MegatronModelMeta(
        MegatronModelType.qwen2_5_vl, [
            ModelType.qwen2_5_vl,
        ], bridge_cls=Qwen2_5VLBridge, visual_cls=Qwen2_5VL_Vit))


class Qwen2VL_Vit(Qwen2_5VL_Vit):
    version = 'v2'


register_megatron_model(
    MegatronModelMeta(
        MegatronModelType.qwen2_vl, [
            ModelType.qwen2_vl,
        ], bridge_cls=Qwen2_5VLBridge, visual_cls=Qwen2VL_Vit))
