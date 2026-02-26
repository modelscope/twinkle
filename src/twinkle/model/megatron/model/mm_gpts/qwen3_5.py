# Copyright (c) ModelScope Contributors. All rights reserved.
# Reference: swift/swift/megatron/model/mm_gpts/qwen3_5.py
# Qwen3.5 / Qwen3.5-MoE multimodal model support for Megatron

import torch
from PIL import Image

from twinkle.model.megatron.args import get_args
from twinkle.utils.torch_utils import to_device
from ..constant import MegatronModelType, ModelType
from ..gpt_bridge import GPTBridge, MultimodalGPTBridge
from ..gpts.qwen3_next import Qwen3_5MoeGatedDeltaNet, Qwen3NextLoader
from ..register import MegatronModelMeta, register_megatron_model
from .utils import HuggingFaceModule


class Qwen3_5Vit(HuggingFaceModule):
    """Vision module for Qwen3.5 / Qwen3.5-MoE models.

    Maps 'model.visual' from HF model to 'visual' in Megatron,
    with merger as aligner.
    """
    module_mapping = {'model.visual': 'visual'}
    _vision_tower = ['visual']
    _aligner = ['visual.merger']

    def __init__(self, config):
        try:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
        except ImportError:
            Qwen3_5TextModel = None
        try:
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel
        except ImportError:
            Qwen3_5MoeTextModel = None
        ignore_cls = [c for c in [Qwen3_5TextModel, Qwen3_5MoeTextModel] if c is not None]
        super().__init__(config, ignore_cls)

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        return self._get_inputs_embeds_hf(inputs_embeds, kwargs, self.visual, self.processor, self.model_config)

    def _get_inputs_embeds_hf(self, inputs_embeds, inputs, visual, processor, config):
        input_ids = inputs['input_ids']
        pixel_values = inputs.get('pixel_values')
        pixel_values_videos = inputs.get('pixel_values_videos')
        image_grid_thw = inputs.get('image_grid_thw')
        video_grid_thw = inputs.get('video_grid_thw')
        dtype = visual.dtype
        if pixel_values is None and pixel_values_videos is None:
            images = [Image.new('RGB', (32, 32), (0, 0, 0))]
            media_inputs = processor.image_processor(images=images, return_tensors='pt')
            media_inputs = to_device(media_inputs, input_ids.device)
            pixel_values = media_inputs['pixel_values'].type(dtype)
            image_embeds = visual(pixel_values, grid_thw=media_inputs['image_grid_thw'])
            if hasattr(image_embeds, 'pooler_output'):
                image_embeds = image_embeds.pooler_output
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
            if hasattr(mixed_embeds, 'pooler_output'):
                mixed_embeds = mixed_embeds.pooler_output
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


class Qwen3_5Bridge(MultimodalGPTBridge):
    """Bridge for Qwen3.5 multimodal models.

    Uses language_model prefix for the LLM backbone since Qwen3.5 has a
    multimodal architecture with model.language_model.layers structure.

    Overrides _set_layer_attn to handle the mixed linear/full attention
    architecture specific to Qwen3-Next/Qwen3.5.
    """
    hf_layers_prefix = 'model.language_model.layers'
    hf_embed_key = 'model.language_model.embed_tokens.weight'
    hf_final_layernorm_key = 'model.language_model.norm.weight'
    hf_mtp_prefix = 'mtp.layers'

    def _set_layer_attn(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool):
        args = self.args
        layer_types = getattr(args, 'layer_types', None)
        if layer_types is None:
            return super()._set_layer_attn(mg_layer, hf_state_dict, layer_idx, to_mcore)

        layer_type = layer_types[layer_idx] if 0 <= layer_idx < len(layer_types) else 'full_attention'
        mg_attn = None if mg_layer is None else mg_layer.self_attention
        if layer_type == 'linear_attention':
            hf_state_dict.update(self._set_module(mg_attn, hf_state_dict, 'linear_attn.', to_mcore))
        elif layer_type == 'full_attention':
            hf_state_dict.update(self._set_attn_state(mg_attn, hf_state_dict, 'self_attn.', layer_idx, to_mcore))
        self._set_state_dict(mg_layer, 'input_layernorm.weight', hf_state_dict, 'input_layernorm.weight', to_mcore)
        return hf_state_dict

    def _convert_mtp_extra(self, mtp_layer, hf_state_dict, to_mcore, origin_hf_state_dict):
        hf_state_dict = self._remove_prefix(origin_hf_state_dict, 'mtp.')
        for mg_key, key in zip(['enorm.weight', 'hnorm.weight', 'eh_proj.weight'],
                               ['pre_fc_norm_embedding.weight', 'pre_fc_norm_hidden.weight', 'fc.weight']):
            self._set_state_dict(mtp_layer, mg_key, hf_state_dict, key, to_mcore)
        self._set_state_dict(mtp_layer, 'final_layernorm.weight', hf_state_dict, 'norm.weight', to_mcore)
        if not to_mcore:
            origin_hf_state_dict.update(self._add_prefix(hf_state_dict, 'mtp.'))


try:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration
except ImportError:
    Qwen3_5MoeForConditionalGeneration = None

_auto_model_cls = Qwen3_5MoeForConditionalGeneration
if _auto_model_cls is None:
    try:
        from transformers import AutoModel
        _auto_model_cls = AutoModel
    except ImportError:
        _auto_model_cls = None


class Qwen3_5MoeLoader(Qwen3NextLoader):
    gated_delta_net = Qwen3_5MoeGatedDeltaNet


register_megatron_model(
    MegatronModelMeta(
        MegatronModelType.qwen3_5,
        [
            ModelType.qwen3_5,
            ModelType.qwen3_5_moe,
        ],
        bridge_cls=Qwen3_5Bridge,
        visual_cls=Qwen3_5Vit,
        auto_model_cls=_auto_model_cls,
        loader=Qwen3_5MoeLoader,
    ))
