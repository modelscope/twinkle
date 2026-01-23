# Copyright (c) twinkle authors. All rights reserved.
# TODO: test

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union
import logging

import numpy as np
import torch
import torch.nn as nn

from twinkle import DeviceMesh, remote_class, remote_function
from twinkle.data_format import InputFeature
from twinkle.hub import HubOperation

from .base import InputProcessor

logger = logging.getLogger(__name__)


@remote_class()
class VLMProcessor(InputProcessor):
    """
    Base processor for Vision-Language Models.
    
    Subclass requirements:
    - _build_messages(): Convert user messages to HF processor format (if different)
    
    Optional overrides:
    - _get_vision_token_id(): For post_encode embedding merge
    - _get_vision_embeddings(): For post_encode embedding extraction
    - _get_position_ids(): For models with special position calculation (e.g., Qwen-VL 3D RoPE)
    """
    
    # VLM fields to concatenate (not pad) in batch
    VLM_CONCAT_FIELDS = {
        'pixel_values', 'image_grid_thw', 
        'pixel_values_videos', 'video_grid_thw',
        'input_features', 'feature_attention_mask',
        'grid_thws',
    }
    
    padding_map = {
        **InputProcessor.padding_map,
        'pixel_values': 0.0,
        'image_grid_thw': 0,
        'pixel_values_videos': 0.0,
        'video_grid_thw': 0,
        'input_features': 0.0,
        'feature_attention_mask': 0,
    }
    
    # Placeholder tokens in user text
    image_placeholder: str = '<image>'
    video_placeholder: str = '<video>'
    audio_placeholder: str = '<audio>'
    
    def __init__(
        self,
        model_id: str,
        device_mesh: Optional[DeviceMesh] = None,
        max_length: int = 8192,
        padding_side: str = 'right',
        **kwargs
    ):
        super().__init__(device_mesh=device_mesh, padding_side=padding_side, **kwargs)
        self.model_id = model_id
        self.max_length = max_length
        self.model = None
        self.config = None
        self._load_hf_processor(model_id)
    
    def _load_hf_processor(self, model_id: str):
        """Load HuggingFace processor."""
        from transformers import AutoProcessor
        model_path = HubOperation.download_model(model_id)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = getattr(self.processor, 'tokenizer', self.processor)
    
    def set_model(self, model: nn.Module):
        """Set model for post_encode."""
        self.model = model
        self.config = getattr(model, 'config', None)
    
    # =========================================================================
    # Core: encode sample -> InputFeature
    # =========================================================================
    
    def encode(self, sample: Dict[str, Any]) -> InputFeature:
        """
        Encode sample with text and media.
        
        Args:
            sample: {'messages': [...], 'images': [...], 'videos': [...]}
        
        Returns:
            InputFeature with input_ids, labels, attention_mask, pixel_values, etc.
        """
        messages = sample.get('messages', [])
        images = self._load_images(sample.get('images', []))
        videos = self._load_videos(sample.get('videos', []))
        
        # Convert to HF processor format
        processed_messages = self._build_messages(messages, images, videos)
        
        # Get text via chat template
        text = self.processor.apply_chat_template(
            processed_messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # HF processor handles: tokenization + image processing + token expansion
        inputs = self.processor(
            text=[text],
            images=images if images else None,
            videos=videos if videos else None,
            return_tensors='pt',
            padding=False
        )
        
        # Create labels (assistant tokens only)
        labels = self._create_labels(messages, inputs['input_ids'])
        
        # Convert to numpy feature
        result = self._to_feature_dict(inputs)
        result['labels'] = labels.numpy().squeeze(0) if isinstance(labels, torch.Tensor) else labels
        
        return InputFeature(**result)
    
    def _load_images(self, images: List) -> List:
        """Load images to PIL format."""
        from PIL import Image
        import io
        
        loaded = []
        for img in images:
            if img is None:
                continue
            if isinstance(img, Image.Image):
                loaded.append(img)
            elif isinstance(img, str):
                if img.startswith(('http://', 'https://')):
                    import requests
                    resp = requests.get(img, timeout=30)
                    loaded.append(Image.open(io.BytesIO(resp.content)))
                else:
                    loaded.append(Image.open(img))
            elif isinstance(img, bytes):
                loaded.append(Image.open(io.BytesIO(img)))
            elif isinstance(img, dict) and 'bytes' in img:
                loaded.append(Image.open(io.BytesIO(img['bytes'])))
            else:
                loaded.append(img)
        return loaded
    
    def _load_videos(self, videos: List) -> List:
        """Load videos. Override if special handling needed."""
        return videos if videos else []
    
    def _build_messages(
        self,
        messages: List[Dict[str, Any]],
        images: List,
        videos: List
    ) -> List[Dict[str, Any]]:
        """
        Convert messages to HF processor's expected format.
        
        Most HF processors accept standard OpenAI format:
        [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': '...'}]}]
        
        Default implementation converts <image> placeholder to type markers.
        Override only if model needs different format.
        """
        result = []
        image_idx = 0
        video_idx = 0
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if isinstance(content, str):
                # Check for placeholders
                has_image = self.image_placeholder in content
                has_video = self.video_placeholder in content
                
                if has_image or has_video:
                    new_content = []
                    # Split by placeholders and insert type markers
                    remaining = content
                    while remaining:
                        img_pos = remaining.find(self.image_placeholder) if has_image else -1
                        vid_pos = remaining.find(self.video_placeholder) if has_video else -1
                        
                        # Find next placeholder
                        if img_pos == -1 and vid_pos == -1:
                            if remaining.strip():
                                new_content.append({'type': 'text', 'text': remaining})
                            break
                        
                        # Determine which comes first
                        if vid_pos == -1 or (img_pos != -1 and img_pos < vid_pos):
                            # Image placeholder
                            if remaining[:img_pos].strip():
                                new_content.append({'type': 'text', 'text': remaining[:img_pos]})
                            if image_idx < len(images):
                                new_content.append({'type': 'image'})
                                image_idx += 1
                            remaining = remaining[img_pos + len(self.image_placeholder):]
                        else:
                            # Video placeholder
                            if remaining[:vid_pos].strip():
                                new_content.append({'type': 'text', 'text': remaining[:vid_pos]})
                            if video_idx < len(videos):
                                new_content.append({'type': 'video'})
                                video_idx += 1
                            remaining = remaining[vid_pos + len(self.video_placeholder):]
                    
                    result.append({'role': role, 'content': new_content})
                else:
                    result.append({'role': role, 'content': content})
            
            elif isinstance(content, list):
                # Already in list format, normalize
                new_content = []
                for part in content:
                    ptype = part.get('type', 'text')
                    if ptype == 'text':
                        new_content.append({'type': 'text', 'text': part.get('text', '')})
                    elif ptype in ('image', 'image_url'):
                        new_content.append({'type': 'image'})
                    elif ptype == 'video':
                        new_content.append({'type': 'video'})
                    else:
                        new_content.append(part)
                result.append({'role': role, 'content': new_content})
            else:
                result.append({'role': role, 'content': str(content)})
        
        return result
    
    def _to_feature_dict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert HF processor output to numpy dict."""
        result = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                arr = value.numpy()
                # Keep VLM concat fields as-is, squeeze batch dim for others
                if key in self.VLM_CONCAT_FIELDS:
                    result[key] = arr
                elif arr.ndim > 1 and arr.shape[0] == 1:
                    result[key] = arr[0]
                else:
                    result[key] = arr
            else:
                result[key] = value
        return result

    def _create_labels(
        self,
        messages: List[Dict[str, Any]],
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Create labels. Only assistant tokens should have loss.
        
        Limitation: Current approach cannot distinguish which assistant turns
        to include in loss for multi-turn conversations. For that, you need
        replace_tag architecture.
        """
        # Try tokenizer's assistant mask
        try:
            result = self.tokenizer.apply_chat_template(
                messages,
                return_assistant_tokens_mask=True,
                return_dict=True,
                tokenize=True
            )
            if 'assistant_masks' in result and sum(result['assistant_masks']) > 0:
                mask = torch.tensor(result['assistant_masks'])
                ids = torch.tensor(result['input_ids'])
                labels = torch.where(mask.bool(), ids, torch.tensor(-100))
                # Align length
                target_len = input_ids.shape[-1]
                if labels.shape[0] < target_len:
                    labels = torch.cat([labels, torch.full((target_len - labels.shape[0],), -100)])
                elif labels.shape[0] > target_len:
                    labels = labels[:target_len]
                return labels.unsqueeze(0)
        except Exception as e:
            raise NotImplementedError(f"Tokenizer assistant_mask not available: {e}")
            
        #     logger.debug(f"Tokenizer assistant_mask not available: {e}, use heuristic")
        
        # # Fallback: heuristic
        # labels = self._create_labels_heuristic(messages, input_ids)
        
        # # Validate
        # if isinstance(labels, torch.Tensor) and (labels != -100).sum().item() == 0:
        #     logger.warning("Labels heuristic failed: no valid tokens.")
        
        # return labels
    
    def _create_labels_heuristic(
        self,
        messages: List[Dict[str, Any]],
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Create labels by finding assistant content in token sequence."""
        input_ids_np = input_ids.numpy() if isinstance(input_ids, torch.Tensor) else input_ids
        labels = np.full_like(input_ids_np, -100)
        input_flat = input_ids_np.flatten()
        labels_flat = labels.flatten()
        
        for msg in messages:
            if msg.get('role') != 'assistant':
                continue
            
            content = msg.get('content', '')
            if not content or not isinstance(content, str):
                continue
            
            try:
                content_ids = self.tokenizer.encode(content, add_special_tokens=False)
                if not content_ids:
                    continue
                
                # Search from end
                for i in range(len(input_flat) - len(content_ids), -1, -1):
                    if list(input_flat[i:i+len(content_ids)]) == content_ids:
                        labels_flat[i:i+len(content_ids)] = input_flat[i:i+len(content_ids)]
                        break
            except Exception:
                continue
        
        return torch.from_numpy(labels_flat.reshape(input_ids_np.shape))
    
    # =========================================================================
    # Batch collation
    # =========================================================================
    
    @remote_function()
    def collate_fn(self, inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch: concat VLM fields, pad text fields."""
        vlm_fields = {k: [] for k in self.VLM_CONCAT_FIELDS}
        text_inputs = []
        
        for inp in inputs:
            inp = dict(inp)
            for field in self.VLM_CONCAT_FIELDS:
                if field in inp:
                    vlm_fields[field].append(inp.pop(field))
            text_inputs.append(inp)
        
        result = super().collate_fn(text_inputs)
        
        for field, values in vlm_fields.items():
            if values:
                if isinstance(values[0], np.ndarray):
                    result[field] = torch.from_numpy(np.concatenate(values, axis=0))
                elif isinstance(values[0], torch.Tensor):
                    result[field] = torch.cat(values, dim=0)
        
        return result
    
    # =========================================================================
    # Post-encode (embedding merge for training)
    # =========================================================================
    
    def post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform inputs for model forward.
        
        Default: use helper methods for embedding merge.
        Override if model handles internally (like Qwen3-VL).
        """
        input_ids = inputs.get('input_ids')
        if input_ids is None:
            return inputs
        
        text_embeds = self._get_text_embeddings(model, input_ids)
        vision_embeds = self._get_vision_embeddings(model, inputs)
        
        if vision_embeds is not None:
            inputs_embeds = self._merge_vision_embeddings(
                text_embeds, vision_embeds, input_ids, inputs
            )
        else:
            inputs_embeds = text_embeds
        
        result = {k: v for k, v in inputs.items() if k != 'input_ids'}
        result['inputs_embeds'] = inputs_embeds
        return result
    
    def _get_text_embeddings(self, model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        """Get text embeddings from model."""
        embed_fn = None
        if hasattr(model, 'get_input_embeddings'):
            embed_fn = model.get_input_embeddings()
        elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            embed_fn = model.model.embed_tokens
        elif hasattr(model, 'language_model') and hasattr(model.language_model, 'embed_tokens'):
            embed_fn = model.language_model.embed_tokens
        
        if embed_fn is None:
            raise ValueError("Cannot find embedding layer in model")
        
        return embed_fn(input_ids)
    
    def _get_vision_embeddings(self, model: nn.Module, inputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Get vision embeddings. Override in subclass."""
        return None
    
    def _get_vision_token_id(self) -> Optional[int]:
        """Get vision placeholder token ID. Override in subclass."""
        return self.tokenizer.encode(self.image_placeholder)
    
    def _merge_vision_embeddings(
        self,
        text_embeds: torch.Tensor,
        vision_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        inputs: Dict[str, Any]
    ) -> torch.Tensor:
        """Merge vision embeddings at placeholder positions."""
        vision_token_id = self._get_vision_token_id()
        if vision_token_id is None:
            return text_embeds
        
        vision_mask = (input_ids == vision_token_id).unsqueeze(-1).expand_as(text_embeds)
        vision_embeds = vision_embeds.to(device=text_embeds.device, dtype=text_embeds.dtype)
        vision_mask = vision_mask.to(device=text_embeds.device)
        
        return text_embeds.masked_scatter(vision_mask, vision_embeds)
    
    def _get_position_ids(self, inputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Get position_ids. Override for models with special position encoding."""
        return None


# =============================================================================
# Qwen VL Processor
# =============================================================================

@remote_class()
class Qwen3VLProcessor(VLMProcessor):
    """
    Processor for Qwen VL series.
    
    Note: Qwen3-VL handles embedding merge internally in forward(),
    so post_encode just passes through.
    """
    
    # _build_messages: Uses base class implementation.
    # Qwen's HF processor accepts the standard format:
    # [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': '...'}]}]
    
    def post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
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
            logger.warning(f"Failed to get rope_index: {e}")
            return None
