import inspect
import numpy as np
import os
import tempfile
import torch
from contextlib import contextmanager
from copy import copy
from io import BytesIO
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Union

from twinkle import remote_class, requires
from twinkle.data_format import InputFeature
from twinkle.template.base import ImageInput, Template, VideoInput
from twinkle.template.utils import get_inputs_embeds_hf
from twinkle.utils import load_mm_file

_ROPE_INDEX_CACHE: Dict[str, Callable] = {}


@contextmanager
def _as_local_video_path(video: str):
    """Yield a path ``qwen_vl_utils.fetch_video`` can read, materializing base64 to a temp file.

    The video reader backends (torchvision / decord / torchcodec) all document support for local
    paths, ``file://`` and ``http(s)://`` -- but not base64, which ``fetch_image`` does accept. So
    only base64 needs handling here; everything else is passed through untouched.

    A temp file (rather than the ``BytesIO`` that ``load_mm_file`` returns) is required because
    ``fetch_video`` dispatches on ``isinstance(ele['video'], str)`` and treats any non-str as a list
    of frames.
    """
    if not (isinstance(video, str) and video.strip().startswith('data:')):
        yield video
        return
    data = load_mm_file(video)
    payload = data.getvalue() if isinstance(data, BytesIO) else data
    # Suffix-less is fine: all three backends sniff the container rather than trust the extension.
    tmp = tempfile.NamedTemporaryFile(prefix='twinkle_video_', delete=False)
    try:
        tmp.write(payload)
        tmp.close()
        yield tmp.name
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _build_rope_index_func(config) -> Callable:
    arch = config.architectures[0]
    fn = _ROPE_INDEX_CACHE.get(arch)
    if fn is not None:
        return fn
    import transformers
    with torch.device('meta'):
        model_cls = getattr(transformers, arch)
        dummy_model = model_cls(config)
    for _, sub_module in dummy_model.named_modules():
        if hasattr(sub_module, 'get_rope_index'):
            _ROPE_INDEX_CACHE[arch] = sub_module.get_rope_index
            return sub_module.get_rope_index
    raise NotImplementedError(f'Module {dummy_model.__class__.__name__} has no get_rope_index method!')


@remote_class()
class Qwen3_5Template(Template):
    """
    Processor for Qwen VL series.

    Note: Qwen3-VL handles embedding merge internally in forward(),
    so post_encode just passes through inputs unchanged.
    """

    def __init__(self,
                 *args,
                 image_min_pixels: Optional[int] = None,
                 image_max_pixels: Optional[int] = None,
                 video_min_pixels: Optional[int] = None,
                 video_max_pixels: Optional[int] = None,
                 video_total_pixels: Optional[int] = None,
                 fps: Optional[float] = None,
                 video_min_frames: Optional[int] = None,
                 video_max_frames: Optional[int] = None,
                 nframes: Optional[int] = None,
                 **kwargs):
        """Qwen-VL template.

        The media arguments tune ``qwen_vl_utils`` preprocessing (resolution budget and video frame
        sampling). They are forwarded per call inside the element dict that ``fetch_image`` /
        ``fetch_video`` accept, so each template instance keeps its own settings -- unlike writing
        the ``vision_process`` module globals, which every instance in the process would share.
        Leaving one as ``None`` defers to the ``qwen_vl_utils`` default.

        Args:
            image_min_pixels / image_max_pixels: per-image resolution budget, in pixels.
            video_min_pixels / video_max_pixels: per-frame resolution budget, in pixels.
            video_total_pixels: budget across all sampled frames of one video.
            fps: frame sampling rate. Mutually exclusive with ``nframes``.
            video_min_frames / video_max_frames: clamp the frame count ``fps`` produces.
            nframes: exact frame count to sample. Mutually exclusive with ``fps``.
        """
        assert fps is None or nframes is None, 'Pass either `fps` or `nframes`, not both.'
        super().__init__(*args, **kwargs)
        # Only non-None entries, so absent keys fall through to the qwen_vl_utils defaults.
        self._image_kwargs = {
            k: v
            for k, v in {
                'min_pixels': image_min_pixels,
                'max_pixels': image_max_pixels,
            }.items() if v is not None
        }
        self._video_kwargs = {
            k: v
            for k, v in {
                'min_pixels': video_min_pixels,
                'max_pixels': video_max_pixels,
                'total_pixels': video_total_pixels,
                'fps': fps,
                'min_frames': video_min_frames,
                'max_frames': video_max_frames,
                'nframes': nframes,
            }.items() if v is not None
        }
        # Fix upstream Qwen3 chat_template parse bugs (orphan </think> handling).
        # Deferred import to avoid cycles; idempotent across Ray actor re-init.
        from twinkle.patch import apply_patch
        from twinkle.patch.qwen3_chat_template import Qwen3AllowToolTailTemplate, Qwen3ChatTemplate
        apply_patch(self.tokenizer, Qwen3ChatTemplate)
        # Allow ScoreFilter to render multi-turn agent prefixes ending in `tool`.
        apply_patch(self.tokenizer, Qwen3AllowToolTailTemplate)
        # Qwen3VLProcessor carries its own chat_template; _apply_chat_template
        # routes through self.processor, so the patch must be applied there too.
        if self.processor is not self.tokenizer:
            apply_patch(self.processor, Qwen3ChatTemplate)
            apply_patch(self.processor, Qwen3AllowToolTailTemplate)
        # Patches above may alter chat_template behavior — re-probe assistant_masks
        # support so `_template_support_assistant_tokens_mask` reflects the patched template.
        self._test_support_assistant_tokens_mask()
        self._patch_size: Optional[int] = None
        self._merge_size: Optional[int] = None
        self._init_vision_config()

    @property
    def rope_index_func(self) -> Callable:
        """Lazily resolve the rope-index function via a module-level cache.

        Kept off ``self`` so the template's ``__dict__`` stays free of
        ``nn.Module`` state, which in turn keeps ``dill.dumps(template)``
        deterministic for HF datasets fingerprinting.
        """
        return _build_rope_index_func(self.config)

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
        image_input.update(self._image_kwargs)
        return fetch_image(image_input, image_patch_size=self.patch_size)

    def preprocess_video(self, video: VideoInput) -> Union[List[Image.Image], torch.Tensor]:
        requires('qwen_vl_utils')
        from qwen_vl_utils.vision_process import fetch_video

        if isinstance(video, str):
            # Base64 has to become a real file first: the reader backends only take paths/URLs.
            with _as_local_video_path(video) as video_path:
                video_input = {'video': video_path, **self._video_kwargs}
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
        image_grid_thw = input_feature.get('image_grid_thw')
        video_grid_thw = input_feature.get('video_grid_thw')
        has_image_grid = image_grid_thw is not None and (torch.is_tensor(image_grid_thw) and image_grid_thw.numel() > 0)
        has_video_grid = video_grid_thw is not None and (torch.is_tensor(video_grid_thw) and video_grid_thw.numel() > 0)
        if 'mm_token_type_ids' in inspect.signature(self.rope_index_func).parameters:
            mm_token_type_ids = torch.zeros_like(input_ids)
            if has_image_grid:
                mm_token_type_ids[input_ids == self.processor.image_token_id] = 1
            if has_video_grid:
                mm_token_type_ids[input_ids == self.processor.video_token_id] = 2
            kwargs['mm_token_type_ids'] = mm_token_type_ids
        position_ids, _ = self.rope_index_func(
            input_ids,
            image_grid_thw=image_grid_thw if has_image_grid else None,
            video_grid_thw=video_grid_thw if has_video_grid else None,
            attention_mask=attention_mask,
            **kwargs)
        return self._concat_text_position_ids(position_ids)

    def get_vllm_input_ids(self, input_ids):
        """Collapse each <vision_start> <image_pad>... <vision_end> group
        into <vision_start> <image_pad> <vision_end> (single pad token)."""
        image_token_id = self.config.image_token_id
        vision_start_id = self.config.vision_start_token_id
        vision_end_id = self.config.vision_end_token_id

        result = []
        i = 0
        while i < len(input_ids):
            if input_ids[i] == vision_start_id:
                result.append(vision_start_id)
                i += 1
                # Skip all consecutive image_pad tokens, keep only one
                found_pad = False
                while i < len(input_ids) and input_ids[i] == image_token_id:
                    if not found_pad:
                        result.append(image_token_id)
                        found_pad = True
                    i += 1
                # Append vision_end if present
                if i < len(input_ids) and input_ids[i] == vision_end_id:
                    result.append(vision_end_id)
                    i += 1
            else:
                result.append(input_ids[i])
                i += 1
        return result

    @staticmethod
    def _concat_text_position_ids(position_ids):
        seq_len = position_ids.shape[-1]
        text_position_ids = torch.arange(seq_len, device=position_ids.device).expand(1, *position_ids.shape[1:])
        return torch.concat([text_position_ids, position_ids], dim=0)
