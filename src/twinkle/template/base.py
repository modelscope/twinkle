# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import os
from collections.abc import Callable, Mapping
from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import json
import numpy as np
from .contract.encoding import engine_input_ids, render_delta as render_delta_tokens
from .contract.output import _reasoning_is_open, decode_output
from twinkle.data_format import copy_trajectory

from twinkle import remote_class
from twinkle.data_format import InputFeature, Message, Trajectory, user_data_get
from twinkle.hub import HubOperation
from twinkle.utils import get_logger, load_image, to_device, trust_remote_code
from .contract import EncodeIntent, EngineBackend, MaxLengthError, check_backend, resolve_intent
from .contract.encoding import check_generation_prompt, normalize_token_fields, truncate_rows

logger = get_logger()
from .contract import (
    DecodeContext,
    DecodedResponse,
    MediaBundle,
    SHIFTED_FIELDS,
    append_and_finalize,
    feature_has_media,
    shift_values,
    split_reasoning,
)
from .contract.encoding import preserve_trajectory, project_sliced_messages
from .tools import ToolCallRegistry
from .utils import TokenizeByRound, transfer_to_standard_message

if TYPE_CHECKING:
    import torch
    from PIL import Image

# Type aliases for multimodal data
ImageInput = Union[str, "Image.Image", "torch.Tensor"]
VideoInput = Union[str, list["Image.Image"], "torch.Tensor"]
AudioInput = Union[str, np.ndarray, "torch.Tensor"]

# What an appended turn is to a trainer: the policy's own completion (scored, and
# a log-prob exists for each of its tokens), someone else's completion offered
# for imitation (scored, no log-prob -- usable by SFT but not by RL), or history
# that no loss may touch. There is deliberately no fourth role: a log-prob is
# only ever needed for a token that is also scored.
_APPEND_ROLES = ("completion", "demonstration", "context")


class _RegistryParser:
    def __init__(self, template):
        self.template = template

    def parse(self, text, tools=None):
        return self.template.parse_tool_call(text)

    def parse_errors(self, text, tools=None):
        return self.template.tool_call_errors(text)

    def clean(self, text):
        return self.template.clean_tool_call(text)


@remote_class()
class Template:
    media_types = frozenset()
    reasoning_open_marker = "<think>"
    reasoning_close_marker = "</think>"

    # Placeholder tokens in user text
    image_placeholder: str = "<image>"
    video_placeholder: str = "<video>"
    audio_placeholder: str = "<audio>"

    # Encode pipeline stages — class-level so deepcopy/dill fingerprints stay stable
    # and subclasses can override at class scope rather than re-assigning per instance.
    pre_pipeline_names: list[str] = [
        "_add_default_system",
        "_to_standard_reasoning_content",
        "_build_standard_messages",
    ]
    post_pipeline_names: list[str] = [
        "_check_max_length",
        "_add_attention_fields",
        "_roll_labels",
    ]

    placeholder_tokens = ()
    task_type = 'causal_lm'
    default_enable_thinking = False
    _server_template_name = None

    def __init__(
        self,
        model_id: str | None = None,
        use_chat_template: bool = True,
        max_length: int | None = None,
        truncation_strategy: Literal['raise', 'left', 'right', 'split', 'delete'] = 'raise',
        default_system: str | None = None,
        enable_thinking: bool | None = None,
        media_types: set[Literal['image', 'video', 'audio']] | None = None,
        *, processor=None, preserve_thinking=None, padding_side='right', **kwargs,
    ):
        self.model_id = model_id or getattr(processor, 'name_or_path', None)
        self._processor_kwargs = dict(kwargs)
        if processor is None:
            if model_id is None:
                raise ValueError('model_id or processor is required to construct a template')
            # Custom-code checkpoints (auto_map) without trust_remote_code send transformers into
            # its interactive trust prompt, which installs signal.SIGALRM -- a ValueError on any
            # non-main thread (e.g. the Ray actor worker thread running set_template). Follow the
            # model loaders' trust convention, gated by twinkle's own policy
            # (TWINKLE_TRUST_REMOTE_CODE); an explicit caller kwarg still wins.
            kwargs.setdefault('trust_remote_code', trust_remote_code())
            model_dir = HubOperation.download_model(model_id, ignore_model=True)
            from transformers import AutoConfig, AutoProcessor, AutoTokenizer
            cls = AutoProcessor if any(os.path.exists(os.path.join(model_dir, name)) for name in
                                       ('preprocessor_config.json', 'processor_config.json')) else AutoTokenizer
            processor = cls.from_pretrained(model_dir, **kwargs)
            self.config = AutoConfig.from_pretrained(model_dir, **kwargs)
        else:
            info = getattr(processor, 'model_info', None)
            self.config = getattr(info, 'config', None) or getattr(processor, 'config', None)
            if self.config is None:
                from transformers import AutoConfig
                if self.model_id is None:
                    raise ValueError('processor must provide config or a model_id')
                self.config = AutoConfig.from_pretrained(self.model_id, **kwargs)
        self.processor = processor
        self._init_common(model_id=self.model_id, max_length=max_length, truncation_strategy=truncation_strategy,
                          use_chat_template=use_chat_template, enable_thinking=enable_thinking,
                          preserve_thinking=preserve_thinking, media_types=media_types, padding_side=padding_side)
        self.default_system = default_system
        logger.info(f'default_system: {self.default_system!r}')
        logger.info(f'max_length: {self.max_length}')
        self._test_support_assistant_tokens_mask()

    def _init_common(self, *, model_id=None, max_length=None, truncation_strategy='raise',
                     use_chat_template=True, enable_thinking=None, preserve_thinking=None,
                     media_types=None, padding_side='right'):
        if max_length is not None and (isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0):
            raise ValueError('max_length must be a positive integer or None')
        if truncation_strategy not in ('raise', 'left', 'right', 'split', 'delete'):
            raise ValueError(f'Invalid truncation_strategy: {truncation_strategy}')
        if padding_side not in ('left', 'right'):
            raise ValueError(f'Invalid padding_side: {padding_side!r}')
        self.model_id = model_id
        if max_length is None:
            info = getattr(getattr(self, 'processor', None), 'model_info', None)
            config = getattr(self, 'config', None)
            text_config = getattr(config, 'text_config', None) or config
            max_length = getattr(info, 'max_model_len', None) or getattr(text_config, 'max_position_embeddings', None)
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy
        self.use_chat_template = use_chat_template
        self._explicit_enable_thinking = enable_thinking is not None
        self.enable_thinking = self.default_enable_thinking if enable_thinking is None else enable_thinking
        self.preserve_thinking = preserve_thinking
        self.padding_side = padding_side
        self.media_types = frozenset(media_types) if media_types is not None else self._infer_media_types()

    def _constructor_config(self):
        return {**getattr(self, '_processor_kwargs', {}), 'model_id': self.model_id,
                'max_length': self.max_length, 'truncation_strategy': self.truncation_strategy,
                'use_chat_template': self.use_chat_template, 'default_system': self.default_system,
                'enable_thinking': self.enable_thinking if self._explicit_enable_thinking else None,
                'preserve_thinking': self.preserve_thinking, 'padding_side': self.padding_side,
                'media_types': sorted(self.media_types)}

    def parse_tool_call(self, decoded: str) -> list[dict[str, Any]]:
        """Parse tool calls from the assistant's decoded output.

        Polls registered :class:`ToolCallParser` in order; first parser whose
        ``detect`` matches takes ownership and produces the result. Other
        parsers are not invoked on the same text — prevents nested re-extraction.
        """
        parser = ToolCallRegistry.detect_first(decoded or "")
        return parser.parse(decoded) if parser else []

    def clean_tool_call(self, decoded: str) -> str:
        """Strip tool-call markup using the same parser that ``parse_tool_call`` would pick."""
        parser = ToolCallRegistry.detect_first(decoded or "")
        return parser.clean(decoded) if parser else (decoded or "").rstrip()

    def tool_call_errors(self, decoded: str) -> list[str]:
        """Why ``parse_tool_call`` returned fewer calls than the text asked for.

        Same parser choice as ``parse_tool_call``, so the two describe one pass
        over the reply. Empty when the reply carries no tool-call markup at all --
        a reply that simply answered is not a failure.
        """
        parser = ToolCallRegistry.detect_first(decoded or "")
        return parser.parse_errors(decoded) if parser else []

    @property
    def tool_call_stop(self) -> str | None:
        """The string a caller stops generation at to hold a reply to one tool call.

        Cannot be answered by ``detect``, which needs the reply that does not
        exist yet, so each template names its own format. None means the format
        has no closing marker -- ReAct and a bare call list end where the reply
        does -- and a caller then lets the reply run to its end.
        """
        return None

    def _infer_media_types(self):
        media_types = set()
        for media_type in ("image", "video", "audio"):
            if getattr(self.config, f"{media_type}_token_id", None) is not None:
                media_types.add(media_type)
        if hasattr(self.processor, "image_processor"):
            media_types.add("image")
        if hasattr(self.processor, "video_processor"):
            media_types.add("video")
        if hasattr(self.processor, "audio_processor") or hasattr(self.processor, "feature_extractor"):
            media_types.add("audio")
        return frozenset(media_types)

    @property
    def tokenizer(self):
        tokenizer = self.processor
        if hasattr(tokenizer, "tokenizer"):
            tokenizer = tokenizer.tokenizer
        return tokenizer

    @property
    def is_multimodal(self) -> bool:
        return bool(self.media_types)

    def _test_support_assistant_tokens_mask(self):
        # Multimodal processors require standard content blocks.
        if self.is_multimodal:
            dummy_inputs = [
                {"role": "user", "content": [{"type": "text", "text": "How are you?"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Fine."}]},
            ]
        else:
            dummy_inputs = [
                Message(role="user", content="How are you?"),
                Message(role="assistant", content="Fine."),
            ]
        try:
            outputs = self.processor.apply_chat_template(
                dummy_inputs, return_assistant_tokens_mask=True, return_dict=True, tokenize=True
            )
            # Check if outputs is a dict (not all processors return dict even with return_dict=True)
            if isinstance(outputs, dict) and "assistant_masks" in outputs:
                assistant_masks = outputs["assistant_masks"]
                self._template_support_assistant_tokens_mask = (
                    0 < np.array(assistant_masks).sum() < len(assistant_masks)
                )
            else:
                # Processor doesn't support return_dict properly
                self._template_support_assistant_tokens_mask = False
        except Exception:  # noqa
            # If any error occurs during testing, fall back to not supporting
            self._template_support_assistant_tokens_mask = False

    def preprocess_image(self, image: ImageInput) -> "Image.Image":
        if isinstance(image, dict):
            image = image.get("bytes") or image.get("path")
        return load_image(image)

    def preprocess_video(self, video: VideoInput) -> list["Image.Image"]:
        return video

    def preprocess_audio(self, audio: AudioInput) -> np.ndarray:
        return audio

    def preprocess_images(self, images: list[ImageInput]) -> list["Image.Image"]:
        """Preprocess a list of images."""
        return [self.preprocess_image(img) for img in images]

    def preprocess_videos(self, videos: list[VideoInput]) -> list[list["Image.Image"]]:
        """Preprocess a list of videos."""
        return [self.preprocess_video(video) for video in videos]

    def preprocess_audios(self, audios: list[AudioInput]) -> list[np.ndarray]:
        """Preprocess a list of audio clips."""
        return [self.preprocess_audio(audio) for audio in audios]

    def _invoke_pre_pipeline(
        self, trajectories: list[Trajectory], skip_stages: set[str] | None = None
    ) -> list[Trajectory]:
        skip_stages = skip_stages or set()
        current = trajectories
        for pipeline_name in self.pre_pipeline_names:
            if pipeline_name in skip_stages:
                continue
            pipeline: Callable[[Trajectory], list[Trajectory]] = getattr(self, pipeline_name)
            next_batch = []
            for trajectory in current:
                next_batch.extend(pipeline(trajectory))
            current = next_batch
        return current

    def _run_post_pipeline(self, input_features: list[InputFeature]) -> list[InputFeature]:
        current = input_features
        for pipeline_name in self.post_pipeline_names:
            pipeline: Callable[[InputFeature], list[InputFeature]] = getattr(self, pipeline_name)
            next_batch = []
            for input_feature in current:
                next_batch.extend(pipeline(input_feature))
            current = next_batch
        return current

    def finalize_appended_features(self, input_features: list[InputFeature]) -> list[InputFeature]:
        result = []
        for feature in input_features:
            try:
                result.extend(self._run_post_pipeline([feature]))
            except MaxLengthError:
                if self.truncation_strategy != 'delete':
                    raise
        return result

    def _append_feature(
        self,
        prompt_input_feature: InputFeature,
        new_tokens: list[int],
        *,
        appended_as: Literal["completion", "demonstration", "context"] = "completion",
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> InputFeature | None:
        if appended_as not in _APPEND_ROLES:
            raise ValueError(f'appended_as must be one of {sorted(_APPEND_ROLES)}, got {appended_as!r}')
        return append_and_finalize(
            prompt_input_feature,
            new_tokens,
            scored=appended_as != "context",
            sampled=appended_as == "completion",
            finalize=self.finalize_appended_features,
        )

    def _add_default_system(self, trajectory: Trajectory) -> list[Trajectory]:
        system = trajectory.get('system')
        if system is not None and trajectory['messages'][0].get('role') != 'system':
            trajectory['messages'].insert(0, Message(role='system', content=system))
        if self.use_chat_template and self.default_system:
            if trajectory["messages"][0]["role"] == "user":
                trajectory["messages"].insert(0, Message(role="system", content=self.default_system))
        return [trajectory]

    def _to_standard_reasoning_content(self, trajectory: Trajectory) -> list[Trajectory]:
        def _extract_reasoning_content(messages: list[Message]) -> list[Message]:
            result = []
            for message in messages:
                if not isinstance(message, dict):
                    result.append(message)
                    continue
                message = message.copy()
                if message.get("role") == "assistant":
                    content = message.get("content", "")
                    if "reasoning_content" not in message and isinstance(content, str):
                        reasoning_content, new_content = split_reasoning(
                            content,
                            open_marker=self.reasoning_open_marker,
                            close_marker=self.reasoning_close_marker,
                        )
                        if reasoning_content is not None:
                            message["reasoning_content"] = reasoning_content
                            message["content"] = new_content
                    # Always emit string (never None/missing) — keeps PyArrow struct schema
                    # stable across shards; empty string renders identically to None in jinja.
                    if not isinstance(message.get("reasoning_content"), str):
                        message["reasoning_content"] = ""

                result.append(message)

            return result

        trajectory["messages"] = _extract_reasoning_content(trajectory["messages"])
        return [trajectory]

    def set_mm_position_ids(self, input_feature: InputFeature):
        return np.arange(len(input_feature["input_ids"]))

    def _engine_input_ids(self, input_ids):
        return input_ids

    @staticmethod
    def _engine_ids_tuple(input_ids) -> tuple[int, ...]:
        """Normalize engine IDs to plain integers for request serialization."""
        return tuple(int(token) for token in input_ids)

    def _check_max_length(self, input_feature: InputFeature) -> list[InputFeature]:
        return truncate_rows(input_feature, self.max_length, self.truncation_strategy,
                             placeholder_tokens=self.placeholder_tokens, multimodal=self.is_multimodal,
                             supervised=self.task_type == 'causal_lm')

    def _add_attention_fields(self, input_feature: InputFeature) -> list[InputFeature]:
        if "input_ids" not in input_feature:
            return [input_feature]
        input_feature['attention_mask'] = [1] * len(input_feature['input_ids'])
        input_feature['length'] = len(input_feature['input_ids'])
        return [normalize_token_fields(self.prepare_row(input_feature))]

    def _roll_labels(self, input_feature: InputFeature) -> list[InputFeature]:
        if "input_ids" not in input_feature:
            return [input_feature]
        if self.task_type != 'causal_lm':
            return [input_feature]
        for key, fill in SHIFTED_FIELDS.items():
            if input_feature.get(key) is not None:
                input_feature[key] = shift_values(input_feature[key], fill)
        return [input_feature]

    def _process_mm_messages(self, messages: list, images: list, videos: list, audios: list) -> list:
        """Process multimodal content with trajectory-level media.

        Args:
            messages: List of messages to process
            images: Trajectory-level images
            videos: Trajectory-level videos
            audios: Trajectory-level audios
        """
        # Determine format: list or string (check first non-system message)
        is_list_format = any(isinstance(m.get("content"), list) for m in messages if m.get("role") != "system")

        if is_list_format:
            return self._process_mm_list_format(messages, images, videos, audios)
        else:
            return self._process_mm_string_format(messages, images, videos, audios)

    def _process_mm_list_format(self, messages: list, images: list, videos: list, audios: list) -> list:
        """Process list format content with trajectory-level media."""
        img_iter, vid_iter, aud_iter = iter(images), iter(videos), iter(audios)
        new_messages = []
        first_user_idx = None

        for idx, message in enumerate(messages):
            message = copy(message)
            content = message.get("content")

            if message.get("role") == "user" and first_user_idx is None:
                first_user_idx = idx

            # Non-user messages: convert to list format but don't consume iterator
            if message.get("role") != "user":
                if isinstance(content, str):
                    message["content"] = [{"type": "text", "text": content}] if content else []
                new_messages.append(message)
                continue

            # User messages: process list content and fill placeholders
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    # OpenAI vision/audio parts nest the media reference one level down --
                    # {"type": "image_url", "image_url": {"url": ...}} (likewise video_url/audio_url).
                    # Unwrap them in place to twinkle's {"type": <media>, "url": ...} so the reference is
                    # preprocessed into pixels exactly like the native format; left untouched the block
                    # reaches the chat template as an unknown type and the media is silently dropped.
                    if btype in ("image_url", "video_url", "audio_url"):
                        payload = block.pop(btype, None) or {}
                        block["url"] = payload.get("url") if isinstance(payload, dict) else payload
                        btype = block["type"] = btype[:-4]
                    # Check if block has inline data
                    has_data = any(block.get(k) for k in (btype, "url", "path"))
                    if has_data:
                        # Preprocess inline URL
                        for key in (btype, "url", "path"):
                            if key and block.get(key) is not None:
                                block[key] = getattr(self, f"preprocess_{btype}", lambda x: x)(block[key])
                                break
                    elif btype == "image":
                        url = next(img_iter, None)
                        if url:
                            block["url"] = url
                    elif btype == "video":
                        url = next(vid_iter, None)
                        if url:
                            block["url"] = url
                    elif btype == "audio":
                        url = next(aud_iter, None)
                        if url:
                            block["url"] = url
            elif isinstance(content, str):
                # Convert string content to list format
                message["content"] = [{"type": "text", "text": content}] if content else []

            new_messages.append(message)

        # Prepend remaining media to first user message
        if first_user_idx is not None:
            prepend = [{"type": "image", "url": u} for u in img_iter]
            prepend += [{"type": "video", "url": u} for u in vid_iter]
            prepend += [{"type": "audio", "url": u} for u in aud_iter]
            if prepend:
                msg = new_messages[first_user_idx]
                content = msg.get("content", [])
                if not isinstance(content, list):
                    content = [{"type": "text", "text": content}] if content else []
                msg["content"] = prepend + content

        return new_messages

    def _process_mm_string_format(self, messages: list, images: list, videos: list, audios: list) -> list:
        """Process string format content with trajectory-level media."""
        placeholders = [getattr(self, f'{kind}_placeholder') for kind in ('image', 'video', 'audio')]
        image_placeholder, video_placeholder, audio_placeholder = [
            value[0] if isinstance(value, (list, tuple)) else value for value in placeholders]
        # Count total placeholders across all messages
        total_img = sum(
            m.get("content", "").count(image_placeholder)
            for m in messages
            if isinstance(m.get("content"), str) and m.get("role") != "system"
        )
        total_vid = sum(
            m.get("content", "").count(video_placeholder)
            for m in messages
            if isinstance(m.get("content"), str) and m.get("role") != "system"
        )
        total_aud = sum(
            m.get("content", "").count(audio_placeholder)
            for m in messages
            if isinstance(m.get("content"), str) and m.get("role") != "system"
        )

        img_missing = len(images) - total_img
        vid_missing = len(videos) - total_vid
        aud_missing = len(audios) - total_aud

        # Find first user message index
        first_user_idx = next((i for i, m in enumerate(messages) if m.get("role") == "user"), None)

        new_messages = []
        img_iter, vid_iter, aud_iter = iter(images), iter(videos), iter(audios)

        for idx, message in enumerate(messages):
            message = copy(message)
            content = message.get("content", "")

            # Non-user messages: convert to list format but don't consume iterator
            if message.get("role") != "user":
                if isinstance(content, str):
                    message["content"] = [{"type": "text", "text": content}] if content else []
                new_messages.append(message)
                continue

            # User messages: skip non-string content
            if not isinstance(content, str):
                new_messages.append(message)
                continue

            # Prepend missing placeholders to first user message
            if idx == first_user_idx:
                if img_missing > 0:
                    content = image_placeholder * img_missing + content
                if vid_missing > 0:
                    content = video_placeholder * vid_missing + content
                if aud_missing > 0:
                    content = audio_placeholder * aud_missing + content

            # Count placeholders in this message and assign media
            msg_img_count = content.count(image_placeholder)
            msg_vid_count = content.count(video_placeholder)
            msg_aud_count = content.count(audio_placeholder)

            msg_images = [next(img_iter, None) for _ in range(msg_img_count)]
            msg_videos = [next(vid_iter, None) for _ in range(msg_vid_count)]
            msg_audios = [next(aud_iter, None) for _ in range(msg_aud_count)]

            message["content"] = content
            message["images"] = [i for i in msg_images if i is not None]
            message["videos"] = [v for v in msg_videos if v is not None]
            message["audios"] = [a for a in msg_audios if a is not None]

            message = transfer_to_standard_message(
                message, image_placeholder, video_placeholder, audio_placeholder, self.is_multimodal
            )

            new_messages.append(message)

        return new_messages

    def _build_standard_messages(self, trajectory: Trajectory) -> list[Trajectory]:
        # Extract trajectory-level media
        extracted_images = trajectory.pop("images", None) or [
            img for msg in trajectory["messages"] for img in msg.get("images", []) or []
        ]
        extracted_videos = trajectory.pop("videos", None) or [
            video for msg in trajectory["messages"] for video in msg.get("videos", []) or []
        ]
        extracted_audios = trajectory.pop("audios", None) or [
            audio for msg in trajectory["messages"] for audio in msg.get("audios", []) or []
        ]
        images = self.preprocess_images(extracted_images)
        videos = self.preprocess_videos(extracted_videos)
        audios = self.preprocess_audios(extracted_audios)

        trajectory["messages"] = self._process_mm_messages(trajectory["messages"], images, videos, audios)
        if not self.is_multimodal:
            for message in trajectory["messages"]:
                c = message.get("content")
                if isinstance(c, list):
                    message['content'] = ''.join(block.get('text', '') for block in c
                                                if isinstance(block, dict) and block.get('type') == 'text')
        return [trajectory]

    @staticmethod
    def decode_tool_calls(message: dict[str, Any]) -> dict[str, Any]:
        """Return ``message`` with ``tool_calls`` in the shape a chat template renders.

        OpenAI-shaped calls carry ``function.arguments`` as a JSON string, and an
        Arrow round-trip can turn the whole list into one; templates index them as
        objects. Arguments that will not parse become ``{}`` rather than reaching
        Jinja as a string it would render verbatim. The message is returned
        untouched when it carries no calls.
        """
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, str):
            tool_calls = json.loads(tool_calls) if tool_calls else []
        elif not tool_calls:
            return message
        decoded = []
        for tool_call in tool_calls:
            fn = tool_call["function"]
            args = fn["arguments"]
            if isinstance(args, dict):
                value = args
            elif isinstance(args, str):
                value = json.loads(args) if args.strip() else {}
            else:
                value = {}
            decoded.append({**tool_call, "function": {**fn, "arguments": value}})
        return {**message, "tool_calls": decoded}

    @staticmethod
    def _collapse_system_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge every system message into one leading message (blank-line-joined).

        The OpenAI protocol allows several system messages, but most model chat
        templates accept exactly one at the beginning -- Qwen3.5's even raises
        "System message must be at the beginning" on a second one, which the
        sampler surfaced as a silent empty stream. A lone non-leading system
        message moves up for the same reason. Non-system messages keep their
        relative order; content blocks (multimodal) concatenate as-is.
        """
        system_msgs = [m for m in messages if m.get("role") == "system"]
        if not system_msgs:
            return messages
        if len(system_msgs) == 1 and messages[0] is system_msgs[0]:
            return messages
        contents = [m.get("content") for m in system_msgs]
        if all(isinstance(c, str) for c in contents):
            merged_content = "\n\n".join(c for c in contents if c)
        else:
            blocks: list[dict[str, Any]] = []
            for c in contents:
                if isinstance(c, str):
                    blocks.append({"type": "text", "text": c})
                elif isinstance(c, list):
                    blocks.extend(b for b in c if isinstance(b, dict))
            merged_content = blocks
        merged = dict(system_msgs[0])
        merged["content"] = merged_content
        rest = [m for m in messages if m.get("role") != "system"]
        return [merged, *rest]

    def render_delta(
        self,
        messages_before: list[dict[str, Any]],
        appended: list[dict[str, Any]],
        *,
        gen_prompt_before: bool,
        gen_prompt_after: bool,
        tools: list[dict[str, Any]] | None = None,
        engine: bool = False,
        backend: str = "vllm",
    ) -> list[int]:
        check_backend(backend)
        messages_before = [self.decode_tool_calls(message) for message in messages_before]
        appended = [self.decode_tool_calls(message) for message in appended]

        def render(messages: list[dict[str, Any]], add_generation_prompt: bool) -> list[int]:
            messages = self._collapse_system_messages(messages)
            text = self.tokenizer.apply_chat_template(
                messages,
                tools=tools or None,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                preserve_thinking=True,
                **({'enable_thinking': self.enable_thinking} if self._explicit_enable_thinking else {}),
            )
            return self.tokenizer.encode(text, add_special_tokens=False)

        delta = render_delta_tokens(
            messages_before,
            appended,
            gen_prompt_before=gen_prompt_before,
            gen_prompt_after=gen_prompt_after,
            render=render,
        )
        return self._engine_input_ids(delta) if engine else delta

    def _prepare_chat_template(self, trajectory: Trajectory, **kwargs):
        if self._explicit_enable_thinking:
            kwargs.setdefault('enable_thinking', self.enable_thinking)
        if self.preserve_thinking is not None:
            kwargs.setdefault('preserve_thinking', self.preserve_thinking)
        messages = [dict(message) for message in trajectory["messages"]]
        messages = self._collapse_system_messages(messages)
        # Arrow serialization may pad content blocks with null keys (e.g. 'image': None
        # on text-only blocks). Jinja checks `'image' in item` on dict keys, so these
        # phantom keys cause wrong token counts. Strip them here.
        for msg in messages:
            if not isinstance(msg.get("content"), list):
                continue
            msg["content"] = [
                {k: v for k, v in b.items() if v is not None} for b in msg["content"] if isinstance(b, dict)
            ]
        messages = [self.decode_tool_calls(msg) for msg in messages]
        return messages, kwargs

    def _apply_chat_template(self, trajectory: Trajectory, add_generation_prompt: bool = False, **kwargs):
        messages, kwargs = self._prepare_chat_template(trajectory, **kwargs)
        # ``tool_calls`` / ``tools`` are already OpenAI-shaped (see
        # :mod:`twinkle.data_format.message`); pass them through verbatim.
        tools = list(trajectory.get("tools") or [])

        processor_kwargs = {**(trajectory.get('mm_processor_kwargs') or {}),
                            **(kwargs.pop('processor_kwargs', None) or {})}
        kwargs.setdefault('tokenize', True)
        kwargs.setdefault('return_dict', True)
        kwargs.setdefault('return_tensors', 'pt')
        kwargs.setdefault('padding', False)
        if 'processor_kwargs' in inspect.signature(self.processor.apply_chat_template).parameters:
            kwargs['processor_kwargs'] = processor_kwargs
        else:
            kwargs = {**processor_kwargs, **kwargs}
        return self.processor.apply_chat_template(
            messages, tools=tools, add_generation_prompt=add_generation_prompt, **kwargs)

    @staticmethod
    def _get_train_indices(trajectory: Trajectory) -> set[int] | None:
        """You can pick any round for training, only set key_rounds in `user_data`"""
        kr = user_data_get(trajectory.get("user_data"), "key_rounds")
        if isinstance(kr, list) and kr:
            return set(kr)
        return None

    def _encode_messages(self, trajectory: Trajectory, add_generation_prompt: bool = False,
                         *, with_labels: bool = True, **kwargs) -> InputFeature:
        """Encode a single trajectory's messages into InputFeature."""
        # Normalize multiple system messages into one leading message BEFORE labeling:
        # TokenizeByRound locates assistant boundaries by prefix-encode LENGTH -- if the
        # full encode merges a trailing system into the head while the prefix slices do
        # not contain it, every boundary shifts and labels silently mark system tokens
        # trainable. Collapsing here (idempotent) keeps full and prefix encodes on the
        # same message sequence; key_rounds indices count the normalized rounds. The
        # render entries collapse again for direct inference callers -- idempotent.
        if self.use_chat_template:
            trajectory['messages'], kwargs = self._prepare_chat_template(trajectory, **kwargs)
        else:
            trajectory['messages'] = self._collapse_system_messages(trajectory['messages'])
        labels = None
        input_ids = None
        # key-round selective training
        train_indices = self._get_train_indices(trajectory) if with_labels and not add_generation_prompt else None
        if self.use_chat_template:
            if add_generation_prompt or not with_labels:
                encoded = self._apply_chat_template(trajectory, add_generation_prompt=add_generation_prompt, **kwargs)
                if "input_ids" in encoded:
                    input_ids = encoded.pop("input_ids")
                    if hasattr(input_ids, "squeeze"):
                        input_ids = input_ids.squeeze(0)
                    labels = None
            elif train_indices is not None:
                # key-round-only: always use TokenizeByRound with filtered indices
                if kwargs.get("tokenize", True):
                    input_ids, labels, encoded = TokenizeByRound.tokenize_with_assistant_labels(
                        self.tokenizer, self._apply_chat_template, trajectory, train_indices=train_indices, **kwargs
                    )
                else:
                    encoded = self._apply_chat_template(trajectory, **kwargs)
            elif self._template_support_assistant_tokens_mask:
                encoded = self._apply_chat_template(
                    trajectory, return_assistant_tokens_mask=kwargs.get("tokenize", True), **kwargs
                )
                if "input_ids" in encoded:
                    input_ids = encoded.pop("input_ids")
                    assistant_masks = encoded.pop("assistant_masks")
                    # _apply_chat_template returns batched tensors ([1, T]); strip the batch dim
                    # so downstream `len(input_ids)` reflects sequence length, not 1.
                    if hasattr(input_ids, "squeeze"):
                        input_ids = input_ids.squeeze(0)
                    if hasattr(assistant_masks, "squeeze"):
                        assistant_masks = assistant_masks.squeeze(0)
                    labels = np.where(assistant_masks, input_ids, -100)
            else:
                if kwargs.get("tokenize", True):
                    input_ids, labels, encoded = TokenizeByRound.tokenize_with_assistant_labels(
                        self.tokenizer, self._apply_chat_template, trajectory, **kwargs
                    )
                else:
                    encoded = self._apply_chat_template(trajectory, **kwargs)
        else:
            assert len(trajectory["messages"]) == 1 and trajectory["messages"][0]["role"] == "user"
            text = trajectory["messages"][0]["content"]
            if isinstance(text, list):
                # The multimodal pre-pipeline (_build_standard_messages) standardizes content
                # into blocks and only collapses it back to str for non-multimodal templates,
                # so a multimodal-family checkpoint reaches this branch with a block list.
                # A raw tokenizer consumes text only: join the text blocks and fail loudly on
                # media, which the non-chat path cannot encode.
                media_types = [block.get("type") for block in text if block.get("type") != "text"]
                if media_types:
                    raise ValueError(f"use_chat_template=False cannot encode media content blocks: {media_types}. "
                                     "Use a chat template to encode multimodal inputs.")
                text = "".join(block.get("text") or "" for block in text)
            if kwargs.pop('tokenize', True):
                input_ids = self.tokenizer.encode(text, **kwargs)
                encoded = {}
                labels = [-100, *input_ids[1:]] if input_ids else []
            else:
                if kwargs:
                    raise TypeError(f'Unsupported un-tokenized encoding kwargs: {sorted(kwargs)}')
                encoded = {'prompt': text}
        if isinstance(encoded, str):
            input_feature = {'prompt': encoded}
        elif input_ids is None:
            input_feature = dict(encoded)
        else:
            input_feature = InputFeature(input_ids=np.array(input_ids), **encoded)
            if labels is not None and with_labels:
                input_feature['labels'] = np.array(labels)
        for key in ('token_type_ids', 'mm_token_type_ids'):
            value = input_feature.get(key)
            if value is not None and getattr(value, 'ndim', 0) == 2 and value.shape[0] == 1:
                input_feature[key] = value[0]
        if self.task_type == 'causal_lm' and input_feature.get('labels') is not None and len(input_feature['labels']):
            input_feature['labels'][0] = -100
        return input_feature

    def encode(
        self, sample=None, add_generation_prompt=None, *, intent=None,
        engine='vllm', return_length=False, **kwargs,
    ):
        sample = self._encode_argument(sample, kwargs)
        check_backend(engine)
        explicit = intent is not None or add_generation_prompt is not None
        intent = resolve_intent(intent, add_generation_prompt)
        from .contract import KTOSample, PreferenceSample
        if isinstance(sample, PreferenceSample):
            if not intent.with_labels:
                raise ValueError(f'A preference sample encodes only under a labelled intent, got {intent}.')
            chosen = copy_trajectory(sample.chosen)
            if sample.margin is not None:
                chosen['margin'] = sample.margin
            sample = {'chosen': chosen, 'rejected': sample.rejected,
                      **({'margin': float(sample.margin)} if sample.margin is not None else {})}
        elif isinstance(sample, KTOSample):
            if not intent.with_labels:
                raise ValueError(f'A preference sample encodes only under a labelled intent, got {intent}.')
            sample = {'chosen': {**sample.conversation, 'label': sample.label},
                      'rejected': sample.rejected, 'label': bool(sample.label)}
        nested = self._get_trajectory_keys(sample, False) if isinstance(sample, Mapping) else []
        if nested:
            if self.truncation_strategy == 'split':
                raise ValueError('preference encoding does not support truncation_strategy="split".')
            result = copy_trajectory(sample)
            for key in nested:
                result[key] = self.encode(sample[key], add_generation_prompt, intent=intent,
                                          engine=engine, return_length=return_length, **kwargs)
                if not isinstance(result[key], Mapping):
                    raise ValueError('paired trajectory encoding requires exactly one result per side')
            if return_length:
                result['lengths'] = [result[key]['length'] for key in nested if 'length' in result[key]]
            return result
        if explicit and self.use_chat_template and (intent.with_labels or intent.add_generation_prompt):
            check_generation_prompt(sample, intent.add_generation_prompt)
        output = self._encode_one(sample, intent, engine=engine, return_length=return_length, **kwargs)
        return output[0] if len(output) == 1 else output

    @staticmethod
    def _encode_argument(sample, kwargs):
        if 'trajectory' in kwargs:
            if sample is not None:
                raise TypeError('Pass sample or trajectory, not both')
            sample = kwargs.pop('trajectory')
        if sample is None:
            raise TypeError('encode requires a sample')
        return {'messages': sample} if isinstance(sample, list) else sample

    def _encode_one(self, sample, intent, *, engine='vllm', return_length=False, **kwargs):
        source = copy_trajectory(sample)
        output = []
        options = {**(source.get('chat_template_kwargs') or {}), **kwargs}
        if self.use_chat_template:
            options.setdefault('preserve_thinking', self.preserve_thinking
                               if self.preserve_thinking is not None else intent.with_labels)
        for rendered in self._invoke_pre_pipeline([copy_trajectory(source)]):
            encoded = self._encode_messages(rendered, add_generation_prompt=intent.add_generation_prompt,
                                            with_labels=intent.with_labels, **options)
            original_ids = list(encoded.get('input_ids', ()))
            if not intent.with_labels:
                for key in SHIFTED_FIELDS:
                    encoded.pop(key, None)
            span_cache = {}
            for feature in self._run_post_pipeline([encoded]):
                result = preserve_trajectory(source, feature)
                if not intent.with_labels:
                    for key in SHIFTED_FIELDS:
                        result.pop(key, None)
                ids = result.get('input_ids')
                if ids is not None:
                    kept = result.pop('_kept_token_indices', None)
                    result.pop('_original_input_ids', None)
                    if kept is not None:
                        def render_prefix(messages):
                            row = copy_trajectory(source)
                            row['messages'] = messages
                            renderer = self.replace(max_length=None)
                            row = renderer._invoke_pre_pipeline([row])[0]
                            return renderer._encode_messages(row, add_generation_prompt=False, **options)['input_ids']
                        result['messages'] = project_sliced_messages(
                            source.get('messages') or [], original_ids, kept, render_prefix=render_prefix,
                            span_cache=span_cache,
                            decode=lambda values: self.tokenizer.decode(values, skip_special_tokens=True))
                    engine_ids = list(self._engine_ids_tuple(self._engine_input_ids(ids)))
                    if not intent.expand_media:
                        result['input_ids'] = engine_ids
                        result = {key: value for key, value in result.items() if key not in self.media_keys(result)}
                        for key in ('position_ids', 'text_position_ids'):
                            result.pop(key, None)
                        result['attention_mask'] = [1] * len(engine_ids)
                        result['position_ids'] = list(range(len(engine_ids)))
                        ids = engine_ids
                    if feature_has_media(result) or engine_ids != list(ids):
                        result['engine_input_ids'] = engine_ids
                    else:
                        result.pop('engine_input_ids', None)
                    result['length'] = len(ids)
                    if return_length:
                        result['lengths'] = [len(ids)]
                    if not intent.with_labels:
                        result['reasoning_started'] = self._reasoning_started(ids)
                output.append(normalize_token_fields(result))
        return output

    @staticmethod
    def map_col_to_row(trajectories: dict[str, Any]):
        if not trajectories:
            return []
        rows = []
        total_count = len(next(iter(trajectories.values())))
        if any(len(values) != total_count for values in trajectories.values()):
            raise ValueError('columnar trajectories must have equally sized columns')
        for i in range(total_count):
            row = {}
            for key in trajectories:
                row[key] = trajectories[key][i]
            rows.append(row)
        return rows

    @staticmethod
    def map_row_to_col(rows: list[dict[str, Any] | InputFeature]) -> dict[str, list[Any]]:
        if not rows:
            return {}

        columns: dict[str, list[Any]] = {}
        keys = dict.fromkeys(key for row in rows for key in row)

        for key in keys:
            columns[key] = [row.get(key) for row in rows]

        return columns

    def _is_trajectory(self, obj: Any) -> bool:
        """Check if an object is a Trajectory (has 'messages' key)."""
        return isinstance(obj, Mapping) and "messages" in obj

    def _get_trajectory_keys(self, trajectories: Mapping, is_columnar: bool) -> list[str]:
        """Get keys whose values are lists of Trajectories."""
        if self._is_trajectory(trajectories):
            return []
        keys = []
        if is_columnar:
            for k, v in trajectories.items():
                if isinstance(v, list) and v and self._is_trajectory(v[0]):
                    keys.append(k)
        else:
            for k, v in trajectories.items():
                if v is not None and self._is_trajectory(v):
                    keys.append(k)
        return keys

    def batch_encode(
        self, trajectories, add_generation_prompt=None, *, intent=None,
        engine='vllm', return_length=False, **kwargs,
    ):
        check_backend(engine)
        resolve_intent(intent, add_generation_prompt)
        columnar = isinstance(trajectories, Mapping)
        rows = self.map_col_to_row(trajectories) if columnar else trajectories
        output = []
        for row in rows:
            try:
                result = self.encode(row, add_generation_prompt, intent=intent,
                                     engine=engine, return_length=return_length, **kwargs)
                output.extend(result if isinstance(result, list) else [result])
            except MaxLengthError:
                if self.truncation_strategy != 'delete':
                    raise
        if columnar:
            if not output:
                return {key: [] for key in trajectories}
            return self.map_row_to_col(output)
        return output

    def format_trajectory(self, trajectory: Trajectory, add_default_system: bool = False) -> Trajectory:
        skip = set() if add_default_system else {"_add_default_system"}
        return self._invoke_pre_pipeline([copy_trajectory(trajectory)], skip_stages=skip)[0]

    def check(self, trajectory: Trajectory) -> Trajectory | None:
        encoded = None
        try:
            encoded = self.encode(trajectory)
            if not encoded:
                return None
            else:
                return trajectory
        except Exception as e:
            logger.warning(f'Error encoding trajectory: {e}')
            return None
        finally:
            if encoded:
                del encoded

    def batch_check(self, trajectories: list[Trajectory]) -> list[Trajectory | None]:
        output = []
        for trajectory in trajectories:
            output.append(self.check(trajectory))
        return output

    def encode_prompt(
        self, sample: Trajectory, *, backend: str = 'vllm', add_generation_prompt: bool = True
    ) -> Trajectory:
        intent = EncodeIntent(add_generation_prompt, False, True)
        feature = self.encode(sample, intent=intent, engine=backend)
        if not isinstance(feature, dict):
            raise ValueError('encode_prompt requires one unsplit prompt feature')
        if not add_generation_prompt:
            feature['reasoning_started'] = self._reasoning_started(feature['input_ids'])
        return feature

    def append_completion(
        self, prompt: Trajectory, tokens: list[int], *, appended_as: str = 'completion',
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> Trajectory:
        prefix = copy_trajectory(prompt)
        ids = engine_input_ids(prefix)
        media = MediaBundle.from_sources(prefix)
        if feature_has_media(prefix) and not media:
            raise ValueError('multimodal continuation requires original media')
        feature = self._append_feature(prefix, tokens, appended_as=appended_as, tool_calls=tool_calls)
        if feature is None:
            raise MaxLengthError('appended completion exceeds max_length under delete truncation strategy')
        response = self.decode_message(
            tokens, parse_tool_calls=tool_calls is None,
            reasoning_started=(prefix['reasoning_started'] if 'reasoning_started' in prefix
                               else self._reasoning_started(prefix['input_ids'])),
            tools=list(prefix.get('tools') or ()), structured_calls=tool_calls,
        )
        message = Message(role='assistant', content=response.content)
        if response.reasoning_content is not None:
            message['reasoning_content'] = response.reasoning_content
        if response.tool_calls:
            message['tool_calls'] = list(response.tool_calls)
        feature['messages'] = list(prefix.get('messages') or ()) + [message]
        feature['tools'] = list(prefix.get('tools') or ())
        if 'engine_input_ids' in prefix:
            feature['engine_input_ids'] = ids + list(tokens)
        else:
            feature.pop('engine_input_ids', None)
        feature['reasoning_started'] = bool(response.reasoning_started)
        return feature

    def _reasoning_started(self, input_ids) -> bool:
        return _reasoning_is_open(input_ids, self.tokenizer, self.reasoning_open_marker, self.reasoning_close_marker)

    def decode_message(
        self, token_ids: list[int], *, context: DecodeContext | None = None, structured_calls=None, **kwargs
    ) -> DecodedResponse:
        if context is None:
            context = DecodeContext(
                parse_tool_calls=kwargs.pop("parse_tool_calls", True),
                reasoning_started=kwargs.pop("reasoning_started", None),
                tools=tuple(kwargs.pop("tools", ()) or ()),
                skip_special_tokens=kwargs.pop("skip_special_tokens", True),
            )
        if kwargs:
            raise TypeError(f'Unsupported decode_message kwargs: {sorted(kwargs)}')
        decoded = self.decode(token_ids, skip_special_tokens=False)
        reasoning_started = self.enable_thinking if context.reasoning_started is None else context.reasoning_started
        return decode_output(
            decoded,
            thinking_marks=(self.reasoning_open_marker, self.reasoning_close_marker),
            reasoning_started=reasoning_started,
            tool_parser=_RegistryParser(self) if context.parse_tool_calls and structured_calls is None else None,
            tools=context.tools,
            special_tokens=tuple(getattr(self.tokenizer, "all_special_tokens", ()))
            if context.skip_special_tokens
            else (),
            structured_calls=structured_calls,
        )

    def decode_response(
        self, token_ids: list[int], *, context: DecodeContext | None = None, **kwargs
    ) -> DecodedResponse:
        return self.decode_message(token_ids, context=context or DecodeContext(first_token=True), **kwargs)

    def decode(self, token_ids, *, context=None, **kwargs):
        if context is not None:
            if kwargs:
                raise TypeError(f'Unsupported decode_message kwargs: {sorted(kwargs)}')
            return self.decode_message(token_ids, context=context)
        return self.tokenizer.decode(token_ids, **kwargs)

    def batch_decode(self, token_ids: list[list[int]], **kwargs) -> list[str]:
        return [self.decode(_ids, **kwargs) for _ids in token_ids]

    def prepare_row(self, row):
        if row.get('input_ids') is None or row.get('position_ids') is not None:
            return row
        import torch
        inputs = dict(row)
        for key in ('input_ids', 'attention_mask'):
            if inputs.get(key) is not None:
                inputs[key] = torch.as_tensor(inputs[key])
        positions = self.set_mm_position_ids(inputs)
        row['position_ids'] = positions.tolist() if getattr(positions, 'ndim', 0) == 1 else positions
        return row

    @staticmethod
    def media_keys(feature):
        from .contract.encoding import model_payload_keys
        return model_payload_keys(feature)

    def pop_media(self, row):
        return {key: row.pop(key) for key in self.media_keys(row)}

    def collate_media(self, rows):
        import torch
        result = {}
        for key in dict.fromkeys(key for row in rows for key in row):
            values = [torch.as_tensor(row[key]) for row in rows if row.get(key) is not None]
            if values:
                result[key] = torch.stack(values) if values[0].ndim == 0 else torch.cat(values, dim=0)
        return result

    def finalize_batch(self, inputs):
        return inputs

    def bind_model(self, model):
        self.model = model

    def replace(self, **overrides):
        supported = {'max_length', 'truncation_strategy'}
        unknown = set(overrides) - supported
        if unknown:
            raise TypeError(f'Template.replace does not support {sorted(unknown)}; supported: {sorted(supported)}')
        derived = copy(self)
        for key, value in overrides.items():
            setattr(derived, key, value)
        return derived

    def _get_vision_token_id(self) -> int | None:
        if self.config is not None:
            return getattr(self.config, "image_token_id", None)
        else:
            return self.processor.encode(self.image_placeholder)

    def _post_encode(self, model: "torch.nn.Module", inputs: dict[str, Any]) -> dict[str, Any]:
        return inputs

    def pre_forward_hook(self, model: "torch.nn.Module", args, kwargs):
        if not self.is_multimodal:
            return args, kwargs
        device = next(model.parameters()).device
        old_kwargs = to_device(kwargs, device)
        kwargs = to_device(self._post_encode(model, old_kwargs), device)
        for k, v in old_kwargs.items():
            if (
                k
                in {
                    "input_ids",
                    "attention_mask",
                    "labels",
                    "position_ids",
                    "output_hidden_states",
                    "logits_to_keep",
                    "max_length_q",
                    "max_length_k",
                    "cu_seq_lens_q",
                    "cu_seq_lens_k",
                    "cu_seqlens_q",
                    "cu_seqlens_kv",
                    "packed_seq_params",
                }
                and k not in kwargs
            ):
                kwargs[k] = v
        if "inputs_embeds" in kwargs:
            kwargs.pop("input_ids", None)

        from peft import PeftModel

        if isinstance(model, PeftModel):
            base_model = model.model
        else:
            base_model = model
        parameters = inspect.signature(base_model.forward).parameters
        if "position_ids" not in parameters:
            kwargs.pop("position_ids", None)
        return args, kwargs
