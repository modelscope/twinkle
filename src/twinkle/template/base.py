# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import os
from collections.abc import Callable, Mapping
from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import json
import numpy as np
from swift.dev.template_contract.encoding import render_delta as render_delta_tokens
from swift.dev.template_contract.output import _reasoning_is_open, decode_output

from twinkle import remote_class
from twinkle.data_format import InputFeature, Message, Trajectory, user_data_get
from twinkle.hub import HubOperation
from twinkle.utils import load_image, to_device
from .contract import (
    CompletionEncoding,
    DecodeContext,
    DecodedResponse,
    MediaBundle,
    PromptEncoding,
    RolloutState,
    append_and_finalize,
    feature_has_media,
    shift_values,
    split_reasoning,
)
from .tools import ToolCallRegistry
from .utils import TokenizeByRound, transfer_to_standard_message

if TYPE_CHECKING:
    import torch
    from PIL import Image

# Type aliases for multimodal data
ImageInput = Union[str, "Image.Image", "torch.Tensor"]
VideoInput = Union[str, list["Image.Image"], "torch.Tensor"]
AudioInput = Union[str, np.ndarray, "torch.Tensor"]

# Fields that are one entry per token and must be sliced with ``input_ids``.
# ``mm_token_type_ids`` is excluded: it may carry a leading batch dim and is
# sliced on its last axis instead.
_SEQUENCE_ALIGNED_FIELDS = ("labels", "completion_mask")

# What an appended turn is to a trainer: the policy's own completion (scored, and
# a log-prob exists for each of its tokens), someone else's completion offered
# for imitation (scored, no log-prob -- usable by SFT but not by RL), or history
# that no loss may touch. There is deliberately no fourth role: a log-prob is
# only ever needed for a token that is also scored.
_APPEND_ROLES = ("completion", "demonstration", "context")


class _RegistryParser:
    @staticmethod
    def _parser(text):
        return ToolCallRegistry.detect_first(text or "")

    def parse(self, text):
        parser = self._parser(text)
        return parser.parse(text) if parser else []

    def parse_errors(self, text):
        parser = self._parser(text)
        return parser.parse_errors(text) if parser else []

    def clean(self, text):
        parser = self._parser(text)
        return parser.clean(text) if parser else (text or "").rstrip()


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

    def __init__(
        self,
        model_id: str,
        use_chat_template: bool = True,
        max_length: int | None = 8192,
        truncation_strategy: Literal["raise", "left", "right", "split", "delete"] = "raise",
        default_system: str | None = None,
        enable_thinking: bool = True,
        media_types: set[Literal["image", "video", "audio"]] | None = None,
        **kwargs,
    ):
        self.model_id = model_id
        model_id = HubOperation.download_model(model_id, ignore_model=True)
        if os.path.exists(os.path.join(model_id, "preprocessor_config.json")) or os.path.exists(
            os.path.join(model_id, "processor_config.json")
        ):
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(model_id, **kwargs)
        else:
            from transformers import AutoTokenizer

            self.processor = AutoTokenizer.from_pretrained(model_id, **kwargs)
        from transformers import AutoConfig

        self.config = AutoConfig.from_pretrained(model_id, **kwargs)
        self.media_types = frozenset(media_types) if media_types is not None else self._infer_media_types()

        self.use_chat_template = use_chat_template
        self.max_length = max_length
        self.enable_thinking = enable_thinking
        self.truncation_strategy = truncation_strategy
        self.default_system = default_system
        self._test_support_assistant_tokens_mask()

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
        return self._run_post_pipeline(input_features)

    def _append_feature(
        self,
        prompt_input_feature: InputFeature,
        new_tokens: list[int],
        *,
        appended_as: Literal["completion", "demonstration", "context"] = "completion",
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> InputFeature | None:
        if appended_as not in _APPEND_ROLES:
            raise ValueError(f"appended_as must be one of {_APPEND_ROLES}, got {appended_as!r}")
        return append_and_finalize(
            prompt_input_feature,
            new_tokens,
            scored=appended_as != "context",
            sampled=appended_as == "completion",
            finalize=self.finalize_appended_features,
        )

    def _add_default_system(self, trajectory: Trajectory) -> list[Trajectory]:
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

    def _truncate_feature(self, feature: InputFeature, strategy: str) -> InputFeature:
        """Truncate the sequence-aligned fields of a single InputFeature."""
        length = len(feature["input_ids"])
        if length <= self.max_length:
            return feature
        if strategy == "raise":
            raise ValueError(f"Input length {length} exceeds max_length {self.max_length}")
        if strategy == "left":
            keep = slice(-self.max_length, None)
        elif strategy == "right":
            keep = slice(None, self.max_length)
        else:
            raise ValueError(f"Unsupported truncation_strategy={strategy!r}.")
        result = dict(feature)
        result["input_ids"] = result["input_ids"][keep]
        for key in _SEQUENCE_ALIGNED_FIELDS:
            if key in result:
                result[key] = result[key][keep]
        if "mm_token_type_ids" in result:
            result["mm_token_type_ids"] = result["mm_token_type_ids"][..., keep]
        return InputFeature(**result)

    def set_mm_position_ids(self, input_feature: InputFeature):
        return np.arange(len(input_feature["input_ids"]))

    def _engine_input_ids(self, input_ids):
        return input_ids

    @staticmethod
    def _engine_ids_tuple(input_ids) -> tuple[int, ...]:
        """``input_ids`` as a tuple of plain python ints, for :attr:`RolloutState.engine_input_ids`.

        ``_encode_messages`` materialises ``input_ids`` as a numpy array, so a plain
        ``tuple(input_ids)`` yields numpy scalars. ``engine_input_ids`` flows into the serving
        engines' request payloads (vLLM ``TokensPrompt``), whose msgpack encoder refuses
        ``numpy.int64`` -- a generation would die at request encoding with "Object of type
        numpy.int64 is not serializable". Normalising here covers both construction sites
        (``encode_prompt`` and the plain-feature branch of ``append_completion``).
        """
        return tuple(int(token) for token in input_ids)

    def _check_max_length(self, input_feature: InputFeature) -> list[InputFeature]:
        if not self.max_length or "input_ids" not in input_feature:
            return [input_feature]

        strategy = self.truncation_strategy

        # Split strategy
        if strategy == "split":
            if self.is_multimodal:
                raise ValueError(
                    "truncation_strategy='split' is unsafe for multimodal templates: "
                    "splitting input_ids across chunks breaks alignment with image tokens, "
                    "and multimodal fields (pixel_values, image_grid_thw, ...) are not partitioned. "
                    "Use 'left' / 'right' / 'delete' / 'raise' instead."
                )
            results = []
            for start in range(0, len(input_feature["input_ids"]), self.max_length):
                end = min(start + self.max_length, len(input_feature["input_ids"]))
                feat = dict(input_feature)
                feat["input_ids"] = feat["input_ids"][start:end]
                for key in _SEQUENCE_ALIGNED_FIELDS:
                    if key in feat:
                        feat[key] = feat[key][start:end]
                if "mm_token_type_ids" in feat:
                    feat["mm_token_type_ids"] = feat["mm_token_type_ids"][..., start:end]
                results.append(InputFeature(**feat))
            return results

        # Drop oversized samples entirely; downstream must tolerate empty list (sample skipped).
        if strategy == "delete":
            if len(input_feature["input_ids"]) > self.max_length:
                return []
            return [input_feature]

        # left/right/raise
        return [self._truncate_feature(input_feature, strategy)]

    def _add_attention_fields(self, input_feature: InputFeature) -> list[InputFeature]:
        if "input_ids" not in input_feature:
            return [input_feature]
        input_ids = input_feature["input_ids"]
        input_feature["attention_mask"] = np.ones_like(input_ids)
        input_feature["position_ids"] = self.set_mm_position_ids(input_feature)
        input_feature["length"] = len(input_ids)
        return [input_feature]

    def _roll_labels(self, input_feature: InputFeature) -> list[InputFeature]:
        if "input_ids" not in input_feature:
            return [input_feature]
        input_feature["labels"] = shift_values(input_feature["labels"], -100)
        if "completion_mask" in input_feature:
            input_feature["completion_mask"] = shift_values(input_feature["completion_mask"], 0)
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
        # Count total placeholders across all messages
        total_img = sum(
            m.get("content", "").count(self.image_placeholder)
            for m in messages
            if isinstance(m.get("content"), str) and m.get("role") != "system"
        )
        total_vid = sum(
            m.get("content", "").count(self.video_placeholder)
            for m in messages
            if isinstance(m.get("content"), str) and m.get("role") != "system"
        )
        total_aud = sum(
            m.get("content", "").count(self.audio_placeholder)
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
                    content = self.image_placeholder * img_missing + content
                if vid_missing > 0:
                    content = self.video_placeholder * vid_missing + content
                if aud_missing > 0:
                    content = self.audio_placeholder * aud_missing + content

            # Count placeholders in this message and assign media
            msg_img_count = content.count(self.image_placeholder)
            msg_vid_count = content.count(self.video_placeholder)
            msg_aud_count = content.count(self.audio_placeholder)

            msg_images = [next(img_iter, None) for _ in range(msg_img_count)]
            msg_videos = [next(vid_iter, None) for _ in range(msg_vid_count)]
            msg_audios = [next(aud_iter, None) for _ in range(msg_aud_count)]

            message["content"] = content
            message["images"] = [i for i in msg_images if i is not None]
            message["videos"] = [v for v in msg_videos if v is not None]
            message["audios"] = [a for a in msg_audios if a is not None]

            message = transfer_to_standard_message(
                message, self.image_placeholder, self.video_placeholder, self.audio_placeholder, self.is_multimodal
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
                    message["content"] = c[0]["text"] if c else ""
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
        messages_before = [self.decode_tool_calls(message) for message in messages_before]
        appended = [self.decode_tool_calls(message) for message in appended]

        def render(messages: list[dict[str, Any]], add_generation_prompt: bool) -> list[int]:
            text = self.tokenizer.apply_chat_template(
                messages,
                tools=tools or None,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=self.enable_thinking,
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

    def _apply_chat_template(self, trajectory: Trajectory, add_generation_prompt: bool = False, **kwargs):
        messages = [dict(message) for message in trajectory["messages"]]
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
        # ``tool_calls`` / ``tools`` are already OpenAI-shaped (see
        # :mod:`twinkle.data_format.message`); pass them through verbatim.
        tools = list(trajectory.get("tools") or [])

        # Use inspect to get apply_chat_template signature params
        sig = inspect.signature(self.processor.apply_chat_template)
        supported_params = set(sig.parameters.keys())

        # Check if processor_kwargs is supported
        if "processor_kwargs" in supported_params:
            # Separate supported params from processor_kwargs
            apply_chat_template_kwargs = {}
            processor_kwargs = {}

            for key in list(kwargs.keys()):
                if key in supported_params:
                    apply_chat_template_kwargs[key] = kwargs.pop(key)

            # tokenize is in apply_chat_template_kwargs, set default value
            if "tokenize" not in apply_chat_template_kwargs:
                apply_chat_template_kwargs["tokenize"] = True

            # Set default values for processor_kwargs
            if "padding" not in kwargs:
                processor_kwargs["padding"] = False

            # Add remaining kwargs to processor_kwargs
            processor_kwargs.update(kwargs)

            inputs = self.processor.apply_chat_template(
                messages,
                tools=tools,
                return_dict=True,
                add_generation_prompt=add_generation_prompt,
                return_tensors="pt",
                processor_kwargs=processor_kwargs,
                enable_thinking=self.enable_thinking,
                **apply_chat_template_kwargs,
            )
        else:
            # No processor_kwargs support, pass all kwargs directly
            if "tokenize" not in kwargs:
                kwargs["tokenize"] = True
            if "enable_thinking" not in kwargs:
                kwargs["enable_thinking"] = self.enable_thinking

            inputs = self.processor.apply_chat_template(
                messages,
                tools=tools,
                padding=False,
                return_dict=True,
                add_generation_prompt=add_generation_prompt,
                return_tensors="pt",
                **kwargs,
            )
        return inputs

    @staticmethod
    def _get_train_indices(trajectory: Trajectory) -> set[int] | None:
        """You can pick any round for training, only set key_rounds in `user_data`"""
        kr = user_data_get(trajectory.get("user_data"), "key_rounds")
        if isinstance(kr, list) and kr:
            return set(kr)
        return None

    def _encode_messages(self, trajectory: Trajectory, add_generation_prompt: bool = False, **kwargs) -> InputFeature:
        """Encode a single trajectory's messages into InputFeature."""
        labels = None
        input_ids = None
        # key-round selective training
        train_indices = self._get_train_indices(trajectory) if not add_generation_prompt else None
        if self.use_chat_template:
            if add_generation_prompt:
                # For inference: just get input_ids with generation prompt, no labels needed
                encoded = self._apply_chat_template(trajectory, add_generation_prompt=True, **kwargs)
                if "input_ids" in encoded:
                    input_ids = encoded.pop("input_ids")
                    if hasattr(input_ids, "squeeze"):
                        input_ids = input_ids.squeeze(0)
                    labels = np.full_like(input_ids, -100)  # No labels for inference
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
            input_ids = self.tokenizer.encode(text, **kwargs)
            encoded = {}
            labels = deepcopy(input_ids)
        if isinstance(encoded, str):
            input_feature = {"prompt": encoded}
        else:
            input_feature = InputFeature(
                input_ids=np.array(input_ids),
                labels=np.array(labels),
                **encoded,
            )
        trajectory.update(input_feature)
        return trajectory

    def encode(self, trajectory: Trajectory, add_generation_prompt: bool = False, **kwargs) -> InputFeature:
        """Encode a single trajectory into an InputFeature.

        This is a convenience wrapper around :meth:`batch_encode` for encoding
        a single trajectory.

        Args:
            trajectory: The trajectory to encode.
            add_generation_prompt: Whether to add generation prompt.

        Returns:
            The encoded InputFeature.
        """
        assert self.truncation_strategy != "split", (
            'encode() does not support truncation_strategy=="split" because it may produce multiple outputs. '
            "Use batch_encode() instead."
        )
        encoded = self.batch_encode([trajectory], add_generation_prompt=add_generation_prompt, **kwargs)
        if encoded:
            return encoded[0]
        return None

    @staticmethod
    def map_col_to_row(trajectories: dict[str, Any]):
        if not trajectories:
            return []
        rows = []
        total_count = len(trajectories[next(iter(list(trajectories.keys())))])
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
        keys = rows[0].keys()

        for key in keys:
            columns[key] = [row[key] for row in rows]

        return columns

    def _is_trajectory(self, obj: Any) -> bool:
        """Check if an object is a Trajectory (has 'messages' key)."""
        return isinstance(obj, Mapping) and "messages" in obj

    def _get_trajectory_keys(self, trajectories: Mapping, is_columnar: bool) -> list[str]:
        """Get keys whose values are lists of Trajectories."""
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
        self,
        trajectories: dict[str, list[Any]] | list[Trajectory],
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> dict[str, Any] | list[InputFeature]:
        """Encode trajectories into InputFeatures.

        Args:
            trajectories: Either List[Trajectory] or columnar Dict[str, List].
                For nested trajectories, columnar format with trajectory list columns
                (e.g., 'chosen'/'rejected') is supported.
            add_generation_prompt: Whether to add generation prompt.

        Returns:
            List[InputFeature] or columnar Dict[str, List[InputFeature]].
        """
        _transfer = False

        # Handle list input
        if isinstance(trajectories, list) and len(trajectories) > 0:
            # Check if first element has nested trajectories
            if isinstance(trajectories[0], Mapping) and len(self._get_trajectory_keys(trajectories[0], False)) > 0:
                # Convert row→columnar, process with columnar logic, convert back
                columnar = self.map_row_to_col(trajectories)
                encoded = self.batch_encode(columnar, add_generation_prompt, **kwargs)
                return self.map_col_to_row(encoded)

        if isinstance(trajectories, Mapping):
            _transfer = True
            # Check if it has nested trajectory columns
            traj_keys = self._get_trajectory_keys(trajectories, True)
            if traj_keys:
                # Nested format: encode each trajectory list separately, keep other columns
                return {
                    key: self.batch_encode(trajectories[key], add_generation_prompt, **kwargs)
                    if key in traj_keys
                    else trajectories[key]
                    for key in trajectories
                }
            else:
                # Standard columnar format
                trajectories = self.map_col_to_row(trajectories)

        # Process List[Trajectory]
        trajectories = self._invoke_pre_pipeline(trajectories)
        output = [self._encode_messages(t, add_generation_prompt=add_generation_prompt, **kwargs) for t in trajectories]
        output = self._run_post_pipeline(output)

        if _transfer:
            output = self.map_row_to_col(output)
        return output

    def format_trajectory(self, trajectory: Trajectory, add_default_system: bool = False) -> Trajectory:
        skip = set() if add_default_system else {"_add_default_system"}
        return self._invoke_pre_pipeline([trajectory], skip_stages=skip)[0]

    def check(self, trajectory: Trajectory) -> Trajectory | None:
        encoded = None
        try:
            encoded = self.encode(trajectory)
            if not encoded:
                return None
            else:
                return trajectory
        except Exception as e:
            import traceback

            print(f"[Template.check] Error encoding trajectory: {e}")  # noqa: T201
            traceback.print_exc()
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
        self, sample: Trajectory, *, backend: str = "vllm", add_generation_prompt: bool = True
    ) -> PromptEncoding:
        media = MediaBundle.from_sources(sample)
        feature = self.encode(sample, add_generation_prompt=add_generation_prompt)
        if feature is None:
            # ``truncation_strategy='delete'`` drops over-long rows dataset-side; a single prompt
            # (rollout generation) has no row to drop, so say so instead of crashing on None below.
            raise ValueError(
                'the prompt exceeds max_length and truncation_strategy=\'delete\' cannot encode a '
                'single prompt; use a truncation strategy that shortens instead of dropping')
        feature.pop("messages", None)
        feature.pop("tools", None)
        engine_ids = self._engine_input_ids(feature["input_ids"])
        media = media or MediaBundle.from_sources(feature)
        state = RolloutState(
            feature=feature,
            engine_input_ids=self._engine_ids_tuple(engine_ids),
            messages=tuple(deepcopy(sample.get("messages") or ())),
            tools=tuple(deepcopy(sample.get("tools") or ())),
            reasoning_started=self._reasoning_started(feature["input_ids"]),
            media=media,
        )
        return PromptEncoding(state)

    def append_completion(
        self,
        prompt: InputFeature | RolloutState,
        tokens: list[int],
        *,
        appended_as: str = "completion",
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> CompletionEncoding:
        state = prompt if isinstance(prompt, RolloutState) else None
        if state is None:
            feature = dict(prompt)
            messages = tuple(deepcopy(feature.pop("messages", ())))
            tools = tuple(deepcopy(feature.pop("tools", ())))
            if feature_has_media(prompt):
                raise NotImplementedError("multimodal continuation requires RolloutState.")
            state = RolloutState(
                feature=feature,
                engine_input_ids=self._engine_ids_tuple(feature["input_ids"]),
                messages=messages,
                tools=tools,
                reasoning_started=self._reasoning_started(feature["input_ids"]),
                media=MediaBundle(),
            )
        prompt_feature = state.feature
        feature = self._append_feature(prompt_feature, tokens, appended_as=appended_as, tool_calls=tool_calls)
        if feature is None:
            raise ValueError("appended completion exceeds max_length under delete truncation strategy.")
        response = self.decode_message(
            tokens,
            parse_tool_calls=tool_calls is None,
            reasoning_started=state.reasoning_started,
            tools=list(state.tools),
            structured_calls=tool_calls,
        )
        message = Message(role="assistant", content=response.content)
        if response.reasoning_content is not None:
            message["reasoning_content"] = response.reasoning_content
        if response.tool_calls:
            message["tool_calls"] = list(response.tool_calls)
        next_state = RolloutState(
            feature=feature,
            engine_input_ids=state.engine_input_ids + tuple(tokens),
            messages=state.messages + (deepcopy(message),),
            tools=state.tools,
            reasoning_started=bool(response.reasoning_started),
            media=state.media,
        )
        return CompletionEncoding(next_state, response)

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
        decoded = self.decode(token_ids, skip_special_tokens=False, **kwargs)
        reasoning_started = self.enable_thinking if context.reasoning_started is None else context.reasoning_started
        return decode_output(
            decoded,
            thinking_marks=(self.reasoning_open_marker, self.reasoning_close_marker),
            reasoning_started=reasoning_started,
            tool_parser=_RegistryParser() if context.parse_tool_calls and structured_calls is None else None,
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

    def decode(self, token_ids: list[int], **kwargs) -> str:
        return self.processor.decode(token_ids, **kwargs)

    def batch_decode(self, token_ids: list[list[int]], **kwargs) -> list[str]:
        return [self.processor.decode(_ids, **kwargs) for _ids in token_ids]

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
