# Copyright (c) ModelScope Contributors. All rights reserved.
"""Wire schema for the inline ``inputs`` data plane.

These models are the *declared* type of the ``inputs`` request field, so FastAPI
validates a batch during body parsing -- before the handler runs, before a future
record exists, and before anything reaches a GPU. The seam that hands data to the
backend (``twinkle.server.lifecycle.submit.to_backend_inputs``) therefore only
exports an already-valid object; it is not the first place a malformed batch is
noticed.

Two asymmetries are deliberate:

- **Strict on declared fields, open on the field set.** Every field Twinkle_Core
  reads is declared with a strict type (``StrictInt`` leaves reject ``true`` and
  ``1.0``), while unknown JSON-native keys are kept and re-exported: a user's
  preprocessor may leave extra columns on an entry and dropping them would lose
  data the caller sent. See :class:`~twinkle_client.types.base.DataModel`.
- **Shallowest-first unions.** Nesting depth encodes tensor rank here, so a rank
  range needs a union. Declaring the deepest branch first is a large, silent
  pessimisation: given a 2-D input the 3-D branch does not fail at element 0, it
  descends into every row and records one error per element. Measured on pydantic
  2.13.4 with a 1024 x 8192 (8.4M element) 2-D ``input_ids``:

      List[List[int]] single type         0.141 s
      Union[3D, 2D, 1D] deepest-first     5.368 s   <-- 41x worse
      Union[1D, 2D, 3D] shallowest-first  0.131 s

  Do NOT reorder these to deepest-first. ``test_wire_schema.py`` asserts the
  declared depth sequence is strictly increasing, so the ordering is checked
  structurally rather than by a flaky timing benchmark.

Known technical debt: encoding tensor shape in JSON nesting depth is what forces
the unions and makes validation cost scale with element count. The target shape is
a flat ``{dtype, shape, data}`` tensor envelope (which is what ``tinker`` uses).
Migrating is a breaking wire change and is out of scope here; do not paper over it
with hand-written Python-level depth checks or extra union branches, which would
add to the debt rather than pay it down.
"""
from __future__ import annotations

from collections.abc import Mapping
from pydantic import BeforeValidator, Field, StrictInt, model_validator
from typing import Annotated, Any, List, Literal, Optional, Union

from twinkle.data_format.encoding import ENCODED_INPUT_KEYS
from twinkle_client.types.base import DataModel

# --------------------------------------------------------------------------- #
# Leaf types. Shallowest-first, and ``StrictInt`` wherever the values come from a
# tensor's ``tolist()`` -- there, a bool or a float is an upstream defect. Lax
# ``int`` coercion would turn ``[true, false]`` into ``[1, 0]``.
#
# ``union_mode='left_to_right'`` makes the declared order load-bearing instead of
# leaving branch selection to pydantic's heuristics.
# --------------------------------------------------------------------------- #

_LEFT_TO_RIGHT = Field(union_mode='left_to_right')

Ints1to2 = Annotated[Union[List[StrictInt], List[List[StrictInt]]], _LEFT_TO_RIGHT]
Ints1to3 = Annotated[Union[List[StrictInt], List[List[StrictInt]], List[List[List[StrictInt]]]], _LEFT_TO_RIGHT]
Ints3 = List[List[List[StrictInt]]]

_Number = Union[StrictInt, float]
Numbers1to2 = Annotated[Union[List[_Number], List[List[_Number]]], _LEFT_TO_RIGHT]
Numbers1to4 = Annotated[Union[List[_Number], List[List[_Number]], List[List[List[_Number]]],
                              List[List[List[List[_Number]]]]], _LEFT_TO_RIGHT]

# Media references travel as strings on the wire (local path, ``http(s)://`` URL, or
# a ``data:`` base64 URI). ``PIL.Image`` / raw ``bytes`` / ``np.ndarray`` are valid in
# the in-process training path but are not JSON, so they are not declared here.
MediaList = List[str]

# The VLM tensor fields batched by concatenation rather than padding. Declared here
# because this module must stay free of Twinkle_Core's heavyweight imports; a
# consistency test asserts this set equals ``InputProcessor.VLM_CONCAT_FIELDS``, so a
# future addition there fails loudly instead of being silently dropped on the wire.
VLM_TENSOR_FIELDS: frozenset[str] = frozenset({
    'pixel_values',
    'image_grid_thw',
    'pixel_values_videos',
    'video_grid_thw',
    'input_features',
    'input_features_mask',
    'feature_attention_mask',
    'grid_thws',
})


class WireMessage(DataModel):
    """One conversation turn, as sent over HTTP."""

    role: Literal['system', 'user', 'assistant', 'tool'] | None = None
    type: str | None = None
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    reasoning_content: str | None = None
    images: MediaList | None = None
    videos: MediaList | None = None
    audios: MediaList | None = None


class WireInputFeature(DataModel):
    """An already-encoded entry: token ids (or embeddings) plus aligned tensors."""

    input_ids: Ints1to2 | None = None
    input_embedding: Numbers1to2 | None = None
    attention_mask: Ints1to2 | None = None
    labels: Ints1to2 | None = None
    completion_mask: Ints1to2 | None = None
    # 1-D standard encoding, 2-D Qwen-VL mrope ``[3, T]``, 3-D megatron ``[3, 1, N]``.
    position_ids: Ints1to3 | None = None
    # Exactly ``[seq_len, num_layers, topk]``.
    routed_experts: Ints3 | None = None
    length: StrictInt | None = None

    # VLM tensors: float values are normal here, so no strict-int leaves.
    pixel_values: Numbers1to4 | None = None
    image_grid_thw: Numbers1to4 | None = None
    pixel_values_videos: Numbers1to4 | None = None
    video_grid_thw: Numbers1to4 | None = None
    input_features: Numbers1to4 | None = None
    input_features_mask: Numbers1to4 | None = None
    feature_attention_mask: Numbers1to4 | None = None
    grid_thws: Numbers1to4 | None = None

    @model_validator(mode='after')
    def require_encoded_key(self) -> WireInputFeature:
        """At least one of the encoded-input keys must be present.

        Declared as a model validator rather than by making ``input_ids`` required:
        an embedding-only batch is legitimately encoded, and this is the same rule
        the backends apply (:data:`ENCODED_INPUT_KEYS`).
        """
        if all(getattr(self, key, None) is None for key in ENCODED_INPUT_KEYS):
            raise ValueError(f'an encoded entry requires one of {list(ENCODED_INPUT_KEYS)}')
        return self


class WireTrajectory(DataModel):
    """A not-yet-encoded entry: messages the server template will encode."""

    messages: list[WireMessage]
    images: MediaList | None = None
    videos: MediaList | None = None
    audios: MediaList | None = None
    tools: list[dict[str, Any]] | None = None
    # ``List[Tuple[str, str]]`` on the wire: the PyArrow-stable encoding of the
    # user-data pairs attached by ``twinkle.data_format.attach_user_data``.
    user_data: list[tuple[str, str]] | None = None


# A batch is homogeneous: every entry is encoded, or none is. Expressed as a union of
# *lists* rather than a list of unions, so a mixed batch fails to match either branch
# instead of being silently accepted and blowing up inside the backend. Order is
# encoded-first, matching ``is_encoded``: a trajectory has neither encoded key, so it
# cannot satisfy ``WireInputFeature``'s validator.
WireInputs = Union[List[WireInputFeature], List[WireTrajectory]]


def _as_batch(value: Any) -> Any:
    """Accept a single entry where a batch is expected.

    Callers have always been allowed to pass one mapping instead of a one-element
    list; normalising here keeps that while letting the declared type stay a batch,
    so downstream code has exactly one shape to handle.
    """
    return [value] if isinstance(value, Mapping) else value


#: The declared type of an inline ``inputs`` request field.
WireInputBatch = Annotated[WireInputs, BeforeValidator(_as_batch)]

# Every ``inputs`` key Twinkle_Core reads. Maintained by hand on purpose: an AST scan
# would have to follow aliases (``inp = inputs[i]`` then ``inp.get('x')``), i.e. do a
# local data-flow analysis, and its false negatives would *silently* disable the
# consistency check that is this schema's only safety net against a dropped field.
# When adding a read of a new ``inputs`` key, add it here.
CORE_INPUT_KEYS: frozenset[str] = frozenset({
    'input_ids',
    'input_embedding',
    'attention_mask',
    'labels',
    'completion_mask',
    'position_ids',
    'routed_experts',
    'length',
    'messages',
    'images',
    'videos',
    'audios',
    'tools',
    'user_data',
}) | VLM_TENSOR_FIELDS


def declared_wire_keys() -> frozenset[str]:
    """Union of the field names declared across the wire input models."""
    return frozenset(WireInputFeature.model_fields) | frozenset(WireTrajectory.model_fields)


def export(entry: WireInputFeature | WireTrajectory) -> dict[str, Any]:
    """Render a validated entry as the plain dict the backend consumes.

    ``exclude_none=True`` is required, not cosmetic: Twinkle_Core branches on key
    *presence* in many places (``is_encoded``, ``inputs.pop('labels', None)``, the
    VLM concat fields), so emitting unset optionals as ``None`` would change
    behaviour. Unknown keys the caller sent are preserved -- that is why
    :class:`DataModel` uses ``extra='allow'``.
    """
    return entry.model_dump(exclude_none=True)


def export_batch(entries: list[Any]) -> list[dict[str, Any]]:
    """Export a validated batch, leaving already-plain entries untouched."""
    return [export(entry) if isinstance(entry, DataModel) else entry for entry in entries]
