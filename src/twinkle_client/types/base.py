# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared pydantic base classes and the naming rulings for the wire contract.

This module is a public contract carrier imported across packages (Twinkle_Server
reverse-imports ``twinkle_client.types``); it therefore intentionally carries **no**
underscore prefix.

Naming rulings (authoritative for all three split specs; kept in code, not only in
the spec, so a later reader cannot merge these away):

1. Schema_Module modules imported across packages do NOT use an underscore prefix.
   ``base.py`` / ``errors.py`` / ``lifecycle.py`` / ``data.py`` are public-contract
   carriers; an underscore means "package-private", and a cross-package import of a
   private module is a violation. Modules used only inside Twinkle_Client (never
   imported by Twinkle_Server) are exempt.
2. New twinkle-native request models do NOT reuse a class name already present in
   ``tinker.types``. Known collision to avoid: ``ForwardBackwardRequest``. Two
   handlers import ``types`` from twinkle_client and from tinker respectively; a
   same-named model is distinguished only by the import alias and is easy to
   misread in a review diff.
3. The field expressing a failure-semantic category is named ``error_code``, NOT
   ``status_code`` -- an execution-time failure is delivered with HTTP 200, so the
   value is systematically unequal to the response status code.
4. A closed value set on a wire field is declared as ``Literal`` / enum, never a
   bare ``str`` (see ``QueueStateLiteral`` in ``errors.py``).

These three base classes are DEFINED here but NOT applied to any existing model by
this spec: applying ``extra='forbid'`` would immediately reject an old client's
request, which would break the zero-wire-change guarantee.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from pydantic.fields import FieldInfo
from typing import Any, Optional


class StrictRequest(BaseModel):
    """Request bodies. Typos fail loudly."""

    model_config = ConfigDict(frozen=True, extra='forbid')


class ResponseModel(BaseModel):
    """Response bodies. An old client tolerates new server fields."""

    model_config = ConfigDict(frozen=True, extra='ignore')


class DataModel(BaseModel):
    """Data-plane models (InputFeature / Trajectory on the wire).

    Same ConfigDict as ResponseModel, different reason -- which is why this is a
    separate class and not an alias. ResponseModel's ``ignore`` exists so an old
    client tolerates new response fields. DataModel's ``ignore`` exists so a
    user's Preprocessor / Template may leave harmless extra keys (the original
    columns left by ``dataset.map``, say) without the request being rejected.

    Do NOT "fix" this to inherit StrictRequest. Doing so rejects those extra keys
    and breaks a large number of existing datasets.
    """

    model_config = ConfigDict(frozen=True, extra='ignore')


# Key under which backend-applicability metadata is stored in a field's
# ``json_schema_extra``. A single constant, helper and reader -- kept here with the
# base classes rather than in a module of their own (no isolation benefit).
BACKEND_ONLY_KEY = 'twinkle_backend_only'


def backend_only(*backends: str, **field_kwargs: Any) -> FieldInfo:
    """Mark a model field as applicable only to the given backend(s).

    Attaches the backend tuple to the field's ``json_schema_extra`` under
    ``BACKEND_ONLY_KEY``; read it back with :func:`read_backend_only`.
    """
    extra = dict(field_kwargs.pop('json_schema_extra', None) or {})
    extra[BACKEND_ONLY_KEY] = tuple(backends)
    return Field(json_schema_extra=extra, **field_kwargs)


def read_backend_only(field_info: FieldInfo) -> Optional[tuple[str, ...]]:
    """Return the backend tuple a field was tagged with, or ``None`` if untagged."""
    extra = getattr(field_info, 'json_schema_extra', None)
    if isinstance(extra, dict):
        value = extra.get(BACKEND_ONLY_KEY)
        if value is not None:
            return tuple(value)
    return None
