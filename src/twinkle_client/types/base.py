# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared pydantic base classes, field roles, and the naming rulings for the wire contract.

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

Field roles
-----------
Every declared request field has exactly one role, and the role -- not the field
name -- decides whether it reaches the backend:

- ``Control`` (the default): consumed by the handler itself, or passed explicitly
  as a named argument. ``inputs``, ``adapter_name``, ``seq_id`` and the data-plane
  reference fields are control fields. Forwarding them again through
  ``**backend_kwargs`` would either duplicate a keyword argument or leak a
  protocol field into a backend signature.
- ``BackendKwarg``: a user-facing backend parameter. Forwarded when, and only
  when, its value is not ``None``.
- ``Passthrough``: a declared dict whose *keys* are dynamic. Its contents are
  flattened into the backend kwargs, and its keys are exempt from
  ``extra='forbid'`` (that setting constrains the model's own field set, not the
  inside of a declared dict). The keys are forwarded as given -- see
  :func:`passthrough` for why they are not checked against the target's signature.

The role lives in the field's ``json_schema_extra`` so that one declaration site
carries it -- there is deliberately no second per-endpoint parameter table.
"""
from __future__ import annotations

from enum import StrEnum
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

    ``extra='allow'``, not ``forbid`` and not ``ignore``, and the difference is
    load-bearing. A user's ``Preprocessor`` / ``Template`` routinely leaves extra
    columns on an entry (whatever ``dataset.map`` did not drop). Rejecting those
    would break a large number of existing datasets -- but *ignoring* them is just
    as wrong, because the entry is then re-exported to the backend without them,
    silently dropping data the caller sent. ``allow`` keeps unknown keys on the
    model so :meth:`export` can hand them back.

    Do NOT "fix" this to ``StrictRequest`` or to ``extra='ignore'``. Strictness on
    the data plane belongs to the *declared* fields (the ones Twinkle_Core reads),
    which carry strict types; it does not belong to the field set.
    """

    model_config = ConfigDict(frozen=True, extra='allow')


class FieldRole(StrEnum):
    """How a declared request field relates to the backend call."""

    Control = 'control'
    BackendKwarg = 'backend_kwarg'
    Passthrough = 'passthrough'


# Keys under which field metadata is stored in a field's ``json_schema_extra``.
FIELD_ROLE_KEY = 'twinkle_field_role'
BACKEND_ONLY_KEY = 'twinkle_backend_only'


def _with_extra(field_kwargs: dict[str, Any], **extra: Any) -> FieldInfo:
    merged = dict(field_kwargs.pop('json_schema_extra', None) or {})
    merged.update(extra)
    return Field(json_schema_extra=merged, **field_kwargs)


def backend_kwarg(*backends: str, **field_kwargs: Any) -> FieldInfo:
    """Declare a field as a backend keyword argument.

    With no ``backends`` the field applies to every backend. Naming one or more
    restricts it: sending a non-``None`` value to a deployment running a different
    backend is rejected before the task is enqueued.

    A restricted field MUST be ``Optional[...] = None``. Giving it a backend's
    constant default would make it carry a non-``None`` value on every deployment,
    so the "non-``None`` on the wrong backend" check would reject every request.
    """
    return _with_extra(field_kwargs, **{
        FIELD_ROLE_KEY: FieldRole.BackendKwarg.value,
        BACKEND_ONLY_KEY: tuple(backends) or None,
    })


def backend_only(*backends: str, **field_kwargs: Any) -> FieldInfo:
    """A backend keyword argument restricted to the given backend(s).

    Kept as a named entry point because "this parameter only exists on megatron"
    is the property a reader looks for at the declaration site; it is
    :func:`backend_kwarg` with a non-empty backend tuple, not a second mechanism.
    """
    if not backends:
        raise ValueError('backend_only() requires at least one backend; use backend_kwarg() for an unrestricted field')
    return backend_kwarg(*backends, **field_kwargs)


def passthrough(**field_kwargs: Any) -> FieldInfo:
    """Declare a dict field whose keys are dynamic backend parameters.

    Its contents are forwarded to the backend as given. They are deliberately not
    checked against the target's signature: a plugin routinely reads a real parameter
    straight out of ``**kwargs`` (``InputProcessor`` does this with ``padding_side``),
    and ``inspect.signature`` cannot see such a read -- so any such check rejects valid
    requests. A misspelled plugin argument therefore still surfaces from the plugin.
    """
    field_kwargs.setdefault('default_factory', dict)
    return _with_extra(field_kwargs, **{FIELD_ROLE_KEY: FieldRole.Passthrough.value})


def _read_extra(field_info: FieldInfo, key: str) -> Any:
    extra = getattr(field_info, 'json_schema_extra', None)
    if isinstance(extra, dict):
        return extra.get(key)
    return None


def read_field_role(field_info: FieldInfo) -> FieldRole:
    """The field's role; ``Control`` when undeclared."""
    value = _read_extra(field_info, FIELD_ROLE_KEY)
    return FieldRole(value) if value is not None else FieldRole.Control


def read_backend_only(field_info: FieldInfo) -> tuple[str, ...] | None:
    """Return the backend tuple a field was restricted to, or ``None`` if unrestricted."""
    value = _read_extra(field_info, BACKEND_ONLY_KEY)
    return tuple(value) if value else None


def fields_with_role(model_cls: type[BaseModel], role: FieldRole) -> dict[str, FieldInfo]:
    """The model's declared fields carrying ``role``, in declaration order."""
    return {name: info for name, info in model_cls.model_fields.items() if read_field_role(info) is role}
