# Copyright (c) ModelScope Contributors. All rights reserved.
"""Build a twinkle-native request body from a caller's keyword arguments.

Every public client method used to hand-assemble a ``json_data`` dict. That made the
request schema a second, implicit source of truth: a field the server declared but the
client never sent, or a name only one side spelled correctly, was invisible until a
request failed on the wire. Here the request *model* is the only source of truth --
the client instantiates it, so the same rules that guard the server also guard the
caller, in-process and without a round trip.

Routing a caller's ``**kwargs`` needs exactly three rules, all read off the model:

1. the name is a declared field -> assign it to that field;
2. it is not declared and the model has exactly one passthrough region -> put it there;
3. it is not declared and the model has none, or more than one -> raise.

Rule 3 does not guess. A processor call has both ``init_kwargs`` (the constructor's
arguments) and ``call_kwargs`` (the invoked method's); no rule based on the name alone
can tell which one a caller meant, and choosing wrong sends a valid argument to the
wrong callable -- a silently wrong result rather than an error.
"""
from __future__ import annotations

from pydantic import BaseModel
from typing import Any, Mapping

from twinkle_client.common.json_utils import json_safe
from twinkle_client.exceptions import TwinkleClientValidationError
from twinkle_client.types.base import FieldRole, fields_with_role


def to_wire_value(value: Any) -> Any:
    """Convert one caller-supplied argument to a JSON-native value.

    Handles the three object kinds the client has always accepted in a request body:
    a server-side component handle (sent as its id), a ``DatasetMeta`` / ``LoraConfig``
    (sent as the canonical serialized form the server decodes), and numpy / torch
    values (sent as nested lists).

    Anything else is passed through for the model to validate, so an unsupported type
    is reported by pydantic with its field path instead of by a generic error here.
    """
    if hasattr(value, 'processor_id'):
        return value.processor_id
    from twinkle.dataset import DatasetMeta
    from peft import LoraConfig
    if isinstance(value, (DatasetMeta, LoraConfig)):
        from twinkle_client.common.serialize import serialize_object
        return serialize_object(value)
    if isinstance(value, BaseModel):
        return value.model_dump(mode='json')
    return json_safe(value)


def build_request(model_cls: type[BaseModel], /, **values: Any) -> BaseModel:
    """Instantiate ``model_cls`` from caller arguments, routing undeclared names.

    ``None`` values for undeclared names are dropped rather than routed: client methods
    pass optional arguments unconditionally, and forwarding an explicit ``None`` into a
    passthrough region would hand the backend a null it never had before.

    Raises:
        TwinkleClientValidationError: an argument has no field and no unambiguous
            passthrough region, or it collides with an explicitly passed region key.
        pydantic.ValidationError: the assembled body violates the model.
    """
    declared = model_cls.model_fields
    regions = fields_with_role(model_cls, FieldRole.Passthrough)

    body: dict[str, Any] = {}
    routed: dict[str, Any] = {}
    for name, value in values.items():
        if name in declared:
            body[name] = to_wire_value(value) if value is not None else None
        elif value is None:
            continue
        elif len(regions) == 1:
            routed[name] = to_wire_value(value)
        elif not regions:
            raise TwinkleClientValidationError(
                f'{model_cls.__name__} has no field {name!r} and no passthrough region to put it in; '
                f'known fields: {sorted(declared)}')
        else:
            raise TwinkleClientValidationError(
                f'{model_cls.__name__} has no field {name!r} and more than one passthrough region, so its '
                f'target is ambiguous. Pass it inside one of: {sorted(regions)}')

    if routed:
        region = next(iter(regions))
        explicit = body.get(region) or {}
        if not isinstance(explicit, Mapping):
            raise TwinkleClientValidationError(f'{region} must be a mapping, got {type(explicit).__name__}')
        collisions = set(explicit) & set(routed)
        if collisions:
            raise TwinkleClientValidationError(
                f'these arguments were passed both directly and inside {region}: {sorted(collisions)}')
        body[region] = {**explicit, **routed}

    return model_cls(**body)


def request_json(body: BaseModel) -> str:
    """Serialize a request body once.

    ``exclude_none=True`` keeps unset optionals off the wire, which is what lets the
    server treat "absent" and "not requested" as the same thing rather than maintaining
    a second set of defaults. One pydantic-core pass, not a Python-level walk followed
    by ``json.dumps``.
    """
    return body.model_dump_json(exclude_none=True)
