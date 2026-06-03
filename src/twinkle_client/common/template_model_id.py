from __future__ import annotations

from typing import Dict

from twinkle_client.http import get_base_url, http_get
from twinkle_client.types.server import GetServerCapabilitiesResponse, SupportedModel

_SERVER_CAPABILITIES_CACHE: Dict[str, GetServerCapabilitiesResponse] = {}


def _get_server_capabilities(refresh: bool = False) -> GetServerCapabilitiesResponse:
    base_url = get_base_url()
    if refresh or base_url not in _SERVER_CAPABILITIES_CACHE:
        response = http_get(f'{base_url}/twinkle/get_server_capabilities')
        response.raise_for_status()
        _SERVER_CAPABILITIES_CACHE[base_url] = GetServerCapabilitiesResponse(**response.json())
    return _SERVER_CAPABILITIES_CACHE[base_url]


def get_supported_model_by_name(model_name: str, refresh: bool = False) -> SupportedModel | None:
    capabilities = _get_server_capabilities(refresh=refresh)
    for supported_model in capabilities.supported_models:
        if supported_model.model_name == model_name:
            return supported_model
    return None


def resolve_template_model_id(model_name: str, explicit_model_id: str | None = None) -> str:
    if explicit_model_id is not None:
        return explicit_model_id

    try:
        supported_model = get_supported_model_by_name(model_name)
        if supported_model and supported_model.template_init_model_id:
            return supported_model.template_init_model_id
    except Exception:
        pass
    return model_name
