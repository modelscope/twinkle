# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for twinkle.hub.model_alias (route-prefix alias resolution)."""
from types import SimpleNamespace

from twinkle.hub.model_alias import (
    MODEL_ID_ALIASES_ENV,
    build_model_alias_map,
    extract_model_alias,
    load_model_alias_map,
    resolve_model_id_alias,
)


class TestExtractModelAlias:

    def test_model_route_prefix_yields_the_alias(self):
        assert extract_model_alias('/model/my-alias') == 'my-alias'
        assert extract_model_alias('/model/a/b') == 'a/b'

    def test_slashes_are_trimmed(self):
        assert extract_model_alias('/model//alias/') == 'alias'

    def test_other_or_missing_marker_yields_none(self):
        assert extract_model_alias('/deployment/foo') is None
        assert extract_model_alias('') is None
        assert extract_model_alias(None) is None
        assert extract_model_alias('/model/') is None

    def test_custom_service_type(self):
        assert extract_model_alias('/embedding/x', service_type='embedding') == 'x'
        assert extract_model_alias('/model/x', service_type='embedding') is None


class TestBuildModelAliasMap:

    def _application(self, *, import_path='model', route_prefix='/model/alias', model_id='real/id'):
        args = SimpleNamespace(model_id=model_id) if model_id is not None else None
        return SimpleNamespace(import_path=import_path, route_prefix=route_prefix, args=args)

    def test_model_applications_map_alias_to_model_id(self):
        applications = [self._application()]
        assert build_model_alias_map(applications) == {'alias': 'real/id'}

    def test_mapping_and_attribute_args_are_both_supported(self):
        application = {
            'import_path': 'model',
            'route_prefix': '/model/alias',
            'args': {'model_id': 'real/id'},
        }
        assert build_model_alias_map([application]) == {'alias': 'real/id'}

    def test_non_model_applications_are_skipped(self):
        applications = [self._application(import_path='deployment')]
        assert build_model_alias_map(applications) == {}

    def test_alias_equal_to_model_id_is_not_recorded(self):
        applications = [self._application(route_prefix='/model/real/id', model_id='real/id')]
        assert build_model_alias_map(applications) == {}

    def test_missing_pieces_are_skipped(self):
        assert build_model_alias_map([self._application(route_prefix='/other')]) == {}
        assert build_model_alias_map([self._application(model_id=None)]) == {}
        assert build_model_alias_map([]) == {}
        assert build_model_alias_map(None) == {}


class TestLoadModelAliasMap:

    def test_json_string_is_parsed(self):
        assert load_model_alias_map('{"a": "b"}') == {'a': 'b'}

    def test_invalid_json_and_non_dict_payload_return_empty(self):
        assert load_model_alias_map('not json') == {}
        assert load_model_alias_map('["a", "b"]') == {}

    def test_empty_keys_and_values_are_dropped(self):
        assert load_model_alias_map('{"": "b", "a": "", "c": "d"}') == {'c': 'd'}

    def test_mapping_input_is_coerced(self):
        assert load_model_alias_map({'a': 1}) == {'a': '1'}

    def test_env_variable_is_the_default_source(self, monkeypatch):
        monkeypatch.setenv(MODEL_ID_ALIASES_ENV, '{"env-alias": "env/id"}')
        assert load_model_alias_map() == {'env-alias': 'env/id'}
        monkeypatch.delenv(MODEL_ID_ALIASES_ENV)
        assert load_model_alias_map() == {}


class TestResolveModelIdAlias:

    def test_known_alias_resolves_to_the_model_id(self):
        assert resolve_model_id_alias('alias', {'alias': 'real/id'}) == 'real/id'

    def test_unknown_id_passes_through(self):
        assert resolve_model_id_alias('other', {'alias': 'real/id'}) == 'other'

    def test_empty_input_returns_as_is(self):
        assert resolve_model_id_alias(None, {'a': 'b'}) is None
        assert resolve_model_id_alias('', {'a': 'b'}) == ''

    def test_env_backed_resolution(self, monkeypatch):
        monkeypatch.setenv(MODEL_ID_ALIASES_ENV, '{"env-alias": "env/id"}')
        assert resolve_model_id_alias('env-alias') == 'env/id'
