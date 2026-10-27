# data_format helper tests (plan T10b-01..T10b-12).
import math

import pytest
import torch

from twinkle.data_format import attach_user_data, last_assistant_text, msg_content_text, pack_user_data, pack_value, user_data_get
from twinkle.data_format.message import FunctionCall, FunctionSpec
from twinkle.data_format.output import ModelOutput
from twinkle.data_format.sampling import PoolingParams, PoolingResponse, pooling_to_list


class TestPackUserData:
    """T10b-01/T10b-09: pack -> get round-trip and value typing."""

    @pytest.mark.parametrize('value', [42, 3.5, True, 'plain', None, {'a': [1, 2]}, [1, 'x']])
    def test_round_trip_preserves_type(self, value):
        packed = pack_user_data({'k': value})
        assert user_data_get(packed, 'k') == value
        assert isinstance(user_data_get(packed, 'k'), type(value)) or value is None

    def test_pack_value_is_json_string(self):
        out = pack_value({'a': 1})
        assert isinstance(out, str)
        assert user_data_get([('k', out)], 'k') == {'a': 1}

    def test_pre_packed_string_entries_left_untouched(self):
        # A value that is already a JSON string is stored verbatim (no double encoding).
        packed = pack_user_data({'k': '"already json"'})
        assert packed == [('k', '"already json"')]
        assert user_data_get(packed, 'k') == 'already json'

    def test_list_of_pairs_input(self):
        packed = pack_user_data([('a', 1), ('b', 2)])
        assert user_data_get(packed, 'a') == 1
        assert user_data_get(packed, 'b') == 2

    def test_none_input_returns_empty_list(self):
        assert pack_user_data(None) == []


class TestUserDataGet:
    """T10b-02/T10b-03/T10b-04: the mapping guard, non-list inputs, duplicate keys."""

    def test_mapping_raises_with_fix_hint(self):
        with pytest.raises(TypeError, match='pack_user_data'):
            user_data_get({'k': 1}, 'k')

    def test_non_list_returns_default(self):
        assert user_data_get(None, 'k') is None
        assert user_data_get('not a list', 'k', default=7) == 7
        assert user_data_get(123, 'k') is None

    def test_first_match_wins_on_duplicate_keys(self):
        items = [('k', '1'), ('k', '2')]
        assert user_data_get(items, 'k') == 1

    def test_non_json_string_returned_verbatim(self):
        # A value that fails json.loads comes back as the raw string.
        assert user_data_get([('k', 'not json {')], 'k') == 'not json {'

    def test_non_string_entry_returned_as_is(self):
        # A non-JSON-packed raw value (int/tensor) is passed through without decoding.
        assert user_data_get([('k', 5)], 'k') == 5


class TestLastAssistantText:
    """T10b-05/T10b-06/T10b-07: last assistant turn, part lists, missing assistant."""

    def test_multi_turn_takes_last_assistant(self):
        traj = {'messages': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'first'},
            {'role': 'user', 'content': 'q2'},
            {'role': 'assistant', 'content': 'final'},
        ]}
        assert last_assistant_text(traj) == 'final'

    def test_part_list_content_text_only(self):
        traj = {'messages': [{'role': 'assistant', 'content': [
            {'type': 'image', 'image': 'x'},
            {'type': 'text', 'text': 'hello '},
            {'type': 'text', 'text': 'world'},
        ]}]}
        # msg_content_text joins text parts with a space separator.
        assert last_assistant_text(traj) == 'hello  world'

    def test_no_assistant_returns_empty(self):
        assert last_assistant_text({'messages': [{'role': 'user', 'content': 'q'}]}) == ''
        assert last_assistant_text({}) == ''

    def test_dict_content_text_type(self):
        assert msg_content_text({'content': {'type': 'text', 'text': 'hi'}}) == 'hi'
        assert msg_content_text({'content': 42}) == ''


class TestAttachUserData:
    """T10b-08: merging into a packed trajectory keeps the readable shape."""

    def test_merge_keeps_existing_and_adds_new(self):
        traj = {'messages': [], 'user_data': pack_user_data({'a': 1})}
        out = attach_user_data(traj, b=2)
        assert user_data_get(out['user_data'], 'a') == 1
        assert user_data_get(out['user_data'], 'b') == 2
        # The input trajectory is not mutated.
        assert 'b' not in [k for k, _ in traj['user_data']]

    def test_new_value_overwrites_same_key(self):
        traj = {'messages': [], 'user_data': pack_user_data({'a': 1})}
        out = attach_user_data(traj, a=9)
        assert user_data_get(out['user_data'], 'a') == 9


class TestPoolingHelpers:
    """T10b-10/T10b-11: PoolingParams/PoolingResponse fields and pooling_to_list coercion."""

    def test_pooling_to_list_tensor(self):
        out = pooling_to_list(torch.tensor([1.0, 2.0]))
        assert out == [1.0, 2.0]
        assert all(isinstance(v, float) for v in out)

    def test_pooling_to_list_scalar(self):
        assert pooling_to_list(0.5) == [0.5]

    def test_pooling_to_list_rejects_nested_lists(self):
        # The contract is a FLAT container (tensor / list of numbers / scalar); a nested
        # list is not one of the backend output shapes and fails loudly here.
        with pytest.raises(TypeError):
            pooling_to_list([[1.0], [2.0]])

    def test_pooling_to_list_normalize(self):
        out = pooling_to_list([3.0, 4.0], normalize=True)
        norm = math.sqrt(3.0 ** 2 + 4.0 ** 2)
        assert out == pytest.approx([3.0 / norm, 4.0 / norm])

    def test_pooling_params_fields_and_response_round_trip(self):
        params = PoolingParams(task='classify', use_activation=True)
        resp = PoolingResponse(data=[0.7], task=params.task)
        assert resp.data == [0.7]
        assert resp.task == 'classify'
        assert params.task == 'classify'


class TestTypedDictContracts:
    """T10b-12: the wire TypedDicts accept their documented shapes."""

    def test_function_call_and_spec(self):
        call = FunctionCall(name='get_weather', arguments={'city': 'beijing'})
        spec = FunctionSpec(name='get_weather', description='d', parameters={'type': 'object'})
        assert call['name'] == 'get_weather'
        assert spec['parameters'] == {'type': 'object'}

    def test_model_output_shape(self):
        out = ModelOutput(loss=0.5, logps=[1.0])
        assert out['loss'] == 0.5
