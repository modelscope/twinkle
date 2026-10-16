from twinkle.template import deepseek_v4_encoding as bundled_encoding
from twinkle.template.deepseek_v4 import DeepseekV4Template, load_deepseek_v4_encoding

DSML_TOOL_CALL = ('Need data.\n\n'
                  '<｜DSML｜tool_calls>\n'
                  '<｜DSML｜invoke name="search">\n'
                  '<｜DSML｜parameter name="q" string="true">weather</｜DSML｜parameter>\n'
                  '<｜DSML｜parameter name="limit" string="false">3</｜DSML｜parameter>\n'
                  '</｜DSML｜invoke>\n'
                  '</｜DSML｜tool_calls>')


def test_load_encoding_from_model_directory(tmp_path):
    encoding_dir = tmp_path / 'encoding'
    encoding_dir.mkdir()
    (encoding_dir / 'encoding_dsv4.py').write_text(
        'dsml_token = "custom"\n'
        'tool_calls_block_name = "calls"\n'
        'eos_token = "eos"\n'
        'def encode_messages(*args, **kwargs): return "custom prompt"\n'
        'def parse_message_from_completion_text(*args, **kwargs): return {}\n',
        encoding='utf-8',
    )

    encoding = load_deepseek_v4_encoding(str(tmp_path))

    assert encoding.dsml_token == 'custom'
    assert encoding.encode_messages([]) == 'custom prompt'


def test_load_encoding_falls_back_to_bundled_module(tmp_path):
    assert load_deepseek_v4_encoding(str(tmp_path)) is bundled_encoding


def test_deepseek_v4_parse_and_clean_tool_call():
    template = DeepseekV4Template.__new__(DeepseekV4Template)

    calls = template.parse(DSML_TOOL_CALL)

    assert calls == [{
        'type': 'function',
        'function': {
            'name': 'search',
            'arguments': {
                'q': 'weather',
                'limit': 3,
            },
        },
    }]
    assert template.clean(DSML_TOOL_CALL) == 'Need data.'
    assert template.parse_tool_call(DSML_TOOL_CALL) == calls
    assert template.clean_tool_call(DSML_TOOL_CALL) == 'Need data.'


def test_deepseek_v4_parse_tool_call_normalizes_block_prefix_whitespace():
    template = DeepseekV4Template.__new__(DeepseekV4Template)
    expected_args = {'q': 'weather', 'limit': 3}

    for separator in ('', '\n', '\n\n\n', '   '):
        text = DSML_TOOL_CALL.replace('Need data.\n\n<｜DSML｜tool_calls>', f'Need data.{separator}<｜DSML｜tool_calls>')

        calls = template.parse(text)

        assert calls[0]['function']['name'] == 'search'
        assert calls[0]['function']['arguments'] == expected_args


def test_template_dispatches_deepseek_tool_call_parser():
    template = DeepseekV4Template.__new__(DeepseekV4Template)

    calls = template.parse_tool_call(DSML_TOOL_CALL)

    assert calls[0]['function']['name'] == 'search'
    assert template.clean_tool_call(DSML_TOOL_CALL) == 'Need data.'
