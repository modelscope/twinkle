from twinkle.template.base import Template
from twinkle.template.deepseek_v4 import DeepseekV4Template

DSML_TOOL_CALL = ('Need data.\n\n'
                  '<｜DSML｜tool_calls>\n'
                  '<｜DSML｜invoke name="search">\n'
                  '<｜DSML｜parameter name="q" string="true">weather</｜DSML｜parameter>\n'
                  '<｜DSML｜parameter name="limit" string="false">3</｜DSML｜parameter>\n'
                  '</｜DSML｜invoke>\n'
                  '</｜DSML｜tool_calls>')


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
    template = Template.__new__(Template)
    template.model_id = 'deepseek-v4'

    calls = template.parse_tool_call(DSML_TOOL_CALL)

    assert calls[0]['function']['name'] == 'search'
    assert template.clean_tool_call(DSML_TOOL_CALL) == 'Need data.'
