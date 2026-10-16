# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tool-call parser selection and the bracketed call-list format."""
import pytest

from twinkle.template.tools import ToolCallRegistry
from twinkle.template.tools.bracket_dsl import BracketDslParser

FENCE = '```'


def names(text):
    parser = BracketDslParser()
    if not parser.detect(text):
        return []
    return [c['function']['name'] for c in parser.parse(text)]


@pytest.mark.parametrize(
    'label, text, expected',
    [
        ('call list alone',
         '[Text Analysis(text="great service"), UserID(username="alex")]',
         ['Text Analysis', 'UserID']),
        ('call list after prose',
         'Here you go: [quarterly_data(stock_symbols=["AAPL", "TSLA"])]',
         ['quarterly_data']),
        ('dotted name', '[database.insert_data(table="t")]', ['database.insert_data']),
    ],
)
def test_bracket_dsl_parses_call_lists(label, text, expected):
    assert names(text) == expected


@pytest.mark.parametrize(
    'label, text',
    [
        # A comprehension is shaped exactly like a call list. Reading one as
        # tool calls invents names like 'float' and 'for _ in range', and the
        # tools the model meant to call never run.
        ('comprehension in a fence',
         f'Sure:\n{FENCE}python\nvals = [float(random.uniform(1, 10)) for _ in range(20)]\n{FENCE}\n'),
        ('nested comprehension in a fence',
         f'{FENCE}\nrows = [dict(zip(h, r)) for r in raw]\n{FENCE}\n'),
        # A reply truncated mid-fence still has to be treated as code.
        ('unterminated fence', f'writing code:\n{FENCE}python\ny = [str(i) for i in xs]'),
        ('plain prose list', 'the values are [1, 2, 3]'),
        # A model writing code while it thinks does not use fences. This is how
        # 10% of the episodes in an agentic run lost their tool calls: the reply
        # was cut off inside <think>, the comprehension in it parsed as calls to
        # `int` and `for _ in range`, and the sandbox was never touched.
        ('comprehension in unfenced prose',
         'I will write vals = [float(random.uniform(1, 10)) for _ in range(20)] next'),
        ('comprehension inside a think block',
         '<think>\nnums = [int(v) for v in raw]\n</think>\nDone.'),
        ('reply truncated inside think, comprehension left open',
         '<think>\nSo the code would be:\n\nvals = [int(x) for x in lines]\nWait, maybe'),
        ('a call list rehearsed while thinking is not a call',
         '<think>\nI could answer [get_price(sym="AAPL")] here.\n</think>\nLet me check first.'),
        ('positional argument is not a call list', '[get_price("AAPL")]'),
    ],
)
def test_bracket_dsl_ignores_code_and_prose(label, text):
    assert names(text) == []


def test_bracket_dsl_sees_the_call_after_a_closed_think_block():
    text = '<think>\nvals = [int(v) for v in raw]\n</think>\n[get_price(sym="AAPL")]'
    assert names(text) == ['get_price']


def test_bracket_dsl_accepts_a_call_with_no_arguments():
    assert names('[get_time()]') == ['get_time']


def test_bracket_dsl_still_sees_calls_outside_a_fence():
    text = f'{FENCE}python\nx = [int(v) for v in raw]\n{FENCE}\n[get_price(sym="AAPL")]'
    assert names(text) == ['get_price']


def test_marked_up_formats_win_over_the_bracket_heuristic():
    """Hermes markup must go to Hermes even when its arguments contain ``[f(``."""
    text = ('<tool_call>\n{"name": "shell_executor", '
            '"arguments": {"command": "python -c \'print([int(x) for x in y])\'"}}\n</tool_call>')
    parser = ToolCallRegistry.detect_first(text)
    assert parser is not None and parser.name != 'bracket_dsl'
    assert [c['function']['name'] for c in parser.parse(text)] == ['shell_executor']
