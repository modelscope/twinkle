"""Content-signature intent detectors must ignore markdown/LaTeX inside <think>.

Regression for the copywriting-tagged-as-code bug: a non-code answer whose
private <think> scratch-pad contained a ``` fence was misclassified as ``code``.
Task type must be decided by the visible response, not the reasoning block.
"""

from twinkle_agentic.preprocessor.intent_classifier import (CodeDetector,
                                                            MathDetector)


def _asst(content):
    return {'role': 'assistant', 'content': content}


def test_code_fence_only_in_think_is_not_code():
    msgs = [
        {'role': 'user', 'content': '为门店写一条短视频口播脚本'},
        _asst('<think>1. 分析需求\n```\n钩子→痛点→转化\n```\n</think>'
              '钩子：这价格我不敢信。转化：现在下单立省八千。'),
    ]
    assert CodeDetector()(msgs) == []


def test_real_code_in_visible_answer_still_detected():
    msgs = [
        {'role': 'user', 'content': '写个快排'},
        _asst('<think>先想边界</think>```python\n'
              'def quicksort(a):\n    return a\n```'),
    ]
    assert CodeDetector()(msgs) == [1]


def test_user_code_request_not_stripped():
    # A code block in the USER turn is a genuine signal and must NOT be stripped.
    msgs = [
        {'role': 'user', 'content': '```python\nprint(1)\n```\n这段有什么问题'},
        _asst('这里没有问题。'),
    ]
    assert CodeDetector()(msgs) == [1]


def test_latex_only_in_think_is_not_math():
    msgs = [
        {'role': 'user', 'content': '把这段话润色一下'},
        _asst(r'<think>可以用 \frac{a}{b} \sum \int \sqrt{x} 打个比方</think>'
              '润色后的文字，通顺自然。'),
    ]
    assert MathDetector()(msgs) == []
