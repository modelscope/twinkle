"""ValueSelector: deterministic value_score + component boundaries + top-frac gate."""

from twinkle_agentic.preprocessor import ValueSelector
from twinkle_agentic.preprocessor import label_schema as L


def _single_turn():
    return {'messages': [
        {'role': 'user', 'content': '写一句短视频文案'},
        {'role': 'assistant', 'content': '这价格我不敢信，现在下单立省八千。'},
    ], 'user_data': []}


def _tool_call(name, args, result='ok', content=''):
    return [
        {'role': 'assistant', 'content': content,
         'tool_calls': [{'id': 't', 'type': 'function',
                         'function': {'name': name, 'arguments': args}}]},
        {'role': 'tool', 'tool_call_id': 't', 'content': result},
    ]


def _agent_row(steps):
    msgs = [{'role': 'user', 'content': 'do the task'}]
    for name, args, result in steps:
        msgs += _tool_call(name, args, result)
    return {'messages': msgs, 'user_data': []}


def _val(row):
    return L.get_label(row, L.KEY_VALUE_SCORE)


def _meta(row):
    return L.get_label(row, L.KEY_VALUE_META) or {}


def test_single_turn_scores_low():
    out, dropped = ValueSelector()([_single_turn()])
    assert dropped == []
    v = _val(out[0])
    assert v is not None and v < 0.25
    m = _meta(out[0])
    assert m['uncertainty'] == 0.0  # single all-pass round -> decided


def test_tool_error_raises_error_signal():
    # a tool that returns an ERROR result -> tool_executed miss -> error signal up
    row = _agent_row([('search', '{"q":"x"}', 'ERROR: not found'),
                      ('search', '{"q":"y"}', 'ERROR: not found')])
    out, _ = ValueSelector()([row])
    assert _meta(out[0])['error'] > 0.0
    assert _val(out[0]) > _val(ValueSelector()([_single_turn()])[0][0])


def test_long_agent_scores_higher_than_single_turn():
    steps = [(f'read', '{"p":"%d"}' % i, 'contents') for i in range(6)]
    steps += [('grep', '{"q":"a"}', 'hit'), ('edit', '{"f":"b"}', 'done')]
    long_row = _agent_row(steps)
    out_long, _ = ValueSelector()([long_row])
    out_short, _ = ValueSelector()([_single_turn()])
    assert _meta(out_long[0])['difficulty'] > _meta(out_short[0])['difficulty']
    assert _val(out_long[0]) > _val(out_short[0])


def test_bad_row_never_crashes():
    out, dropped = ValueSelector()([{'messages': None, 'user_data': []},
                                    {'messages': [], 'user_data': []}])
    assert dropped == []
    assert all(_val(r) == 0.0 for r in out)


def test_select_top_for_rubric_marks_global_top():
    from datasets import Dataset as HFDataset

    class _Wrap:
        def __init__(self, hf):
            self.dataset = hf

    from twinkle_agentic.preprocessor import select_top_for_rubric

    def _row_with_value(v):
        return {'messages': [{'role': 'user', 'content': 'x'}],
                'user_data': [(L.KEY_VALUE_SCORE, str(v))]}

    hf = HFDataset.from_list([_row_with_value(v) for v in [0.1, 0.9, 0.5, 0.8, 0.2]])
    ds = _Wrap(hf)
    ds, n_sel = select_top_for_rubric(ds, select_frac=0.4)  # top 2 of 5
    assert n_sel == 2
    selected = [L.get_label(ds.dataset[i], L.KEY_SELECTED_FOR_RUBRIC, False)
                for i in range(len(ds.dataset))]
    # rows with value 0.9 and 0.8 are the top-2
    assert selected == [False, True, False, True, False]


class _SpyRubric:
    """Records how many times the rubric was invoked."""
    max_votes = 1

    def __init__(self):
        self.calls = 0
        self.diagnose_calls = 0

    def score_detail(self, segment, query=None, intent=None, extra_context=None):
        self.calls += 1

        class _D:
            scalar = 0.5
            votes = [0.5]
        return _D()

    def diagnose(self, segment, query=None, intent=None):
        self.diagnose_calls += 1

        class _Item:
            def __init__(self, index, verdict, reason, fix):
                self.index, self.verdict, self.reason, self.fix = index, verdict, reason, fix

        class _RItem:
            def __init__(self, text, is_hard):
                self.text, self.is_hard = text, is_hard

        class _Diag:
            scalar = 0.5
            overall_ok = False
            summary = 'one criterion failed'
            query = 'do it'
            segment_text = 'user: do it ...'
            raw = '1. FAIL: boom -> retry'
            rubric = [_RItem('args are valid JSON', True)]
            items = [_Item(0, False, 'tool errored', 'retry with valid args')]
        return _Diag()


def _low_quality_agent():
    # a tool call whose result errors -> low hard score -> not short-circuited,
    # so fuse_segment will actually reach for the rubric (unless gated).
    return {'messages': [
        {'role': 'user', 'content': 'do it'},
        {'role': 'assistant', 'content': '',
         'tool_calls': [{'id': 't', 'type': 'function',
                         'function': {'name': 'f', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 't', 'content': 'ERROR: boom'},
    ], 'user_data': []}


def test_gate_skips_rubric_for_unselected_row():
    from twinkle_agentic.preprocessor import TrajectoryScorer

    spy = _SpyRubric()
    scorer = TrajectoryScorer(rubric_verifier=spy, calibrate=False)

    gated = L.set_label(_low_quality_agent(), L.KEY_SELECTED_FOR_RUBRIC, False)
    scorer([gated])
    assert spy.calls == 0  # not selected -> no LLM rubric

    selected = L.set_label(_low_quality_agent(), L.KEY_SELECTED_FOR_RUBRIC, True)
    scorer([selected])
    assert spy.calls >= 1  # selected -> rubric runs


def test_persist_diagnosis_writes_verdict_reason_fix():
    from twinkle_agentic.preprocessor import TrajectoryScorer

    spy = _SpyRubric()
    scorer = TrajectoryScorer(rubric_verifier=spy, calibrate=False,
                              persist_diagnosis=True)

    selected = L.set_label(_low_quality_agent(), L.KEY_SELECTED_FOR_RUBRIC, True)
    out, _ = scorer([selected])
    diag = L.get_label(out[0], L.KEY_RUBRIC_DIAGNOSIS)
    assert spy.diagnose_calls >= 1              # diagnosis ran for the scored segment
    assert isinstance(diag, list) and diag       # persisted, one entry per segment
    entry = diag[0]
    assert entry['overall_ok'] is False
    assert entry['raw'] == '1. FAIL: boom -> retry'          # SFT target
    assert entry['segment_text'] and entry['query']          # SFT inputs
    assert entry['items'][0]['verdict'] is False
    assert entry['items'][0]['reason'] == 'tool errored'     # the "why"
    assert entry['items'][0]['fix'] == 'retry with valid args'
    assert entry['rubric'][0]['text'] == 'args are valid JSON'


def test_no_diagnosis_for_gated_out_row():
    from twinkle_agentic.preprocessor import TrajectoryScorer

    spy = _SpyRubric()
    scorer = TrajectoryScorer(rubric_verifier=spy, calibrate=False,
                              persist_diagnosis=True)
    gated = L.set_label(_low_quality_agent(), L.KEY_SELECTED_FOR_RUBRIC, False)
    out, _ = scorer([gated])
    assert spy.diagnose_calls == 0                            # gated -> no LLM at all
    assert L.get_label(out[0], L.KEY_RUBRIC_DIAGNOSIS) is None


class _SpySafety:
    """Stands in for the RubricVerifier a SafetyScorer holds."""
    fixed_rubric = None

    def __init__(self):
        self.calls = 0

    def score_detail(self, trajectory, **kwargs):
        self.calls += 1

        class _D:
            scalar = 1.0
        return _D()


def test_safety_gate_skips_llm_for_unselected_row():
    from twinkle_agentic.preprocessor import SafetyScorer

    spy = _SpySafety()
    scorer = SafetyScorer(rubric_verifier=spy, gate_label=L.KEY_SELECTED_FOR_RUBRIC)

    gated = L.set_label(_low_quality_agent(), L.KEY_SELECTED_FOR_RUBRIC, False)
    out, _ = scorer([gated])
    assert spy.calls == 0  # not selected -> no LLM safety pass
    assert L.get_label(out[0], L.KEY_SAFETY_SCORE) == 1.0  # tagged neutral-safe
    assert L.get_label(out[0], L.KEY_SAFETY_UNSAFE) is False

    selected = L.set_label(_low_quality_agent(), L.KEY_SELECTED_FOR_RUBRIC, True)
    scorer([selected])
    assert spy.calls >= 1  # selected -> safety LLM runs


def test_safety_no_gate_scores_every_row():
    from twinkle_agentic.preprocessor import SafetyScorer

    spy = _SpySafety()
    scorer = SafetyScorer(rubric_verifier=spy)  # gate_label=None -> score all
    scorer([L.set_label(_low_quality_agent(), L.KEY_SELECTED_FOR_RUBRIC, False)])
    assert spy.calls == 1  # ungated: LLM runs even for a "not selected" row
