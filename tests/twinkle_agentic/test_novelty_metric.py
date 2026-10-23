# Copyright (c) ModelScope Contributors. All rights reserved.
"""A novelty score a program can compute, and the measurement that justifies it.

The point of the module is that a term constant across a GRPO group contributes
exactly zero gradient, so the tests that matter are the ones about variation
inside a group rather than about the absolute value of a score. ``spread`` and
``constant_group_rate`` need no scikit-learn and are tested unconditionally; the
scorer itself is skipped where scikit-learn is absent, which is also what the
module's own lazy import promises.
"""
import pytest

from twinkle_agentic.challenger import TaskBank
from twinkle_agentic.novelty_metric import NoveltyScorer, _union_find_components, constant_group_rate, spread

sklearn = pytest.importorskip('sklearn', reason='NoveltyScorer vectorises with scikit-learn')

# Eight siblings that share their scaffolding verbatim and differ in content --
# the shape a real proposal group has, and the shape that makes per-group IDF fit
# the boilerplate instead of the task.
SCAFFOLD = ('## Task:\nCreate the file `{name}` in the workspace.\n### Raw Input\n'
            'Read `{source}` and write the result to `{name}`.\n')
GROUP = [
    SCAFFOLD.format(name='out.json', source='orders.csv'),
    SCAFFOLD.format(name='out.json', source='orders.csv backup'),
    SCAFFOLD.format(name='out.json', source='orders.csv copy'),
    SCAFFOLD.format(name='summary.txt', source='incidents.log'),
    SCAFFOLD.format(name='summary.txt', source='incidents.log second'),
    SCAFFOLD.format(name='report.md', source='metrics.parquet'),
    SCAFFOLD.format(name='clean.csv', source='dirty.csv'),
    SCAFFOLD.format(name='index.json', source='docs corpus'),
]
CORPUS = GROUP + [
    SCAFFOLD.format(name='plan.md', source='roadmap.txt'),
    SCAFFOLD.format(name='schema.sql', source='legacy dump'),
    SCAFFOLD.format(name='diff.patch', source='upstream repo'),
]


# --------------------------------------------------------------- no dependencies
def test_spread_is_max_minus_min_and_zero_for_a_single_value():
    assert spread([0.2, 0.9, 0.5]) == pytest.approx(0.7)
    assert spread([0.4]) == 0.0
    assert spread([]) == 0.0


def test_spread_ignores_the_nones_a_judge_leaves_behind():
    # An unparseable verdict is recorded as None, and a group of seven Nones and
    # one score has a spread of zero, not of the one score.
    assert spread([None, None, 0.5, None]) == 0.0


def test_constant_group_rate_counts_the_groups_grpo_cannot_see():
    groups = [[0.0, 0.0, 0.0], [0.1, 0.9, 0.4], [0.5, 0.5], [0.25]]
    # The group of one is skipped, not counted as constant: it has no siblings to
    # differ from, so it says nothing about the metric.
    assert constant_group_rate(groups) == pytest.approx(2 / 3)


def test_constant_group_rate_of_nothing_is_zero_not_a_division():
    assert constant_group_rate([]) == 0.0
    assert constant_group_rate([[0.5]]) == 0.0


def test_union_find_reports_the_size_of_each_component_per_node():
    assert _union_find_components(4, [(0, 1), (1, 2)]) == [3, 3, 3, 1]
    assert _union_find_components(3, []) == [1, 1, 1]


# --------------------------------------------------------------------- the scorer
def test_an_empty_corpus_is_refused_rather_than_fitted():
    with pytest.raises(ValueError, match='at least one non-empty statement'):
        NoveltyScorer(['', '   ', None])


def test_scores_come_back_in_input_order_and_one_per_statement():
    scorer = NoveltyScorer(CORPUS, cluster_threshold=0.5)
    scores = scorer.score(GROUP)
    assert len(scores) == len(GROUP)
    assert all(0.0 <= value <= 1.0 for value in scores)


def test_the_degenerate_inputs_have_the_documented_answers():
    scorer = NoveltyScorer(CORPUS, cluster_threshold=0.5)
    assert scorer.score([]) == []
    # A group of one has nothing to differ from, and 1.0 keeps it from reading as
    # "this proposal is a duplicate of something".
    assert scorer.score([GROUP[0]]) == [1.0]


def test_a_near_duplicate_cluster_scores_below_the_odd_one_out():
    # Three siblings that differ by one word, against one that does not. The
    # spread term is what separates them: all four have a low max-cosine to
    # *something*, but only the three form a cluster.
    statements = [
        SCAFFOLD.format(name='out.json', source='orders.csv'),
        SCAFFOLD.format(name='out.json', source='orders.csv backup'),
        SCAFFOLD.format(name='out.json', source='orders.csv copy'),
        SCAFFOLD.format(name='schema.sql', source='a completely unrelated legacy dump'),
    ]
    scorer = NoveltyScorer(CORPUS, cluster_threshold=0.3)
    scores = scorer.score(statements)
    assert scores[3] > max(scores[:3])


def test_identical_siblings_are_indistinguishable_and_the_scorer_says_so():
    # The failure being replaced: a group where every member scores the same
    # contributes no gradient at all. Asserted as the property, not as a number.
    scorer = NoveltyScorer(CORPUS, cluster_threshold=0.5)
    same = ['Write the given file verbatim.'] * 8
    assert spread(scorer.score(same)) == 0.0
    assert constant_group_rate([scorer.score(same)]) == 1.0


def test_a_real_group_is_not_constant_which_is_the_whole_point():
    scorer = NoveltyScorer(CORPUS, cluster_threshold=0.3)
    scores = scorer.score(GROUP)
    assert spread(scores) > 0.0
    assert constant_group_rate([scores]) == 0.0


def test_the_threshold_can_be_calibrated_from_sibling_groups():
    calibration = [GROUP[:4], GROUP[4:]]
    scorer = NoveltyScorer(CORPUS, calibration_groups=calibration, cluster_quantile=0.75)
    assert scorer.calibrated is not None
    assert scorer.cluster_threshold == pytest.approx(scorer.calibrated['threshold'])
    assert scorer.calibrated['quantile'] == 0.75
    assert scorer.calibrated['pairs'] == 6 + 6


def test_without_calibration_groups_the_threshold_falls_back():
    scorer = NoveltyScorer(CORPUS)
    assert scorer.cluster_threshold == 0.5
    assert scorer.calibrated is None


def test_an_explicit_threshold_is_not_overridden_by_calibration():
    scorer = NoveltyScorer(CORPUS, cluster_threshold=0.9, calibration_groups=[GROUP[:4]])
    assert scorer.cluster_threshold == 0.9
    assert scorer.calibrated is None


def test_boilerplate_every_statement_shares_carries_almost_no_weight():
    # The load-bearing claim about fitting on the corpus rather than the group.
    # `task` opens every statement in it, `parquet` appears in one, so IDF has to
    # rank the second above the first. Fitted per group the order inverts and
    # every pair of siblings looks like a duplicate of the scaffolding.
    scorer = NoveltyScorer(CORPUS, cluster_threshold=0.3)
    idf = dict(zip(scorer.vectoriser.get_feature_names_out(), scorer.vectoriser.idf_))
    assert idf['task'] < idf['parquet']


def test_words_the_corpus_never_used_are_invisible_to_the_score():
    # Documented in the module and asserted here so it cannot change quietly:
    # these two differ in content and share only scaffolding, but neither
    # `payroll` nor `orbit` is in the fitted vocabulary, so both reduce to the
    # same vector and the group scores as eight copies of one idea.
    scorer = NoveltyScorer(CORPUS, cluster_threshold=0.3)
    outside = [
        '## Task:\nCreate the file `a`.\n### Raw Input\nParse the payroll ledger.',
        '## Task:\nCreate the file `b`.\n### Raw Input\nRender the orbit plot.',
    ]
    assert spread(scorer.score(outside)) == 0.0
    assert constant_group_rate([scorer.score(outside)]) == 1.0


def test_the_weights_move_the_score():
    statements = GROUP[:4]
    only_distance = NoveltyScorer(CORPUS, cluster_threshold=0.3, weight_distance=1.0, weight_spread=0.0)
    only_spread = NoveltyScorer(CORPUS, cluster_threshold=0.3, weight_distance=0.0, weight_spread=1.0)
    assert only_distance.score(statements) != only_spread.score(statements)


# ------------------------------------------------------------------- integration
def test_a_task_bank_supplies_the_calibration_groups(tmp_path):
    # sibling_groups exists for this and nothing else, so the two have to fit
    # together without an adapter.
    bank = TaskBank(str(tmp_path / 'bank.jsonl'))
    for index, statement in enumerate(GROUP):
        bank.add(statement, 'assert 1', group_id=index // 4)
    groups = bank.sibling_groups(minimum=2)
    assert len(groups) == 2
    scorer = NoveltyScorer(bank.statements(), calibration_groups=groups)
    assert scorer.calibrated is not None
    assert scorer.calibrated['pairs'] == 6 + 6
