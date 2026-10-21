# Copyright (c) ModelScope Contributors. All rights reserved.
"""The check kinds this package adds beyond the file_* / answer_* basics.

Each kind gets its validation tested as well as its verdict: a malformed task
should fail at build time, next to whoever wrote it, rather than at grading time
where the failure reads as a hard task.
"""
import json
import sqlite3

import pytest

from twinkle_agentic.envs import LocalEnv
from twinkle_agentic.verifier.result_check import Check, CheckContext, run_checks


def outcome(check, *, workspace='', final_answer='', env=None, receipts=()):
    """Run one check and return its outcome."""
    report = run_checks([check],
                        CheckContext(workspace=workspace,
                                     final_answer=final_answer,
                                     env=env,
                                     receipts=receipts))
    assert len(report.outcomes) == 1
    return report.outcomes[0]


# ---------------------------------------------------------------- answer_one_of


def test_answer_one_of_passes_on_any_listed_choice():
    check = Check(kind='answer_one_of', value=['yes', 'no', 'maybe'])
    assert outcome(check, final_answer=' no ').passed is True
    assert outcome(check, final_answer='maybe').passed is True


def test_answer_one_of_is_case_sensitive_like_the_other_answer_kinds():
    # _norm strips and does not casefold, and answer_contains / answer_equals
    # behave the same way. A task that wants case-insensitivity has to list the
    # spellings it accepts rather than have the verifier guess.
    check = Check(kind='answer_one_of', value=['yes', 'no'])
    assert outcome(check, final_answer='NO').passed is False
    both = Check(kind='answer_one_of', value=['yes', 'no', 'NO', 'No'])
    assert outcome(both, final_answer='NO').passed is True


def test_answer_one_of_fails_when_nothing_matches():
    check = Check(kind='answer_one_of', value=['yes', 'no'])
    result = outcome(check, final_answer='perhaps')
    assert result.passed is False
    assert 'perhaps' in result.detail


@pytest.mark.parametrize('value', [
    'yes',  # not a list
    [],  # empty
    None,
    [{'a': 1}],  # not scalar
])
def test_answer_one_of_rejects_a_malformed_choice_list(value):
    with pytest.raises(ValueError, match='non-empty list of scalar answers'):
        Check(kind='answer_one_of', value=value)


# --------------------------------------------------------------- answer_number


@pytest.mark.parametrize('answer,expected,kwargs,should_pass', [
    ('42', 42, {}, True),
    ('42.0', 42, {}, True),
    ('  42  ', 42, {}, True),
    ('-3.5', -3.5, {}, True),
    ('43', 42, {}, False),
    ('42.4', 42, {'abs_tol': 0.5}, True),
    ('42.6', 42, {'abs_tol': 0.5}, False),
    ('101', 100, {'rel_tol': 0.02}, True),
    ('103', 100, {'rel_tol': 0.02}, False),
    ('1e3', 1000, {}, True),
])
def test_answer_number_tolerance(answer, expected, kwargs, should_pass):
    check = Check(kind='answer_number', value=expected, **kwargs)
    assert outcome(check, final_answer=answer).passed is should_pass


def test_answer_number_ignores_a_trailing_unit_it_was_told_about():
    check = Check(kind='answer_number', value=42, unit='GB')
    assert outcome(check, final_answer='42 GB').passed is True


def test_answer_number_rejects_an_answer_that_is_not_a_number():
    check = Check(kind='answer_number', value=42)
    result = outcome(check, final_answer='forty-two')
    assert result.passed is False


@pytest.mark.parametrize('kwargs,message', [
    ({
        'value': 'x'
    }, 'must be numeric'),
    ({
        'value': True
    }, 'must be numeric'),
    ({
        'value': float('inf')
    }, 'must be finite'),
    ({
        'value': 1,
        'abs_tol': -1
    }, 'abs_tol must be finite and non-negative'),
    ({
        'value': 1,
        'rel_tol': float('nan')
    }, 'rel_tol must be finite and non-negative'),
    ({
        'value': 1,
        'unit': 5
    }, 'unit must be a string'),
])
def test_answer_number_validates_its_own_specification(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Check(kind='answer_number', **kwargs)


# -------------------------------------------------------------------- sql_query


@pytest.fixture
def db(tmp_path):
    path = tmp_path / 'ops.db'
    conn = sqlite3.connect(path)
    conn.execute('CREATE TABLE incidents (service TEXT, severity TEXT)')
    conn.executemany('INSERT INTO incidents VALUES (?, ?)', [('auth', 'p1'), ('billing', 'p2')])
    conn.commit()
    conn.close()
    return str(tmp_path)


def test_sql_query_passes_and_ignores_row_order_by_default(db):
    # SQL without ORDER BY has no guaranteed row order, so requiring one would
    # fail a correct solution on an accident of the planner.
    check = Check(kind='sql_query',
                  db_path='ops.db',
                  query='SELECT service, severity FROM incidents',
                  expect_rows=[['billing', 'p2'], ['auth', 'p1']])
    assert outcome(check, workspace=db).passed is True


def test_sql_query_ordered_does_care_about_row_order(db):
    rows = [['billing', 'p2'], ['auth', 'p1']]
    check = Check(kind='sql_query',
                  db_path='ops.db',
                  query='SELECT service, severity FROM incidents',
                  expect_rows=rows,
                  ordered=True)
    assert outcome(check, workspace=db).passed is False
    check.desc = True
    ordered = Check(kind='sql_query',
                    db_path='ops.db',
                    query='SELECT service, severity FROM incidents ORDER BY service DESC',
                    expect_rows=rows,
                    ordered=True)
    assert outcome(ordered, workspace=db).passed is True


def test_sql_query_fails_on_wrong_rows(db):
    check = Check(kind='sql_query',
                  db_path='ops.db',
                  query='SELECT service FROM incidents',
                  expect_rows=[['auth']])
    result = outcome(check, workspace=db)
    assert result.passed is False
    assert 'expected 1 row' in result.detail


def test_sql_query_cannot_write_to_the_database_it_grades(db):
    # Read-only by URI, so a task cannot pass by mutating the artifact during
    # verification -- that is not a solved task.
    check = Check(kind='sql_query',
                  db_path='ops.db',
                  query="INSERT INTO incidents VALUES ('x','y')",
                  expect_rows=[])
    result = outcome(check, workspace=db)
    assert result.passed is False
    assert 'query failed' in result.detail
    conn = sqlite3.connect(db + '/ops.db')
    assert conn.execute('SELECT COUNT(*) FROM incidents').fetchone()[0] == 2
    conn.close()


def test_sql_query_reports_a_missing_database_rather_than_raising(db):
    check = Check(kind='sql_query', db_path='nope.db', query='SELECT 1', expect_rows=[[1]])
    result = outcome(check, workspace=db)
    assert result.passed is False
    assert 'does not exist' in result.detail


def test_sql_query_refuses_a_path_outside_the_workspace(tmp_path, db):
    outside = tmp_path.parent / 'elsewhere.db'
    check = Check(kind='sql_query',
                  db_path=f'../{outside.name}',
                  query='SELECT 1',
                  expect_rows=[[1]])
    assert outcome(check, workspace=db).passed is False


@pytest.mark.parametrize('kwargs,message', [
    ({
        'query': '',
        'db_path': 'a.db',
        'expect_rows': []
    }, 'non-empty query'),
    ({
        'query': 'SELECT 1',
        'db_path': '',
        'expect_rows': []
    }, 'requires a db_path'),
    ({
        'query': 'SELECT 1',
        'db_path': 'a.db'
    }, 'requires expect_rows'),
    ({
        'query': 'SELECT 1',
        'db_path': 'a.db',
        'expect_rows': 'x'
    }, 'expect_rows must be a list'),
    ({
        'query': 'SELECT 1',
        'db_path': 'a.db',
        'expect_rows': [1]
    }, 'entries must be lists'),
])
def test_sql_query_validates_its_own_specification(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Check(kind='sql_query', **kwargs)


# ------------------------------------------------------------------ web_receipt


def _receipt(tool='web_search', urls=(), query='q'):
    return {'tool': tool, 'query': query, 'args': {},
            'urls': [{'url': u, 'published': p, 'title': 't'} for u, p in urls]}


def test_web_receipt_passes_when_the_cited_url_was_actually_served():
    check = Check(kind='web_receipt', value=['https://ex.com/a'])
    receipts = (_receipt(urls=[('https://ex.com/a', '2026-01-01')]), )
    assert outcome(check, receipts=receipts).passed is True


def test_web_receipt_fails_a_url_that_was_never_served():
    # A policy that assembles a plausible url from a site's naming scheme has
    # not retrieved anything, and without this check the two look identical.
    check = Check(kind='web_receipt', value=['https://ex.com/invented'])
    receipts = (_receipt(urls=[('https://ex.com/a', '2026-01-01')]), )
    result = outcome(check, receipts=receipts)
    assert result.passed is False
    assert 'invented' in result.detail


def test_web_receipt_fails_when_nothing_was_retrieved_at_all():
    check = Check(kind='web_receipt', value=['https://ex.com/a'])
    result = outcome(check, receipts=())
    assert result.passed is False


def test_web_receipt_refuses_an_answer_with_no_citations():
    # Passing vacuously would teach the policy that citing nothing is safe.
    check = Check(kind='web_receipt', value=[])
    assert outcome(check, receipts=(_receipt(urls=[('https://ex.com/a', '')]), )).passed is False


def test_web_receipt_enforces_a_minimum_publication_date():
    check = Check(kind='web_receipt', value=['https://ex.com/old'], min_published='2026-01-01')
    stale = (_receipt(urls=[('https://ex.com/old', '2019-05-05')]), )
    assert outcome(check, receipts=stale).passed is False
    fresh = (_receipt(urls=[('https://ex.com/old', '2026-06-06')]), )
    assert outcome(check, receipts=fresh).passed is True


def test_web_receipt_accepts_a_url_that_fetch_page_served():
    check = Check(kind='web_receipt', value=['https://ex.com/b'])
    receipts = (_receipt(tool='fetch_page', urls=[('https://ex.com/b', '')]), )
    assert outcome(check, receipts=receipts).passed is True


def test_web_receipt_fails_on_a_value_that_is_not_a_url_list():
    # Reported as a failed check with a usable message rather than raised: the
    # task was already built, and crashing the grading run over one bad check
    # would take the other checks down with it.
    result = outcome(Check(kind='web_receipt', value=42), receipts=())
    assert result.passed is False
    assert 'cited url' in result.detail


# ------------------------------------------------------------------------- lean


def test_lean_without_an_environment_fails_rather_than_guessing():
    # Compiling needs the toolchain, which lives in the sandbox. Grading locally
    # would report a failure that has nothing to do with the submission.
    check = Check(kind='lean', path='Proof.lean', theorem='my_thm')
    result = outcome(check, env=None)
    assert result.passed is False
    assert 'require an environment' in result.detail


@pytest.mark.parametrize('kwargs,message', [
    ({
        'path': '',
        'theorem': 'my_thm'
    }, 'requires path'),
    ({
        'path': 'Proof.lean',
        'theorem': ''
    }, 'requires theorem'),
])
def test_lean_validates_its_own_specification(kwargs, message):
    # Without a named declaration an empty file would compile and pass.
    with pytest.raises(ValueError, match=message):
        Check(kind='lean', **kwargs)


def test_lean_reports_a_missing_toolchain_as_a_failure_not_a_crash(tmp_path):
    check = Check(kind='lean', path='Proof.lean', theorem='my_thm')
    env = LocalEnv(workspace=str(tmp_path))
    result = outcome(check, workspace=str(tmp_path), env=env)
    assert result.passed is False
    assert isinstance(result.detail, str) and result.detail


@pytest.mark.parametrize('declaration,expected', [('fooXbar', False), ('foo.bar', True)])
def test_lean_declaration_names_are_literal_and_paths_are_quoted(tmp_path, declaration, expected):
    # Stub only the compiler; exercise the actual shell grader and declaration check.
    prefix = tmp_path / 'toolchain'
    (prefix / 'bin').mkdir(parents=True)
    lake = prefix / 'bin' / 'lake'
    lake.write_text('#!/bin/sh\nexit 0\n')
    lake.chmod(0o755)
    project = tmp_path / "project with 'quotes'"
    project.mkdir()
    (project / 'Proof with spaces.lean').write_text(f'theorem {declaration} : True := by trivial\n')
    env = LocalEnv(workspace=str(project), envs={'LEAN_PREFIX': str(prefix)})
    check = Check(kind='lean', path='Proof with spaces.lean', theorem='foo.bar', lean_project=str(project))
    assert outcome(check, env=env).passed is expected


def test_lean_missing_path_is_reported_without_shell_expansion(tmp_path):
    env = LocalEnv(workspace=str(tmp_path))
    check = Check(kind='lean', path='$(touch expanded).lean', theorem='t')
    result = outcome(check, env=env)
    assert not result.passed
    assert not (tmp_path / 'expanded').exists()


# --------------------------------------------------------------- hidden_python


def test_hidden_python_runs_in_the_environment_like_python_does(tmp_path):
    check = Check(kind='hidden_python', code='print(6 * 7)', expect_exit=0, value='42')
    env = LocalEnv(workspace=str(tmp_path))
    assert outcome(check, workspace=str(tmp_path), env=env).passed is True


def test_hidden_python_fails_on_a_non_zero_exit(tmp_path):
    check = Check(kind='hidden_python', code='raise SystemExit(3)', expect_exit=0)
    env = LocalEnv(workspace=str(tmp_path))
    result = outcome(check, workspace=str(tmp_path), env=env)
    assert result.passed is False
    assert 'exit 3' in result.detail


# --------------------------------------------------------------------- plumbing


def test_checks_from_dicts_accepts_the_new_fields():
    from twinkle_agentic.verifier.result_check import checks_from_dicts
    raw = [
        {
            'kind': 'answer_number',
            'value': 42,
            'abs_tol': 0.5,
            'unit': 'GB'
        },
        {
            'kind': 'web_receipt',
            'value': ['https://ex.com/a'],
            'min_published': '2026-01-01'
        },
    ]
    checks = checks_from_dicts(raw)
    assert checks[0].abs_tol == 0.5 and checks[0].unit == 'GB'
    assert checks[1].min_published == '2026-01-01'
    assert json.loads(json.dumps({'k': checks[0].kind}))['k'] == 'answer_number'


def test_an_unknown_kind_is_rejected_at_build_time():
    with pytest.raises(ValueError, match='unknown check kind'):
        Check(kind='not_a_kind')
