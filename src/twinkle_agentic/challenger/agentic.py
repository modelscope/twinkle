# Copyright (c) ModelScope Contributors. All rights reserved.
"""Agentic challenger: invent tasks by doing them first.

The approach mirrors how the code challenger works, adapted to tool-using
agents. Instead of writing a problem statement and hoping it is achievable, the
model first *does* something in a sandbox, then -- in the same conversation --
writes the check script that verifies the end state it just produced, then the
problem statement someone else would need to reproduce it.

Steps for one candidate:

    1. Choose direction + keywords. Optionally start from a seed trajectory.
    2. Explore (multi-turn with tools): model acts in a clean sandbox, producing
       a tool-call chain and a final workspace state, and stops calling tools.
    3. A user message is appended to that same conversation carrying the
       workspace listing, asking for a python check script. Tools are no longer
       dispatched from here on.
    4. Verify: run the check script in the sandbox (must pass).
    5. A second user message is appended asking for the problem statement.
    6. Difficulty filter: reset workspace, let the solver do the task N times,
       run checks, keep only "sometimes pass" tasks.

Steps 3 and 5 are appended to the episode rather than sent as fresh calls, so
every assistant turn in the chain keeps its ``labels`` and ``logprobs`` and the
whole proposal -- acting, checking, describing -- is one trainable sample. The
follow-up messages come back from :meth:`AgenticChallenger._followup`, which the
rollout calls at the moment the model stops calling tools; that is where the
sandbox work (snapshot, running the check) happens, because only the caller can
do it.

Because every episode needs a clean workspace and because episodes share a
single long-lived sandbox, proposing is **serial** -- one proposal at a time with
a workspace reset in between.

Prompt text is not here. Every string the model sees arrives in
:class:`AgenticPrompts`, built by whoever runs the challenger -- see
``cookbook/rsi/agentic/prompts.py``.
"""
import ast
import json
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from twinkle.data_format import SamplingParams, Trajectory, user_data_get
from twinkle.utils import get_logger
from .base import Challenger, Explorer, assistant_text, attach_user_data
from .code import KeywordStore, parse_keyword_list

logger = get_logger()

__all__ = [
    'AgenticChallenger',
    'AgenticPrompts',
    'parse_check_script',
    'parse_problem_statement',
]

_FENCE_RE = re.compile(r'```(?:python|py)?\s*\n(.*?)```', re.S)
# A fence around the *whole* reply, which is packaging rather than content.
_WHOLE_FENCE_RE = re.compile(r'```[\w+-]*\s*\n?(.*?)```', re.S)
# The JSON body of a tool call: the proposing episode uses tools, so at the check
# stage a 4B model often keeps calling one instead of writing a fenced block.
_TOOLCALL_RE = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.S)


# ── parsing ───────────────────────────────────────────────────────────────

def parse_check_script(text: str) -> Optional[str]:
    """Extract a python check script from the model's reply.

    Prefers the last fenced python block after ``</think>``. When the reply has
    no fence at all, falls back to reading the tail as bare code: 8 of
    armA2shellV6's 11 check_parse_fail rejections were a complete, parseable
    check script that the model simply did not wrap in backticks, and throwing
    the task away over the packaging loses a task that was ready.

    Returns ``None`` when nothing usable is found.
    """
    body = text or ''
    idx = body.rfind('</think>')
    if idx >= 0:
        body = body[idx + len('</think>'):]
    blocks = _FENCE_RE.findall(body)
    if blocks:
        script = blocks[-1].strip()
        return script if script else None
    bare = _bare_check_script(body)
    if bare:
        return bare
    return _toolcall_check_script(body)


def _bare_check_script(body: str) -> Optional[str]:
    """Read an unfenced reply as code, or None.

    Advances the start line until the rest parses, which drops whatever prose
    came first (the "ALSO CORRECT:" line, a sentence introducing the script)
    without needing to recognise it. Requires an ``assert`` so that a one-line
    reply of prose -- which can be a syntactically valid expression -- is not
    mistaken for a check.
    """
    lines = body.strip().split('\n')
    for start in range(len(lines)):
        cand = '\n'.join(lines[start:]).strip()
        if 'assert' not in cand:
            break  # no assert left in the tail; nothing further can qualify
        try:
            ast.parse(cand)
        except SyntaxError:
            continue
        return cand
    return None


def _toolcall_check_script(body: str) -> Optional[str]:
    """Recover a check script the model put inside a tool call, or None.

    The proposing episode uses tools, and at the check stage a 4B model often
    keeps calling one -- it emits ``python_executor(code="...assert...")`` (or a
    shell command, or ``write_file(content=...)``) instead of a fenced block.
    The script is right there in the call's ``code``/``command``/``content``
    argument, so pull it out rather than lose the task: measured on run_clean1,
    most first-round check_parse_fail rejections were tool-call wrapped.

    Only code that parses and actually asserts is accepted, so a shell
    ``command`` that merely runs a file -- which has no assert of its own --
    does not slip through as a check.
    """
    blobs = _TOOLCALL_RE.findall(body)
    for blob in reversed(blobs):
        code = None
        try:
            obj = json.loads(blob)
            args = obj.get('arguments') if isinstance(obj, dict) else None
            if isinstance(args, dict):
                code = args.get('code') or args.get('command') or args.get('content')
        except (ValueError, AttributeError):
            m = re.search(r'"(?:code|command|content)"\s*:\s*"(.*?)"\s*\}', blob,
                          re.S)
            if m:
                try:
                    code = m.group(1).encode().decode('unicode_escape')
                except (UnicodeDecodeError, ValueError):
                    code = None
        if not code or 'assert' not in code:
            continue
        try:
            ast.parse(code)
        except SyntaxError:
            continue
        return code.strip()
    return None



# Two rules CHECK_FOLLOWUP already states -- no equality on a script's source
# text, no byte count or checksum on a binary -- were broken by 9 and 8 of 41
# measured tasks respectively, so stating them a third time is not the fix. A
# check that pins the exact source of a .py rejects every equivalent solution,
# and one that pins a .png's byte count rejects every matplotlib version; both
# make a task nobody but the author can pass.
_SIZE_OR_HASH_NAMES = ('getsize', 'st_size', 'sha256', 'sha1', 'md5', 'hexdigest',
                       'digest')
# What makes a string python rather than data. Checked instead of "is it long and
# multi-line", because the contents of a csv or a json file are legitimately
# asserted verbatim -- the statement handed those to the solver -- while the text
# of a script never is.
_LOOKS_LIKE_PYTHON = ('import ', 'def ', 'print(', 'with open(', 'if __name__')


def brittle_check_reason(script: str) -> Optional[str]:
    """Why this check script would reject a correct solution, or None.

    Returned text goes back to the model through the same retry path a failing
    assertion uses, because the defect is the same kind: an assertion that does
    not hold for solutions other than the one in front of it.

    Read off the syntax tree rather than matched as text. Both defects survive
    patterns easily: source equality reads the file into a name first
    (``c = f.read()``, then ``assert c == '...'``) so nothing sits between
    ``open()`` and ``==``, and a size check can put the call either around the
    name (``getsize("a.png")``) or after it.
    """
    try:
        tree = ast.parse(script)
    except SyntaxError:
        # Unparseable means it cannot run either, so let the sandbox report it.
        return None
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Compare)
                and any(isinstance(o, ast.Eq) for o in node.ops)):
            continue
        for side in [node.left] + list(node.comparators):
            if not (isinstance(side, ast.Constant) and isinstance(side.value, str)):
                continue
            if any(m in side.value for m in _LOOKS_LIKE_PYTHON):
                return ('AssertionError: this check compares a file against the '
                        'full text of a python script with ==, which only the '
                        'exact script you wrote can pass. Assert what running '
                        'that script produces instead.')
    # A byte count or a checksum compared for equality. Not restricted to
    # binary suffixes: CHECK_FOLLOWUP says "NEVER assert a file size in bytes"
    # about any file, and keying on a suffix list let
    # ``getsize('data.mat') == 264`` through. Only equality against a literal is
    # a defect -- ``getsize(f) > 0`` is a fine way to say "not empty".
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Compare)
                and any(isinstance(o, ast.Eq) for o in node.ops)):
            continue
        sides = [node.left] + list(node.comparators)
        has_literal = any(isinstance(s, ast.Constant)
                          and isinstance(s.value, (int, float, str))
                          and not isinstance(s.value, bool) for s in sides)
        if not has_literal:
            continue
        for side in sides:
            names = {n.attr for n in ast.walk(side) if isinstance(n, ast.Attribute)}
            names |= {n.id for n in ast.walk(side) if isinstance(n, ast.Name)}
            hit = names & set(_SIZE_OR_HASH_NAMES)
            if hit:
                what = ('a checksum' if hit - {'getsize', 'st_size'}
                        else 'a byte count')
                return (f'AssertionError: this check pins {what} of a file, and '
                        'correct solutions differ there. Assert what can be read '
                        'out of the file instead -- its structure, or the values '
                        'inside it.')
    # Comparing raw bytes of a file: same defect, different spelling.
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Compare)
                and any(isinstance(o, ast.Eq) for o in node.ops)):
            continue
        for side in [node.left] + list(node.comparators):
            if isinstance(side, ast.Constant) and isinstance(side.value, bytes):
                return ('AssertionError: this check compares the raw bytes of a '
                        'file, and correct solutions differ there. Assert what '
                        'can be read out of it instead.')
    return None


# Literals shorter than this match by accident: a statement contains "3" or "id"
# for its own reasons. Measured on 188 tasks from run_clean9, one and two digit
# integers appeared in both the check and the statement 91% of the time, which is
# what an unattributable coincidence rate looks like.
_MIN_DERIVED_LEN = 3


def derived_check_literals(script: str) -> List[str]:
    """The values a check compares against that the solver is meant to work out.

    A measurement tool, not part of the proposing path. It exists because "does the
    statement give away the answer" cannot be asked without first separating the
    three kinds of thing a check's literals are, and only one of them is a leak:

      an identifier, or a name with a file extension
            The statement MUST carry these. It is naming the file to create and the
            fields to put in it; a statement that withheld them would describe no
            particular output at all. Present in 84-93% of run_clean9's statements,
            which is the correct rate.
      text that appears in the workspace
            Input data, which the statement is meant to quote verbatim so the solver
            can write the same bytes. Not separated here -- the caller filters on
            the snapshot if it wants to.
      a long number, a float, or a string that is none of the above
            Only exists once the work has been done. This is the group returned.

    Feeding the result back to the statement stage as a forbidden list was tried and
    did not reduce the leak: see the note above PROBLEM_FOLLOWUP_RULES_ONLY in
    cookbook/rsi/agentic/prompts.py for the two forms measured and their p-values.

    Wrong at the edges by construction: a column named ``total_2024`` reads as an
    identifier and is not returned, and a computed value that lands on two digits is
    below the length floor.
    """
    try:
        tree = ast.parse(script)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        for side in [node.left] + list(node.comparators):
            if not isinstance(side, ast.Constant):
                continue
            v = side.value
            if isinstance(v, bool) or v is None:
                continue
            if isinstance(v, (int, float)):
                text = repr(v)
                if len(text.lstrip('-').replace('.', '')) < _MIN_DERIVED_LEN:
                    continue
                out.append(text)
            elif isinstance(v, str):
                if len(v) < _MIN_DERIVED_LEN:
                    continue
                if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', v):
                    continue
                # A trailing extension means a filename -- but only when the part
                # after the dot is not itself digits. '0.001' read out of a CSV is a
                # string here, and skipping it as a filename would let exactly the
                # kind of value this function exists to catch through.
                if re.search(r'\.[A-Za-z]\w{0,4}$', v) and ' ' not in v:
                    continue
                out.append(v)
    # Longest first: a short literal is often a substring of a longer one, and
    # naming the long one first makes the list read as distinct values rather than
    # as prefixes of each other.
    return sorted(set(out), key=len, reverse=True)



def parse_problem_statement(text: str) -> Optional[str]:
    """Extract a problem statement from the model's reply.

    Everything after ``</think>`` is the statement. A fence around the whole
    reply is unwrapped; fences *inside* it are kept.

    Keeping them matters more than it sounds: a statement that says what a file
    must contain puts the content in a fence, and stripping every fence left
    "1. `data.json` containing:" with nothing after it. 7 of ex11's 16 measured
    statements had a fence, and 5 of those 7 were solved 0 times out of 8 --
    against 1 of the 9 statements that had no fence to lose. The tasks were not
    hard, they were unanswerable.

    Returns ``None`` when the result is empty.
    """
    body = text or ''
    idx = body.rfind('</think>')
    if idx >= 0:
        body = body[idx + len('</think>'):]
    body = body.strip()
    whole = _WHOLE_FENCE_RE.fullmatch(body)
    if whole:
        body = whole.group(1).strip()
    return body if body else None


# The fields a local rollout splices into a trajectory, and the only ones a
# later GRPO step needs: ``labels`` marks which of ``input_ids`` are trainable
# (-100 elsewhere) and ``logprobs`` holds one entry per trainable token, taken
# from the policy that actually generated it.
_TRAINABLE_KEYS = ('input_ids', 'labels', 'logprobs')


def _propose_round(stage: str, trajectory: Trajectory) -> Dict[str, Any]:
    """One proposing round, reduced to what a later training step would read.

    ``messages`` comes along for reading by humans; it is redundant with
    ``input_ids`` and is not what a trainer should encode from.
    """
    record: Dict[str, Any] = {
        'stage': stage,
        'messages': [dict(m) for m in trajectory.get('messages') or []],
    }
    for key in _TRAINABLE_KEYS:
        value = trajectory.get(key)
        if value is not None:
            record[key] = value
    return record


# ── prompts ────────────────────────────────────────────────────────────────

@dataclass
class AgenticPrompts:
    """Every string an :class:`AgenticChallenger` sends.

    All fields are injected by the caller (no defaults with real text here).
    Placeholder validation happens at construction time.
    """

    # Explore: model acts in sandbox
    system: str
    from_scratch: str
    from_seed: str = ''
    from_keywords: str = ''
    from_seed_keywords: str = ''

    # Appended to the same conversation once the model stops calling tools:
    # first "write the check script" (which carries the workspace listing), then
    # "write the problem statement". Each has to repeat the rules that used to
    # live in a system message of its own, because there is no second system
    # message in a single conversation.
    check_followup: str = ''
    # Sent instead of the statement stage when the check script does not pass, so
    # the model can fix it from the traceback. Required only when the challenger
    # is built with ``check_retries`` above 0.
    check_retry_followup: str = ''
    problem_followup: str = ''

    # Keyword generation (same structure as code side)
    keyword_system: str = ''
    keyword_user: str = ''
    keyword_expand_user: str = ''

    _REQUIRED_FIELDS = {
        'from_seed': ('seed',),
        'from_keywords': ('keywords',),
        'from_seed_keywords': ('seed', 'keywords'),
        'check_followup': ('final_state',),
        'check_retry_followup': ('error', 'final_state'),
        'keyword_user': ('k', 'desc'),
        'keyword_expand_user': ('kw', 'm'),
    }

    def __post_init__(self):
        for name in ('system', 'from_scratch', 'check_followup', 'problem_followup'):
            if not getattr(self, name).strip():
                raise ValueError(f'AgenticPrompts.{name} is required')
        for name, placeholders in self._REQUIRED_FIELDS.items():
            text = getattr(self, name)
            if not text:
                continue
            for placeholder in placeholders:
                if '{' + placeholder + '}' not in text:
                    raise ValueError(f'AgenticPrompts.{name} must contain '
                                     f'{{{placeholder}}}')

    def require(self, *names: str) -> None:
        """Raise unless every named prompt was supplied."""
        missing = [n for n in names if not getattr(self, n).strip()]
        if missing:
            raise ValueError(f'this configuration needs AgenticPrompts.'
                             f'{", AgenticPrompts.".join(missing)}')


# ── challenger ─────────────────────────────────────────────────────────────

class AgenticChallenger(Challenger):
    """Propose tool-using tasks by first doing them, then describing them.

    Args:
        prompts: every string sent to the model.
        explorer: batch-in / batch-out generation with sandbox tools (multi-turn).
        seeds: optional pool of seed trajectories (dicts with a ``query`` key),
            drawn with replacement.
        keyword_store: optional bank for diversity control.
        category_desc: category -> description for keyword generation.
        seed_mix_prob: chance a proposal carries a seed.
        reset_fn: called before each round-1 episode to clean the sandbox
            workspace. Must be synchronous and leave the workspace empty.
        run_check_fn: run a python script in the sandbox's current state.
            Signature: ``(source: str) -> (exit_code: int, output: str)``.
        workspace_snapshot_fn: after round 1, return a text summary of the
            workspace state (e.g. ``find . -type f``). If None, a default
            that lists messages is used.
        tool_schemas: the executor's tool schemas, in the OpenAI shape the
            template renders. Attached to the trajectories that are *meant* to
            call tools -- the exploring episode and each solve attempt. Without
            this the model is never told the tool names, so it writes code in
            prose instead of calling anything: the workspace stays empty, every
            check fails, and the difficulty numbers describe a model that had no
            tools rather than a hard task. The check-writing and
            problem-writing stages sit in the same conversation and so see the
            same list, which is why the rollout stops dispatching calls once a
            follow-up has been appended -- a python block written as an *answer*
            parses as a call list, and 41 of 146 such replies in a measured run
            edited the very workspace the answer was about.
        combo_arity: ``'triple'`` or ``'mix'``, as in :class:`.CodeChallenger`.
        arity_weights: weights for the ``'mix'`` subset size.
        single_kw_prob: chance of using one category in ``'triple'`` mode.
        proposals_per_group: how many proposals answer the same keyword draw and
            the same prompt, tagged with a shared ``group_id``. This is the group
            size the proposing side's advantage is computed over; at 1 every
            group has one member and every advantage is zero. At a fixed
            proposal count it does not change the compute -- it divides the
            number of distinct keyword draws per round by the same factor.
        keyword_refill_target / keyword_gen_calls / keyword_refill_tries /
        keyword_params: keyword bank refill parameters.
        check_params / problem_params: sampling params for the two appended
            stages. ``None`` keeps whatever the episode was already using, which
            is sized for one agent turn; the check-writing stage reads the whole
            episode plus the end state and reasons at length before answering,
            and one that runs out of budget mid-thought never emits its code
            block and is thrown away as unparseable.
        followup_api: optional OpenAI-compatible API client (e.g. qwen3.8-max). When
            given, exploration still runs on the local explorer -- so its turns keep
            their ``labels`` and ``logprobs`` and remain trainable -- but the
            check-script (success judgement) and problem-statement stages are
            generated by this API instead of the local model, and are appended
            neither to the trainable trajectory nor to its token stream. This is
            the "explore locally, judge and describe over an API, train only the
            exploration" split. ``None`` keeps the single-model behaviour where the
            local model writes those two stages in the same conversation.
        followup_extra_body: extra request body forwarded on every ``followup_api``
            call (e.g. ``{'thinking_budget': N}`` to cap qwen3.8-max reasoning).
            ``None`` sends the request unmodified. Ignored when ``followup_api`` is
            ``None``.
        keyword_explorer: explorer used to brainstorm keywords. Should have no
            tools wired to it: a list is a text answer, and a bracketed list in
            a reply is exactly what the sandbox explorer would try to dispatch as
            a call. ``None`` reuses the main explorer, which for a sandbox setup
            means its tools are live there too.
        min_batch: smallest batch worth sending to the explorer.
        problem_max_chars: reject problem statements longer than this.
        check_retries: how many times a check script that did not pass is handed
            back, with the traceback and the workspace listing, for a rewrite
            before the proposal is rejected. 0 restores the old behaviour of
            rejecting on the first failure. Measured in ex12: 36 of 72 proposals
            died on the check, and 29 of those were a single assertion naming a
            value the model had not read -- a row count, a content string that
            was nearly right, a timestamp -- over a workspace state that was
            perfectly good. Each retry costs one more sampling call for that
            episode and nothing for the ones that pass first time.
        reject_sink: called with a dict for every rejected proposal.
        propose_sink: called once per proposal attempt -- kept, rejected while
            building, or dropped by the difficulty band alike -- with the
            token-level record of the episode that produced it. This is the only
            way the proposing episode survives: it is generation like any
            other, so it carries ``input_ids`` / ``labels`` / ``logprobs`` and
            could later be trained on, but nothing downstream of ``build``
            looks at it and without a sink it is dropped on the floor.
            Rejects are included on purpose: they are the zero-reward half of a
            GRPO group, so a set of kept-only records has no variance to learn
            from. Requires a local sampler -- an API explorer returns text only.
        solver_sink: called once per solver attempt in the difficulty stage, with
            the statement, the check script, the attempt, the workspace it left
            and the check's verdict. ``n_pass`` alone cannot distinguish a task
            that is impossible from one whose statement withholds a value its
            check demands, and both look like a hard task worth keeping.
        keyword_sink: called once per keyword-generation call, with the prompt,
            the raw reply and what ``parse_keyword_list`` made of it. A bank that
            refuses to fill is invisible otherwise -- proposals fall back to the
            no-keyword prompt and the run carries on looking normal -- and a count
            of zero does not say whether the model broke the format or the parser
            rejected output that was fine.
    """

    def __init__(
        self,
        prompts: AgenticPrompts,
        explorer: Explorer,
        *,
        seeds: Sequence[Dict[str, Any]] = (),
        keyword_store: Optional[KeywordStore] = None,
        category_desc: Optional[Dict[str, str]] = None,
        seed_mix_prob: float = 0.5,
        reset_fn: Callable[..., None],
        run_check_fn: Callable[..., Tuple[int, str]],
        workspace_snapshot_fn: Optional[Callable[..., str]] = None,
        snapshot_error_fn: Optional[Callable[..., str]] = None,
        tool_schemas: Optional[Sequence[Dict[str, Any]]] = None,
        episode_concurrency: int = 1,
        episode_tool_managers: Optional[Sequence[Any]] = None,
        combo_arity: str = 'triple',
        arity_weights: Optional[Sequence[float]] = None,
        single_kw_prob: float = 0.1,
        proposals_per_group: int = 1,
        keyword_refill_target: int = 128,
        keyword_gen_calls: int = 8,
        keyword_refill_concurrency: int = 1,
        keyword_refill_tries: int = 2,
        keyword_params: Optional[SamplingParams] = None,
        check_params: Optional[SamplingParams] = None,
        problem_params: Optional[SamplingParams] = None,
        followup_api: Optional[Any] = None,
        followup_extra_body: Optional[Dict[str, Any]] = None,
        keyword_explorer: Optional[Explorer] = None,
        min_batch: int = 1,
        problem_max_chars: int = 8192,
        max_proposals_total: int = 0,
        setup_script_fn: Optional[Callable[..., str]] = None,
        solver_prompt_fn: Optional[Callable[[str], Trajectory]] = None,
        check_retries: int = 1,
        task_bank: Optional[Any] = None,
        novelty_fn: Optional[Callable[[List[Dict[str, Any]]], List[Optional[float]]]] = None,
        novelty_floor: float = 0.5,
        keep_per_group: int = 0,
        reject_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        propose_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        solver_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        keyword_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        **challenger_kwargs: Any,
    ):
        super().__init__(explorer, system=prompts.system, **challenger_kwargs)
        if combo_arity not in ('triple', 'mix'):
            raise ValueError(f"combo_arity must be 'triple' or 'mix', got {combo_arity!r}")
        if keyword_store is not None:
            desc = category_desc or {}
            missing_cats = [c for c in keyword_store.categories if not desc.get(c)]
            if missing_cats:
                raise ValueError(f'category_desc is missing a description for '
                                 f'{missing_cats}; a dry category could not be refilled.')
            prompts.require('keyword_system', 'keyword_user', 'from_keywords')
        self.prompts = prompts
        self.seeds = list(seeds)
        self.store = keyword_store
        self.category_desc = dict(category_desc or {})
        self.seed_mix_prob = seed_mix_prob
        self.reset_fn = reset_fn
        self.run_check_fn = run_check_fn
        self.workspace_snapshot_fn = workspace_snapshot_fn
        # Asked, when a snapshot came back empty, why: the text of the failure if
        # the listing could not be read, '' if the workspace really was empty.
        # Without it the two are one outcome, and a sandbox the host had paused is
        # filed as the model having built nothing -- 63 of run_clean6's 71
        # ``empty_workspace`` rejections were the 410 "sandbox is not proxyable"
        # error, so that reject class was 89% broken environment.
        self.snapshot_error_fn = snapshot_error_fn
        self.tool_schemas = list(tool_schemas) if tool_schemas else None
        # More than one episode at a time needs more than one sandbox: an episode
        # owns its workspace from the reset until its check has run. The three
        # sandbox callables above are then called with ``slot=i`` to say which one,
        # and ``episode_tool_managers[i]`` must dispatch tool calls into that same
        # sandbox -- an episode acting in one workspace and checking another
        # produces a task whose check nobody can pass.
        if episode_concurrency < 1:
            raise ValueError(f'episode_concurrency must be >= 1, got {episode_concurrency}')
        if episode_concurrency > 1:
            if not episode_tool_managers or len(episode_tool_managers) != episode_concurrency:
                raise ValueError(
                    f'episode_concurrency={episode_concurrency} needs exactly that many '
                    f'episode_tool_managers, one per sandbox; got '
                    f'{len(episode_tool_managers) if episode_tool_managers else 0}.')
        self.episode_concurrency = episode_concurrency
        self.episode_tool_managers = (list(episode_tool_managers)
                                      if episode_tool_managers else None)
        # Held while writing to the dump files and while bumping ``stats``: with
        # concurrent episodes those are the only shared mutable things the
        # follow-up callback touches, and a half-written json line is unreadable.
        self._sink_lock = threading.Lock()
        # Separate from the sink lock: the keyword path holds this while it draws
        # from the shared rng and bumps the prompt nonce, and it must not be held
        # while a sink write is waiting on disk.
        self._kw_lock = threading.Lock()
        self.combo_arity = combo_arity
        self.arity_weights = list(arity_weights) if arity_weights else None
        self.single_kw_prob = single_kw_prob
        # How many proposals answer each keyword draw. Above 1 they form a GRPO
        # group on the proposing side; see :meth:`propose`. Raising it does not
        # cost more compute at a fixed proposal count -- it trades keyword
        # variety for group size, since a round's proposals then come from
        # ``count / proposals_per_group`` draws instead of ``count`` of them.
        if proposals_per_group < 1:
            raise ValueError(f'proposals_per_group must be >= 1, got {proposals_per_group}')
        self.proposals_per_group = proposals_per_group
        self._next_group_id = 0
        self.keyword_refill_target = keyword_refill_target
        self.keyword_gen_calls = keyword_gen_calls
        # How many of a refill's generating calls go out together. At 1 each call
        # is told what the ones before it produced, which is the point; raising it
        # is what the first round of arm measurements ran with, where a whole
        # first refill went out at once with nothing yet to avoid and came back
        # with synonyms. Kept configurable so the two can be compared on one build
        # rather than across two versions of this file.
        if keyword_refill_concurrency < 1:
            raise ValueError('keyword_refill_concurrency must be >= 1, got '
                             f'{keyword_refill_concurrency}')
        self.keyword_refill_concurrency = keyword_refill_concurrency
        self.keyword_refill_tries = keyword_refill_tries
        self.keyword_params = keyword_params
        self.check_params = check_params
        self.problem_params = problem_params
        # When set, exploration runs on the (trainable) local explorer as before,
        # but the check-script and problem-statement stages are generated by this
        # OpenAI-compatible API (e.g. qwen3.8-max) instead of the local model. The
        # two stages then contribute nothing to the trainable trajectory: the API
        # returns text only, so the episode's ``input_ids`` / ``labels`` /
        # ``logprobs`` stay exactly the exploration turns the local sampler
        # produced -- which is what "train only the exploration part" means. The
        # generated check script and statement are used solely to build the task.
        self.followup_api = followup_api
        # extra_body sent on every followup API call (e.g. {'thinking_budget': N}
        # to cap qwen3.8-max reasoning). None sends the request unmodified.
        self.followup_extra_body = dict(followup_extra_body) if followup_extra_body else None
        self.keyword_explorer = keyword_explorer
        self.min_batch = max(1, min_batch)
        self.problem_max_chars = problem_max_chars
        # A budget in proposals rather than in kept tasks, for runs whose purpose
        # is to measure what the current configuration produces: with a keep-rate
        # near 6% a keep-target of 8 is 128 proposals, and comparing two
        # configurations means giving them the same number of tries, not the same
        # output. 0 leaves the run governed by its keep-target.
        self.max_proposals_total = max_proposals_total
        # Arm B. Returns a python script that recreates this episode's input files,
        # captured while the workspace still holds them, and replayed before every
        # solver attempt. None leaves the solver starting from an empty directory.
        self.setup_script_fn = setup_script_fn
        # How a task statement becomes the solver's opening conversation. Without
        # one, the solver is handed the statement as a bare user message and no
        # system message at all -- nothing says it is working in a sandbox, that it
        # may take many turns, or that a reply carries one tool call. Measured on
        # arm B: 71 of 80 attempts used 2-3 turns, writing 8-12k characters into a
        # single python_executor argument and truncating there, so ``n_pass`` was
        # reporting that omission rather than the task. Passing the same function
        # the eval script uses is what keeps the two measuring the same thing.
        self.solver_prompt_fn = solver_prompt_fn
        if check_retries < 0:
            raise ValueError(f'check_retries must be >= 0, got {check_retries}')
        self.check_retries = check_retries
        if check_retries:
            prompts.require('check_retry_followup')
        # Novelty, off unless both of these are given. ``task_bank`` supplies the
        # tasks earlier iterations produced (see :mod:`.task_bank`); ``novelty_fn``
        # takes a list of ``{statement, check, references}`` and returns one score in
        # [0, 1] per entry, or None where it could not judge. Kept as injected
        # callables for the same reason the sandbox ones are: this class then holds
        # no opinion about which judge, model or API produces the number, and a test
        # can hand it a fixed one.
        self.task_bank = task_bank
        self.novelty_fn = novelty_fn
        if not 0.0 <= novelty_floor <= 1.0:
            raise ValueError(f'novelty_floor must be in [0, 1], got {novelty_floor}')
        # How much of the reward a proposal keeps when it is judged fully redundant.
        # 0.5 halves it; 0.0 would be Ornith-1.5's plain ``V x D x N``, which zeroes
        # it. The floor exists because our N is coarse where theirs is continuous:
        # scored over run_clean9's 188 tasks, 44% came out at exactly 0.0, and eight
        # proposals sharing one keyword draw can all land there -- at floor 0 that
        # group's rewards are all zero, its advantages are all zero after GRPO
        # subtracts the mean, and eight sandbox rollouts bought nothing. Ornith's own
        # text says novelty 'should remain secondary to validity and difficulty'.
        self.novelty_floor = float(novelty_floor)
        # At most this many of a keyword group's in-band proposals become tasks the
        # solver side trains on. 0 keeps every in-band proposal, which is what this
        # did before. At 1 the two sides come out the same size -- eight groups of
        # eight proposals give 64 proposing trajectories and 8 tasks x 8 attempts =
        # 64 solving ones -- and the tasks are one per keyword direction instead of
        # three from the same one.
        #
        # The proposals not selected are NOT wasted from the proposing side: each one
        # still earns its own reward from its own n_pass, so the whole group still
        # trains. What is dropped is their solver attempts, which were already run to
        # measure difficulty: at keep_per_group=1 that is 56 proposals x 8 attempts
        # per round measured and then not trained on.
        if keep_per_group < 0:
            raise ValueError(f'keep_per_group must be >= 0, got {keep_per_group}')
        self.keep_per_group = keep_per_group
        self.reject_sink = reject_sink
        self.propose_sink = propose_sink
        self.solver_sink = solver_sink
        self.keyword_sink = keyword_sink
        if self.seeds:
            prompts.require('from_seed')
            if self.store is not None:
                prompts.require('from_seed_keywords')
        self._nonce = 0
        self.stats: Dict[str, int] = {
            'explore_done': 0, 'check_parse_fail': 0, 'check_run_fail': 0,
            'empty_workspace': 0, 'solver_truncated': 0,
            # The workspace listing could not be read, as opposed to being empty.
            # Kept apart from ``empty_workspace`` because it says nothing about
            # what the model did.
            'snapshot_unavailable': 0,
            'problem_parse_fail': 0, 'too_long': 0, 'parsed': 0,
            # How often a check that failed was handed back for a rewrite, and
            # how often the rewrite passed. The two together say whether the
            # retry earns its extra sampling call.
            'check_retry': 0, 'check_retry_pass': 0,
            # The episode ended before the appended stages could run or finish:
            # it used up ``max_turns``, hit the trajectory token cap, or left no
            # room for the follow-up message. Distinct from every other reason
            # here, which is the model producing something unusable.
            'episode_cut_short': 0,
            # Arm B only. ``setup_capture_fail``: the episode's input files could
            # not be read back, so the task was dropped. ``setup_replay_fail``: a
            # solver attempt was skipped because putting those files back failed,
            # which would otherwise have scored as the task being too hard.
            'setup_capture_fail': 0, 'setup_replay_fail': 0,
            # followup_api mode only: a check or statement API call failed. The
            # conversation is then unusable and the proposal is rejected.
            'followup_api_error': 0,
            # Novelty judging. ``novelty_error``: the batch's call raised, so every
            # proposal in it scored None and none lost reward for it.
            # ``novelty_length_mismatch``: the judge returned a different number of
            # scores than proposals sent, which would pair scores with the wrong
            # tasks, so all are dropped. ``novelty_unjudged``: proposals the judge
            # left without a verdict. ``in_band_not_selected``: proposals inside the
            # difficulty band whose group already contributed its keep_per_group
            # task.
            #
            # These four have to be listed here: _bump does self.stats[key] += n on
            # a fixed dict, so an unregistered key raises KeyError and takes the
            # whole collection down. That is what killed loop2/iter1 -- seven
            # proposals were measured and scored, then the counter line at the end
            # of _score_novelty crashed and all seven trajectories were lost.
            'novelty_error': 0, 'novelty_length_mismatch': 0,
            'novelty_unjudged': 0, 'in_band_not_selected': 0,
            # Keyword groups given up on because the judge never scored one of
            # their proposals. Nothing from them is used.
            'novelty_group_dropped': 0,
        }
        self._hard: List[Tuple[str, str]] = []

    # ------------------------------------------------------------- proposing

    def propose(self, count: int) -> List[Trajectory]:
        """Build ``count`` prompt trajectories for round 1.

        Each carries a direction + keywords + optional seed. The explorer will
        run these multi-turn in the sandbox.

        ``proposals_per_group`` of them share one keyword draw, one seed choice
        and one identical prompt, and are tagged with the same ``group_id``.
        That is what makes a GRPO group on the proposing side: the advantage of
        a proposal is its reward minus the mean over the others answering the
        same prompt, so the members have to differ only by sampling noise. At 1
        -- which is what this used to be, every proposal its own keyword draw --
        every group has one member, the mean equals the reward, and every
        advantage is zero.
        """
        proposals: List[Trajectory] = []
        directions: List[str] = []
        metas: List[Tuple[List[Tuple[str, str]], bool, str, int]] = []
        per_group = max(1, self.proposals_per_group)
        while len(metas) < count:
            picks = self._draw_keywords()
            body = '\n'.join(f'- {c}: {t}' for c, t in picks)
            use_seed = bool(self.seeds) and self.rng.random() < self.seed_mix_prob
            seed = self.rng.choice(self.seeds) if use_seed else None
            if use_seed and picks:
                user = self.prompts.from_seed_keywords.format(
                    seed=seed['query'], keywords=body)
            elif use_seed:
                user = self.prompts.from_seed.format(seed=seed['query'])
            elif picks:
                user = self.prompts.from_keywords.format(keywords=body)
            else:
                user = self.prompts.from_scratch
            # The whole group gets the same prompt, so a short final group is a
            # group whose advantage is computed over fewer samples -- noisier,
            # but not wrong. Truncating to a multiple of per_group instead would
            # silently return fewer proposals than asked for.
            #
            # The counter is per-run, not per-call: ``propose`` runs once per
            # round, and restarting at 0 each round would give two unrelated
            # groups the same id in the dump.
            gid = self._next_group_id
            self._next_group_id += 1
            for _ in range(min(per_group, count - len(metas))):
                directions.append(user)
                metas.append((picks, use_seed, body, gid))

        for user, (picks, use_seed, body, gid) in zip(directions, metas):
            proposal: Trajectory = {
                'messages': [{'role': 'system', 'content': self.prompts.system},
                             {'role': 'user', 'content': user}],
            }
            if self.tool_schemas:
                proposal['tools'] = self.tool_schemas
            proposals.append(attach_user_data(
                proposal, keywords=picks, seeded=use_seed, keyword_block=body,
                group_id=gid))
        return proposals


    # ------------------------------------------------------------- building

    def build(self, explored: List[Trajectory]) -> List[Optional[Trajectory]]:
        """Satisfy the abstract method; not usable outside ``_round``.

        Building happens inside the episode now: :meth:`_followup` runs while the
        model is still generating and needs the sandbox to hold that episode's
        workspace state, which is only guaranteed inside the serial ``_round``
        loop.
        """
        raise RuntimeError(
            f'{type(self).__name__}.build() must not be called directly; '
            f'the serial _round() loop drives one episode at a time instead.')

    def _reject_for_empty_snapshot(self, state: Dict[str, Any], slot: int) -> None:
        """File an episode whose workspace listing came back empty.

        Empty means one of two unrelated things -- the episode built nothing, or
        the listing could not be read -- and only the first says anything about
        the model. ``snapshot_error_fn`` is what tells them apart; with no such
        callback every case is filed as ``empty_workspace``, which is what used
        to happen for all of them.
        """
        detail = self.snapshot_error_fn(slot=slot) if self.snapshot_error_fn else ''
        if detail:
            self._bump('snapshot_unavailable')
            state['reject'] = ('snapshot_unavailable', detail)
        else:
            self._bump('empty_workspace')
            state['reject'] = ('empty_workspace', '')

    def _followup(self, state: Dict[str, Any], trajectory: Trajectory,
                  n_before: int) -> Optional[Tuple[str, Optional[SamplingParams]]]:
        """What to say next when the model stops calling tools; ``None`` to stop.

        The rollout calls this once per stage, handing over the episode as it
        stands. ``state`` is this episode's scratchpad, read afterwards by
        :meth:`_finish_episode`: the workspace listing and the check script are
        produced here, and anything that goes wrong before a statement exists is
        left in ``state['reject']``.

        The sandbox work has to happen at this moment and nowhere else -- the
        workspace holds this episode's end state right now, and the next
        episode's reset wipes it. ``state['slot']`` says which sandbox that is;
        with concurrent episodes several of these run at once, each against its
        own.
        """
        slot = state.get('slot', 0)
        if n_before == 0:
            snapshot = self.workspace_snapshot_fn(slot=slot) if self.workspace_snapshot_fn else ''
            state['snapshot'] = snapshot
            # An episode that left nothing behind has no end state to write checks
            # about, and asking for them anyway is worse than useless: the only
            # true thing to assert is that the directory is empty, which every
            # solver passes by doing nothing. Five of run5's ten verified tasks
            # were that task. Reject here instead.
            if not snapshot.strip():
                self._reject_for_empty_snapshot(state, slot)
                return None
            return (self.prompts.check_followup.format(final_state=snapshot),
                    self.check_params)

        # Every follow-up from here until a check passes is a check-script reply:
        # the first one, plus up to ``check_retries`` rewrites.
        if not state.get('checked'):
            attempt = state.get('check_attempts', 0) + 1
            state['check_attempts'] = attempt
            reply = assistant_text(trajectory)
            script = parse_check_script(reply)
            if script is None:
                # Same one-rewrite budget a run failure gets: hand the parse
                # failure back and let it regenerate, rather than dropping a task
                # whose only fault was packaging. Shares the check_attempts
                # count, so parse and run failures together get check_retries
                # extra tries, not one each.
                if attempt <= self.check_retries:
                    self._bump('check_retry')
                    err = ('Could not read a check script from your reply: it was '
                           'not a fenced python code block. Do not wrap it in a '
                           'tool call and do not add prose -- return ONLY a fenced '
                           'python code block.')
                    return (self.prompts.check_retry_followup.format(
                        error=err, final_state=state.get('snapshot') or ''),
                        self.check_params)
                self._bump('check_parse_fail')
                # The whole reply, not a tail: this stage fails either because the
                # model declared the state untestable (it says so) or because it
                # ran out of tokens while thinking, and a record that cannot tell
                # them apart sends the next reader back to re-run the batch.
                state['reject'] = ('check_parse_fail', reply)
                return None
            state['script'] = script
            brittle = brittle_check_reason(script)
            if brittle is not None:
                # Same bookkeeping as a check that ran and failed: the script is
                # rejected before it can pass on the author's own state, because
                # passing there is exactly what hides the defect.
                exit_code, output = 1, brittle
            else:
                exit_code, output = self.run_check_fn(script, slot=slot)
            if exit_code == 0:
                state['checked'] = True
                if attempt > 1:
                    self._bump('check_retry_pass')
                # Capture the inputs now, at the one moment the workspace holds
                # exactly the state this check just passed on. A capture after the
                # statement stage would be the same bytes only by luck.
                if self.setup_script_fn is not None:
                    setup = self.setup_script_fn(slot=slot)
                    if not setup:
                        self._bump('setup_capture_fail')
                        state['reject'] = (
                            'setup_capture_fail',
                            'no input files to hand the solver, or their bytes '
                            'could not be read back')
                        return None
                    state['setup_script'] = setup
                return (self.prompts.problem_followup, self.problem_params)
            # Snapshot again, after the failure. A check that asserts only
            # paths from the snapshot it was shown and still fails leaves two
            # very different bugs indistinguishable -- the model asserted
            # something untrue, or the workspace changed under it -- and the
            # difference is visible only in the state at the moment the check
            # ran. It is also what the rewrite gets to read.
            after = self.workspace_snapshot_fn(slot=slot) if self.workspace_snapshot_fn else ''
            state.setdefault('attempts', []).append(
                f'--- attempt {attempt}: exit {exit_code} ---\n{output}\n'
                f'--- check script ---\n{script}')
            if attempt <= self.check_retries:
                self._bump('check_retry')
                return (self.prompts.check_retry_followup.format(
                    error=output, final_state=after or state.get('snapshot') or ''),
                    self.check_params)
            self._bump('check_run_fail')
            state['reject'] = (
                'check_run_fail',
                '\n'.join(state['attempts'])
                + f"\n--- state before check ---\n{state.get('snapshot') or ''}\n"
                + f'--- state after check ---\n{after}')
            return None

        return None

    def _api_reply(self, messages: List[Dict[str, Any]], user_text: str,
                   params: Optional[SamplingParams]) -> Optional[str]:
        """Append ``user_text`` and one ``followup_api`` reply to ``messages``.

        ``messages`` is a throwaway copy owned by :meth:`_run_followup_api`, never
        the trainable trajectory, so mutating it in place costs the model nothing.
        Returns the assistant text, or ``None`` when the API call raised -- the
        caller then rejects rather than building a task on a broken conversation.

        Tools are withdrawn for these stages on purpose (they are answers, not
        actions), so only the text is kept; any structured ``tool_calls`` the API
        returned are dropped.
        """
        messages.append({'role': 'user', 'content': user_text})
        request: Trajectory = {'messages': messages}
        try:
            if self.followup_extra_body:
                reply = self.followup_api(request, params, extra_body=self.followup_extra_body)
            else:
                reply = self.followup_api(request, params)
        except Exception as exc:  # noqa: BLE001 -- one bad call must not kill the round
            logger.warning(f'[{type(self).__name__}] followup API call failed: '
                           f'{type(exc).__name__}: {exc}')
            return None
        if isinstance(reply, list):
            reply = reply[0] if reply else {}
        content = (reply.get('content') if isinstance(reply, dict) else None) or ''
        messages.append({'role': 'assistant', 'content': content})
        return content

    def _run_followup_api(self, state: Dict[str, Any], explored: Trajectory) -> None:
        """Generate the check script and problem statement over ``followup_api``.

        The API-only counterpart of :meth:`_followup`: the same stages, the same
        sandbox work (snapshot, run the check, capture inputs) and the same retry
        budget, but driven imperatively here instead of turn-by-turn by the
        rollout, and answered by the API rather than the local model. Results land
        in ``state`` for :meth:`_finish_episode`:

          * ``state['script']`` / ``state['checked']`` -- the check that passed,
          * ``state['setup_script']`` -- captured inputs (Arm B),
          * ``state['statement']`` -- the problem-statement text,
          * ``state['reject']`` -- ``(reason, detail)`` when a stage fails.

        Nothing here touches ``explored``'s ``input_ids`` / ``labels`` /
        ``logprobs``: the messages the API sees are a private copy, so the
        trainable trajectory stays exactly the exploration turns the local sampler
        produced.
        """
        slot = state.get('slot', 0)
        messages: List[Dict[str, Any]] = [dict(m) for m in explored.get('messages') or []]

        snapshot = self.workspace_snapshot_fn(slot=slot) if self.workspace_snapshot_fn else ''
        state['snapshot'] = snapshot
        # An episode that left nothing behind has no end state to write checks
        # about; rejecting here mirrors the n_before==0 branch of _followup.
        if not snapshot.strip():
            self._reject_for_empty_snapshot(state, slot)
            return

        # Check-script stage: the first ask plus up to ``check_retries`` rewrites,
        # sharing one attempt counter across parse and run failures exactly as the
        # single-model path does.
        user_text = self.prompts.check_followup.format(final_state=snapshot)
        attempt = 0
        while True:
            attempt += 1
            state['check_attempts'] = attempt
            reply = self._api_reply(messages, user_text, self.check_params)
            if reply is None:
                self._bump('followup_api_error')
                state['reject'] = ('followup_api_error', 'check-script API call failed')
                return
            script = parse_check_script(reply)
            if script is None:
                if attempt <= self.check_retries:
                    self._bump('check_retry')
                    err = ('Could not read a check script from your reply: it was '
                           'not a fenced python code block. Do not wrap it in a '
                           'tool call and do not add prose -- return ONLY a fenced '
                           'python code block.')
                    user_text = self.prompts.check_retry_followup.format(
                        error=err, final_state=snapshot)
                    continue
                self._bump('check_parse_fail')
                state['reject'] = ('check_parse_fail', reply)
                return
            state['script'] = script
            brittle = brittle_check_reason(script)
            if brittle is not None:
                # Rejected before it can pass on the author's own state, since
                # passing there is exactly what hides the defect.
                exit_code, output = 1, brittle
            else:
                exit_code, output = self.run_check_fn(script, slot=slot)
            if exit_code == 0:
                state['checked'] = True
                if attempt > 1:
                    self._bump('check_retry_pass')
                # Capture inputs now, while the workspace still holds the state
                # this check just passed on.
                if self.setup_script_fn is not None:
                    setup = self.setup_script_fn(slot=slot)
                    if not setup:
                        self._bump('setup_capture_fail')
                        state['reject'] = (
                            'setup_capture_fail',
                            'no input files to hand the solver, or their bytes '
                            'could not be read back')
                        return
                    state['setup_script'] = setup
                break
            after = self.workspace_snapshot_fn(slot=slot) if self.workspace_snapshot_fn else ''
            state.setdefault('attempts', []).append(
                f'--- attempt {attempt}: exit {exit_code} ---\n{output}\n'
                f'--- check script ---\n{script}')
            if attempt <= self.check_retries:
                self._bump('check_retry')
                user_text = self.prompts.check_retry_followup.format(
                    error=output, final_state=after or snapshot)
                continue
            self._bump('check_run_fail')
            state['reject'] = (
                'check_run_fail',
                '\n'.join(state['attempts'])
                + f"\n--- state before check ---\n{state.get('snapshot') or ''}\n"
                + f'--- state after check ---\n{after}')
            return

        # Problem-statement stage: one API reply, kept as the task's statement.
        reply = self._api_reply(messages, self.prompts.problem_followup,
                                self.problem_params)
        if reply is None:
            self._bump('followup_api_error')
            state['reject'] = ('followup_api_error', 'problem-statement API call failed')
            return
        state['statement'] = reply

    def _finish_episode(self, state: Dict[str, Any],
                        explored: Trajectory) -> Optional[Trajectory]:
        """Turn a finished episode into a task, or record why it is not one.

        Everything the model wrote is in ``explored``: the tool-using turns, the
        check script, and the problem statement as the last assistant message.
        ``state`` carries what only the sandbox could say -- the end state, and
        whether the check passed on it.
        """
        keywords = user_data_get(explored.get('user_data'), 'keywords', [])
        seeded = user_data_get(explored.get('user_data'), 'seeded', False)
        group_id = user_data_get(explored.get('user_data'), 'group_id', None)
        # The episode as one record: a single conversation, so a single set of
        # token ids and logprobs. Handed to propose_sink with whatever verdict the
        # proposal ends up with, so a rejected attempt is recorded as fully as a
        # kept one.
        rounds = [_propose_round('episode', explored)]

        def reject(reason: str, detail: str = '') -> None:
            self._reject_record(explored, reason, detail=detail)
            self._emit_propose(rounds, reason, keywords=keywords, seeded=seeded,
                               group_id=group_id)

        if state.get('reject'):
            reason, detail = state['reject']
            reject(reason, detail)
            return None

        if not state.get('checked'):
            # The stages never ran, or the check-writing one never got a reply:
            # the episode used up its turns, hit the trajectory token cap, or left
            # no room to append the next message. The model produced nothing
            # wrong here, so this is not one of the other reasons. Keyed on the
            # check having *passed* rather than on a script existing: a rewrite
            # that never came back leaves the failed script in ``state``, and
            # building a task on it would ship a check nobody can pass.
            self._bump('episode_cut_short')
            reject('episode_cut_short',
                   detail=f"stop_reason={explored.get('stop_reason')} "
                          f"truncated={bool(explored.get('truncated'))} "
                          f"turns={explored.get('turns')} "
                          f"followups={explored.get('followups')}")
            return None

        script = state['script']
        # In followup_api mode the statement was written by the API and is not in
        # ``explored`` (whose last assistant turn is the final exploration reply);
        # it lives in ``state``. The single-model path keeps it as the last
        # assistant message of the episode.
        if self.followup_api is not None:
            statement = parse_problem_statement(state.get('statement') or '')
        else:
            statement = parse_problem_statement(assistant_text(explored))
        if statement is None:
            self._bump('problem_parse_fail')
            reject('problem_parse_fail')
            return None
        if len(statement) > self.problem_max_chars:
            self._bump('too_long')
            reject('too_long')
            return None

        self._bump('parsed')
        task: Trajectory = {
            'messages': [{'role': 'user', 'content': statement}],
        }
        # group_id travels with the task, not just with the reject path above: the
        # difficulty stage emits the surviving proposals from the task, so a task
        # that forgets its group reaches the dump ungrouped and the proposing side
        # has no advantage to compute. Leaving it off here made every kept and
        # outside_band proposal group_id=None, and only the episodes that failed
        # early -- which emit straight off ``explored`` -- kept theirs.
        task = attach_user_data(task, check_script=script, keywords=keywords, seeded=seeded,
                                group_id=group_id,
                                setup_script=state.get('setup_script', ''))
        # Carried, not emitted: the verdict this proposal earns depends on the
        # difficulty measurement, which has not run yet. A plain top-level key
        # rather than user_data, which json-encodes every value on each update.
        task['propose_rounds'] = rounds
        return task

    def _reject_record(self, traj: Trajectory, reason: str, detail: str = '') -> None:
        """Record a rejected proposal, with enough of the episode to tell why.

        The reason alone is not diagnosable. Nine ``empty_workspace`` rejections in
        one run all looked like the model refusing to act; the messages showed a
        single assistant turn each, and the question of whether it had run out of
        tokens or simply emitted no call could not be answered from the record --
        the fields that answered it were on the trajectory and were dropped. So
        how the episode ended travels with the reason.
        """
        if self.reject_sink is None:
            return
        messages = traj.get('messages') or []
        payload: Dict[str, Any] = {'reason': reason}
        if detail:
            payload['detail'] = detail
        payload['stop_reason'] = traj.get('stop_reason')
        payload['truncated'] = bool(traj.get('truncated'))
        payload['turns'] = traj.get('turns')
        payload['n_assistant'] = sum(1 for m in messages
                                     if isinstance(m, dict) and m.get('role') == 'assistant')
        payload['n_tool_calls'] = sum(len(m.get('tool_calls') or []) for m in messages
                                      if isinstance(m, dict))
        payload['last_assistant'] = assistant_text(traj)
        with self._sink_lock:
            self.reject_sink(payload)

    def _emit_propose(self, rounds: Optional[List[Dict[str, Any]]], outcome: str, *,
                      keywords: Any = (), seeded: bool = False,
                      n_pass: Optional[int] = None,
                      group_id: Optional[int] = None,
                      novelty: Optional[float] = None,
                      selected: Optional[bool] = None,
                      novelty_dropped: bool = False) -> None:
        """Hand one proposal attempt's rounds to ``propose_sink``.

        ``pass_rate`` is the raw fraction of solver attempts that succeeded.
        ``challenger_reward`` is that fraction scored against a 50% target by
        :meth:`challenger_reward`, which is the number the proposing side trains
        on; both are written so a run can be re-scored under a different target
        without re-solving anything.

        ``novelty`` is written next to it for the same reason: the reward already
        has it multiplied in, and a run cannot be re-scored at a different floor --
        or with novelty taken back out -- from the product alone.

        A proposal with no ``n_pass`` never reached difficulty measurement -- it
        was rejected before that -- and scores 0, the same as one nobody or
        everybody solved.
        """
        if self.propose_sink is None or not rounds:
            return
        rollouts = self.solver_rollouts or None
        payload = {
            'outcome': outcome,
            'group_id': group_id,
            'n_pass': n_pass,
            'n_rollouts': rollouts,
            'pass_rate': (n_pass / rollouts) if (n_pass is not None and rollouts) else None,
            'novelty': novelty,
            'novelty_factor': self.novelty_factor(novelty),
            'challenger_reward': self.challenger_reward(n_pass, novelty=novelty),
            # Whether this proposal's task went on to the solver side. Not the same as
            # ``outcome``: with keep_per_group set, a proposal can be in the difficulty
            # band and still not be the one its group contributed. Its own reward is
            # unaffected either way.
            'selected': selected,
            # True when the novelty judge never returned a score for at least one
            # proposal in this group, after NOVELTY_TRIES attempts. The record is
            # written either way -- the episode really happened and the file is the
            # audit trail -- but training skips every group carrying this flag.
            'novelty_dropped': bool(novelty_dropped),
            'keywords': list(keywords or ()),
            'seeded': bool(seeded),
            'rounds': rounds,
        }
        with self._sink_lock:
            self.propose_sink(payload)

    # Where the pass-rate reward peaks, and how wide the peak is. 0.2 is Ornith-1.5's
    # target (ornith.ai/ornith_1_5.html), which trains its proposer on
    # ``exp(-(p-p*)^2 / 2s^2)`` rather than on a peak at one half.
    PASS_RATE_TARGET = 0.2
    PASS_RATE_WIDTH = 0.3

    # How many times the novelty judge is asked before a proposal is given up on.
    # Only the proposals still missing a score are re-sent. Measured need for this:
    # loop3/iter1 had 1 of 61 measured proposals come back without a verdict, in 1
    # of its 10 keyword groups, and giving up on that group costs the 56 sandbox
    # attempts already spent on its 7 proposals.
    NOVELTY_TRIES = 3

    def novelty_factor(self, novelty: Optional[float]) -> float:
        """What a proposal's difficulty score gets multiplied by for its novelty.

        ``floor + (1 - floor) * N``, so N=1 leaves the reward alone and N=0 leaves
        ``novelty_floor`` of it. See ``novelty_floor`` in ``__init__`` for why there
        is a floor at all.

        ``None`` returns 1.0, not the floor: it means nobody judged this proposal --
        no bank, no judge, or the judge's API failed -- and charging a proposal for a
        measurement that did not happen would make the reward depend on API uptime.
        """
        if novelty is None:
            return 1.0
        n = min(1.0, max(0.0, float(novelty)))
        return self.novelty_floor + (1.0 - self.novelty_floor) * n

    def challenger_reward(self, n_pass: Optional[int],
                          novelty: Optional[float] = None) -> float:
        """Score a proposal by how close the solver came to a target pass rate.

        ``exp(-(p - p*)^2 / 2s^2)`` for ``p = n_pass / solver_rollouts``, peaked at
        ``p* = 0.2`` with width ``s = 0.3``.

        This replaced ``1 - 2*|p - 1/2|``, which is what R-Zero (arXiv 2508.05004)
        uses, for two reasons measured on run_clean9's 87 in-band proposals:

        It was not injective. With 8 rollouts the seven in-band values of ``n_pass``
        mapped onto four rewards -- 1 and 7 both scored 0.25, 2 and 6 both 0.50 --
        so a proposal one solver out of eight could do and one seven out of eight
        could do were worth the same. The whole distinction between too hard and too
        easy was erased. The gaussian separates all seven.

        Its signal was smaller than its noise. ``n_pass`` is a binomial draw around
        the proposal's real difficulty, and propagating that draw through each shape
        gives a noise SD to compare the spread of rewards against: 0.280 signal over
        0.246 noise for the old shape, against 0.347 over 0.177 here. A ratio of 1.14
        means over half of what a GRPO group ranks on is which way eight coin flips
        landed.

        A peak below one half is also the more useful target. A group's update size
        goes with reward variance, which for a pass/fail solver peaks at p=0.5 -- the
        argument for the old shape -- but a proposal only teaches the solver
        something when the solver mostly cannot do it yet.

        ``None`` means the proposal never got as far as being solved, and 0 means no
        attempt passed. Both score 0, and that floor is now load-bearing rather than
        incidental: the gaussian evaluated at p=0 is 0.801, higher than the 0.607 it
        gives a proposal half the attempts solve. Without the gate the best thing a
        proposer could do is write tasks nobody can finish.

        ``novelty`` multiplies the result through :meth:`novelty_factor`, which is
        Ornith-1.5's ``R = V x D x N`` with a floor under the N. Left at ``None`` --
        which is what happens with no task bank or no judge -- the returned number is
        exactly what it was before novelty existed.
        """
        rollouts = self.solver_rollouts or 0
        if n_pass is None or not rollouts or n_pass <= 0:
            return 0.0
        gap = n_pass / rollouts - self.PASS_RATE_TARGET
        difficulty = math.exp(-(gap * gap) / (2.0 * self.PASS_RATE_WIDTH ** 2))
        return difficulty * self.novelty_factor(novelty)

    def _take_rounds(self, task: Trajectory) -> Optional[List[Dict[str, Any]]]:
        """Detach a task's proposing rounds. Popped even with no sink attached:
        token ids for a whole agentic episode are large, and a kept task is held
        until the caller's batch is full.
        """
        return task.pop('propose_rounds', None)

    # ------------------------------------------------------------ revised _round

    def _bump(self, key: str, n: int = 1) -> None:
        """Thread-safe stats increment."""
        with self._sink_lock:
            self.stats[key] += n

    def _parallel(self, fn: Callable[[Any], Any], items: Sequence[Any]) -> List[Any]:
        """Map ``fn`` over ``items`` at once, results in input order.

        Every use of this is waiting on a sandbox, not computing, so the thread
        pool is the point. One item runs inline: a pool for a single sandbox call
        only adds a thread, and it keeps the serial configuration on exactly the
        same code path it had before.
        """
        items = list(items)
        if len(items) <= 1:
            return [fn(item) for item in items]
        out: List[Any] = [None] * len(items)
        with ThreadPoolExecutor(max_workers=len(items)) as pool:
            futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
            for fut in as_completed(futures):
                out[futures[fut]] = fut.result()
        return out

    def _run_episode(self, proposal: Trajectory, slot: int) -> Optional[Trajectory]:
        """One episode top-to-bottom, using sandbox slot ``slot``."""
        self.reset_fn(slot=slot)
        state: Dict[str, Any] = {'slot': slot}
        tm = self.episode_tool_managers[slot] if self.episode_tool_managers else None
        if self.followup_api is not None:
            # Split path: explore on the local (trainable) model with NO
            # followup_fn, so the rollout ends the moment the model stops calling
            # tools and the returned trajectory carries only the exploration
            # turns' input_ids/labels/logprobs. The check-script and
            # problem-statement stages then run over the API against the same end
            # state, appended to a throwaway copy of the messages -- never to the
            # trainable trajectory.
            kwargs: Dict[str, Any] = {}
            if tm is not None:
                kwargs['tool_manager'] = tm
            result = self.explore([proposal], **kwargs)
            if not result:
                return None
            explored = result[0]
            self._bump('explore_done')
            # A reply cut off at the token budget never finished its thought, so
            # continuing the conversation over the API would build a check on a
            # half-written turn. Leave ``state`` untouched and let
            # ``_finish_episode`` record it as ``episode_cut_short``, matching the
            # single-model path which does not run the stages after a length cut.
            if explored.get('stop_reason') != 'length':
                self._run_followup_api(state, explored)
            return self._finish_episode(state, explored)
        kwargs = {'followup_fn': partial(self._followup, state)}
        if tm is not None:
            kwargs['tool_manager'] = tm
        result = self.explore([proposal], **kwargs)
        if not result:
            return None
        explored = result[0]
        self._bump('explore_done')
        return self._finish_episode(state, explored)

    def _round(self, missing: int) -> Optional[List[Trajectory]]:
        """One cycle: episodes in parallel across sandbox slots, then difficulty."""
        count = min(self._estimate(missing), self.max_proposals_per_round)
        if self.max_proposals_total > 0:
            left = self.max_proposals_total - self.n_proposed
            if left <= 0:
                # Budget spent. None is the 'source exhausted' answer the batching
                # loop already knows how to stop on, so the run ends after this
                # round's keepers are handed back rather than mid-episode.
                logger.info(f'[{type(self).__name__}] proposal budget spent '
                            f'({self.n_proposed}/{self.max_proposals_total}); stopping')
                return None
            count = min(count, left)
        proposals = self.propose(count)
        if not proposals:
            return None

        usable: List[Trajectory] = []
        n_slots = self.episode_concurrency

        if n_slots <= 1 or len(proposals) <= 1:
            # Serial fallback (original path).
            for proposal in proposals:
                task = self._run_episode(proposal, slot=0)
                if task is not None:
                    usable.append(task)
        else:
            # One worker per sandbox slot, each draining its own share serially.
            # A slot is a single sandbox and cannot host two episodes at once, so
            # the split is by slot, never round-robin into a shared pool where two
            # tasks could land on the same slot concurrently.
            buckets: List[List[Trajectory]] = [[] for _ in range(n_slots)]
            for i, proposal in enumerate(proposals):
                buckets[i % n_slots].append(proposal)

            def _drain(slot: int) -> List[Trajectory]:
                out: List[Trajectory] = []
                for proposal in buckets[slot]:
                    task = self._run_episode(proposal, slot=slot)
                    if task is not None:
                        out.append(task)
                return out

            with ThreadPoolExecutor(max_workers=n_slots) as pool:
                futures = [pool.submit(_drain, s) for s in range(n_slots) if buckets[s]]
                for fut in as_completed(futures):
                    usable.extend(fut.result())

        kept = self._filter_difficulty(usable) if self.solver_rollouts else usable
        if not self.solver_rollouts:
            # No difficulty stage, so the verdict is final as soon as it is built.
            for task in usable:
                self._emit_propose(self._take_rounds(task), 'kept',
                                   keywords=user_data_get(task.get('user_data'), 'keywords', []),
                                   seeded=user_data_get(task.get('user_data'), 'seeded', False),
                                   group_id=user_data_get(task.get('user_data'), 'group_id', None))
        self.n_proposed += len(proposals)
        self.n_kept += len(kept)
        band = (f', in difficulty band {len(kept)}' if self.solver_rollouts else '')
        logger.info(f'[{type(self).__name__}] proposed {len(proposals)}, usable '
                    f'{len(usable)}{band} (cumulative {self.n_kept}/{self.n_proposed})')
        return kept

    # ------------------------------------------------------------ difficulty

    def _filter_difficulty(self, tasks: List[Trajectory]) -> List[Trajectory]:
        """Override: every solver attempt needs its own clean workspace.

        Attempts are run in waves of ``episode_concurrency``, attempt k of a wave
        in sandbox slot k. Within a wave all attempts go out in one explorer call,
        so the sampler generates them as one batch instead of leaving the GPUs
        waiting on a single sequence, and the wave's clears, input replays and
        checks all run at the same time too -- they are sandbox round-trips, not
        compute.

        The slot is what keeps this honest: a wave's attempts each clear, act in
        and get checked against their own sandbox. Sharing one would let attempt A
        pass on files attempt B wrote, and ``n_pass`` would stop being a
        difficulty measurement.

        An attempt cut off at the generation budget is counted in
        ``stats['solver_truncated']`` but still counts as a failure, because
        deciding otherwise decides which tasks are kept. Watch that number: when
        it is a large share of ``solver_rollouts`` times the task count, ``n_pass``
        is reporting the token budget rather than the difficulty. It was 15 of 50
        on one run, and one task lost all four attempts that way and was discarded
        as too hard without a solver ever touching the workspace. Raising
        ``solver_params.max_tokens`` took it to 0 of 20.
        """
        if not tasks:
            return []
        passes = [0] * len(tasks)
        n_slots = max(1, self.episode_concurrency)
        # Which task each attempt belongs to, flattened, so a wave is a fixed
        # number of sandboxes no matter how attempts distribute over tasks.
        plan = [i for i in range(len(tasks)) for _ in range(self.solver_rollouts)]

        for start in range(0, len(plan), n_slots):
            wave = plan[start:start + n_slots]
            slots = list(range(len(wave)))
            setups = [user_data_get(tasks[i].get('user_data'), 'setup_script', '')
                      for i in wave]

            def _prepare(k: int) -> bool:
                """Clear slot k, then put back the inputs this task hands out."""
                self.reset_fn(slot=k)
                if not setups[k]:
                    return True
                exit_code, output = self.run_check_fn(setups[k], slot=k)
                if exit_code != 0:
                    # Measuring this attempt against a workspace missing its
                    # inputs would score the task as harder than it is, so the
                    # attempt is skipped and counted rather than run.
                    logger.warning(f'[{type(self).__name__}] input setup failed in '
                                   f'slot {k} (exit {exit_code}): {output[-200:]}')
                    return False
                return True

            ready = self._parallel(_prepare, slots)
            live = [k for k in slots if ready[k]]
            self._bump('setup_replay_fail', len(slots) - len(live))
            if not live:
                continue
            prompts = [dict(self.solver_prompt(tasks[wave[k]])) for k in live]
            kwargs: Dict[str, Any] = {}
            if self.episode_tool_managers:
                kwargs['tool_manager'] = [self.episode_tool_managers[k] for k in live]
            attempts = self._solver_explore(prompts, sampling_params=self.solver_params,
                                            **kwargs)
            if len(attempts) != len(prompts):
                # Counting a partial return would silently understate every
                # affected task's pass count, i.e. report tasks as harder than
                # they are.
                raise RuntimeError(f'explorer returned {len(attempts)} attempts for '
                                   f'{len(prompts)} solver prompts; expected one per prompt.')
            for attempt in attempts:
                if attempt is not None and attempt.get('truncated'):
                    self._bump('solver_truncated')
            verdicts = self._parallel(
                lambda j: (attempts[j] is not None
                           and self.judge_attempt(tasks[wave[live[j]]], attempts[j],
                                                  slot=live[j])),
                list(range(len(live))))
            for j, passed in enumerate(verdicts):
                if passed:
                    passes[wave[live[j]]] += 1

        measured = [
            attach_user_data(task, n_pass=passes[i], n_rollouts=self.solver_rollouts)
            for i, task in enumerate(tasks)
        ]
        self.on_difficulty_measured(measured)
        novelties = self._score_novelty(measured)
        high = self.solver_rollouts - self.keep_max_pass_margin
        in_band = [self.keep_min_pass <= n <= high for n in passes]
        # Which of the in-band tasks the solver side actually trains on. Decided
        # before emitting so each proposal's record says whether its task was taken.
        selected = self._select_per_group(measured, passes, in_band, novelties)
        dropped = self._unscored_group_ids(measured, novelties)
        if dropped:
            self._bump('novelty_group_dropped', len(dropped))
        # Emit here, not in _round: this is where a proposal's verdict is
        # decided, and both sides of the band are worth keeping -- a task nobody
        # solved and one everybody solved are the two failure modes the
        # proposer would need to learn to avoid.
        for i, (task, n, kept_flag, nov) in enumerate(zip(measured, passes, in_band,
                                                          novelties)):
            gid = user_data_get(task.get('user_data'), 'group_id', None)
            self._emit_propose(self._take_rounds(task),
                               'kept' if kept_flag else 'outside_band',
                               keywords=user_data_get(task.get('user_data'), 'keywords', []),
                               seeded=user_data_get(task.get('user_data'), 'seeded', False),
                               n_pass=n,
                               group_id=gid,
                               novelty=nov,
                               selected=selected[i],
                               novelty_dropped=gid in dropped)
        return [t for t, take in zip(measured, selected) if take]

    def _unscored_group_ids(self, measured: List[Trajectory],
                            novelties: List[Optional[float]]) -> set:
        """Keyword groups the novelty judge never finished answering for.

        A group lands here when at least one of its proposals still has no score
        after ``NOVELTY_TRIES`` attempts. Nothing from such a group is used: no task
        is taken from it (``_select_per_group``) and its proposals are marked
        ``novelty_dropped`` so training skips the whole group. The collecting loop
        then keeps going and a later keyword draw makes up the shortfall.

        Only meaningful when novelty is on. With it off every score is ``None`` by
        design, which must not drop everything, so an off judge returns no groups.
        """
        if self.task_bank is None or self.novelty_fn is None:
            return set()
        return {user_data_get(task.get('user_data'), 'group_id', None)
                for task, nov in zip(measured, novelties) if nov is None}

    def _select_per_group(self, measured: List[Trajectory], passes: List[int],
                          in_band: List[bool],
                          novelties: List[Optional[float]]) -> List[bool]:
        """Which in-band proposals become tasks: all of them, or the best few per group.

        With ``keep_per_group = k > 0``, each keyword group contributes at most its ``k``
        highest-reward in-band proposals -- reward being the same number the proposing
        side trains on, ``challenger_reward``, so the task kept is the one whose pass
        rate sat closest to the target and, when novelty is on, was not judged a repeat
        of something already in the bank.

        A group with no in-band proposal contributes nothing and is not replaced here:
        the collecting loop keeps proposing rounds until the run's target number of
        tasks is reached, so a group that produced none is skipped and paid for by one
        more group later.

        Proposals with no ``group_id`` (a run with ``proposals_per_group=1``) are each
        their own group, so this is a no-op for them beyond the in-band filter.
        """
        if self.keep_per_group <= 0:
            return list(in_band)
        ranked: Dict[Any, List[Tuple[float, int]]] = {}
        unscored_groups = self._unscored_group_ids(measured, novelties)
        for i, task in enumerate(measured):
            if not in_band[i]:
                continue
            gid = user_data_get(task.get('user_data'), 'group_id', None)
            if gid in unscored_groups:
                # The judge never finished scoring this group, so there is no honest
                # way to rank its members against each other.
                continue
            key = gid if gid is not None else f'_ungrouped_{i}'
            ranked.setdefault(key, []).append(
                (self.challenger_reward(passes[i], novelty=novelties[i]), i))
        selected = [False] * len(measured)
        for key, entries in ranked.items():
            # Ties broken by the earlier proposal, so the choice does not depend on
            # dict or sort instability.
            entries.sort(key=lambda pair: (-pair[0], pair[1]))
            for _, i in entries[:self.keep_per_group]:
                selected[i] = True
        dropped = sum(1 for i in range(len(measured)) if in_band[i] and not selected[i])
        if dropped:
            self._bump('in_band_not_selected', dropped)
        return selected

    def _score_novelty(self, measured: List[Trajectory]) -> List[Optional[float]]:
        """One novelty score per measured proposal, ``None`` for every one if off.

        Scored for the whole batch in one call, and with the batch's own statements
        as part of each proposal's reference set, because the comparison that matters
        is against the siblings sharing a keyword draw: GRPO subtracts the group mean,
        so a term that comes out the same for all eight members of a group cancels
        exactly and the API calls bought nothing. Only same-group siblings go in --
        an unrelated proposal from the same round is not evidence of redundancy.

        Failures return ``None`` rather than 0.0 and never raise: a judge that is
        down must not turn into every proposal being redundant, and must not lose a
        round of sandbox work either.

        A proposal the judge skipped is asked about again, up to ``NOVELTY_TRIES``
        attempts in total, sending only the ones still missing. Whatever is still
        unscored after that leaves its whole keyword group out of both the task
        selection and the training data -- see ``_select_per_group`` and the
        ``novelty_dropped`` field written by ``_emit_propose``.
        """
        if self.task_bank is None or self.novelty_fn is None or not measured:
            return [None] * len(measured)
        statements, checks, groups = [], [], []
        for task in measured:
            statements.append(self.statement_of(task))
            checks.append(user_data_get(task.get('user_data'), 'check_script', '') or '')
            groups.append(user_data_get(task.get('user_data'), 'group_id', None))
        payload = []
        for i, statement in enumerate(statements):
            siblings = [statements[j] for j in range(len(statements))
                        if j != i and groups[j] is not None and groups[j] == groups[i]]
            payload.append({'statement': statement, 'check': checks[i],
                            'references': self.task_bank.references(statement, siblings)})
        scores: List[Optional[float]] = [None] * len(measured)
        pending = list(range(len(measured)))
        for attempt in range(self.NOVELTY_TRIES):
            batch = [payload[i] for i in pending]
            try:
                got = list(self.novelty_fn(batch))
            except Exception as e:  # noqa
                logger.warning(f'[{type(self).__name__}] novelty scoring failed on '
                               f'{len(batch)} proposals, try {attempt + 1} of '
                               f'{self.NOVELTY_TRIES} ({type(e).__name__}: {e})')
                self._bump('novelty_error', len(batch))
                continue
            if len(got) != len(batch):
                # Zipping a short list would pair later proposals with someone
                # else's number, so the whole reply is dropped.
                logger.warning(f'[{type(self).__name__}] novelty judge returned '
                               f'{len(got)} scores for {len(batch)} proposals, try '
                               f'{attempt + 1} of {self.NOVELTY_TRIES}; ignoring them')
                self._bump('novelty_length_mismatch', len(batch))
                continue
            still: List[int] = []
            for i, score in zip(pending, got):
                if score is None:
                    still.append(i)
                else:
                    scores[i] = score
            if not still:
                break
            logger.info(f'[{type(self).__name__}] novelty: {len(still)} of '
                        f'{len(batch)} left unscored on try {attempt + 1}; '
                        f'asking again')
            pending = still
        else:
            pending = [i for i, s in enumerate(scores) if s is None]
        unscored = [i for i, s in enumerate(scores) if s is None]
        if unscored:
            self._bump('novelty_unjudged', len(unscored))
            logger.warning(f'[{type(self).__name__}] {len(unscored)} proposal(s) '
                           f'still unscored after {self.NOVELTY_TRIES} tries; their '
                           f'keyword groups are dropped')
        return scores

    def statement_of(self, task: Trajectory) -> str:
        """The statement text a task was built around: its first user message."""
        for message in task.get('messages') or []:
            if isinstance(message, dict) and message.get('role') == 'user':
                return message.get('content') or ''
        return ''

    def solver_prompt(self, task: Trajectory) -> Trajectory:
        """The statement as the solver first sees it: system message, query, tools.

        ``solver_prompt_fn`` is how the surrounding script hands over the same
        opening the eval script builds, so a task kept at n_pass=4 here is a task
        the eval measures the same way. Without one this falls back to the bare
        statement, which is what it used to be.

        The schemas travel with the prompt for the same reason they do in round 1:
        a solver that cannot see the tool names cannot use them, and would score
        zero on every task regardless of difficulty.
        """
        if self.solver_prompt_fn is not None:
            messages = task.get('messages') or []
            query = next((m.get('content', '') for m in messages
                          if m.get('role') == 'user'), '')
            prompt = self.solver_prompt_fn(query)
            if not prompt.get('tools') and self.tool_schemas:
                prompt['tools'] = self.tool_schemas
            return prompt
        prompt: Trajectory = {'messages': [dict(m) for m in task.get('messages') or []]}
        if self.tool_schemas:
            prompt['tools'] = self.tool_schemas
        return prompt

    def judge_attempt(self, task: Trajectory, attempt: Trajectory,
                      slot: int = 0) -> bool:
        """Run the check script against sandbox ``slot``'s current state.

        Also hands the whole attempt to ``solver_sink`` when one is given. The
        difficulty stage otherwise reports a single number per task, and
        ``n_pass=0`` reads the same whether the task is impossible, the statement
        withholds something the check demands, or the solver merely gave up --
        which are three different things to fix. The evidence that separates them
        is the attempt itself and the state it left, so both are recorded here
        rather than reconstructed later.
        """
        script = user_data_get(task.get('user_data'), 'check_script', '')
        if not script:
            return False
        exit_code, output = self.run_check_fn(script, slot=slot)
        if self.solver_sink is not None:
            messages = task.get('messages') or [{}]
            record = {
                'statement': messages[0].get('content', ''),
                'check_script': script,
                'passed': exit_code == 0,
                'check_exit': exit_code,
                'check_output': output,
                # Whether the reply was cut off at the generation budget. The
                # difficulty stage drops such an attempt from its denominator, so
                # the flag has to travel with the record for the dropped count to
                # be reproducible from the dump.
                'truncated': bool((attempt or {}).get('truncated')),
                'attempt': attempt,
                'end_state': (self.workspace_snapshot_fn(slot=slot)
                              if self.workspace_snapshot_fn else ''),
            }
            with self._sink_lock:
                self.solver_sink(record)
        return exit_code == 0

    def on_difficulty_measured(self, candidates: List[Trajectory]) -> None:
        """Remember keywords behind candidates nobody solved."""
        if self.store is None:
            return
        seen = {(c, t.lower()) for c, t in self._hard}
        for task in candidates:
            data = task.get('user_data')
            if user_data_get(data, 'n_pass', 0) > 0:
                continue
            for pick in user_data_get(data, 'keywords', []) or []:
                if isinstance(pick, (list, tuple)) and len(pick) >= 2:
                    c, t = pick[0], pick[1]
                    if (c, t.lower()) not in seen:
                        seen.add((c, t.lower()))
                        self._hard.append((c, t))

    # ------------------------------------------------------------ keywords

    def _draw_keywords(self) -> List[Tuple[str, str]]:
        """Consume one keyword combination from the bank; [] without a bank."""
        if self.store is None:
            return []
        categories = self.store.categories
        if self.combo_arity == 'mix':
            if self.arity_weights and len(self.arity_weights) == len(categories):
                k = self.rng.choices(range(1, len(categories) + 1),
                                     weights=self.arity_weights)[0]
            else:
                k = self.rng.randint(1, len(categories))
            cats = self.rng.sample(list(categories), k)
        elif self.rng.random() < self.single_kw_prob:
            cats = [self.rng.choice(categories)]
        else:
            cats = list(categories)
        picks: List[Tuple[str, str]] = []
        # Refill every dry category at once rather than as each one is reached: the
        # three refills are independent model calls that used to run one after
        # another (20s each at the start of a run), and they touch separate
        # entries of the bank.
        dry = [c for c in cats if not self.store.unused(c)]
        if dry:
            self._parallel(self._refill, dry)
        for c in cats:
            text = self.store.take(c, self.rng)
            if text is not None:
                picks.append((c, text))
        return picks

    def _refill(self, category: str) -> None:
        """Ask the model for more keywords in ``category``.

        Says so when it comes back empty. A silent no-op here is the worst
        outcome available: ``_draw_keywords`` then hands out no keywords, every
        proposal quietly falls back to the from-scratch prompt, and the run looks
        normal while producing one identical prompt over and over. That is exactly
        what happened for whole runs when the prompt asked for one keyword per
        line and the parser wanted a JSON array.
        """
        tries = 0
        while not self.store.unused(category):
            new = self._generate_keywords(category, self.keyword_refill_target)
            added = self.store.add(category, new, source='gen')
            tries += 1
            if added:
                logger.info(f'[AgenticChallenger] keyword category {category!r} '
                            f'refilled +{added} (try {tries})')
                continue
            logger.warning(
                f'[AgenticChallenger] keyword refill for {category!r} produced '
                f'nothing on try {tries}: {len(new)} parsed, 0 new. Proposals will '
                f'run without keywords unless this recovers -- pass keyword_sink '
                f'to see the replies.')
            if tries >= self.keyword_refill_tries:
                if self.store.items[category]:
                    self.store.recycle(category)
                    logger.info(f'[AgenticChallenger] keyword category {category!r} '
                                f'exhausted -> recycled {len(self.store.items[category])} topics')
                break

    def _generate_keywords(self, category: str, n_want: int) -> List[str]:
        """Up to ``n_want`` keywords the bank does not already hold.

        Runs on ``keyword_explorer`` when there is one: brainstorming a list is a
        text round, and putting it through the sandbox-tool explorer both wastes
        turns and lets a bracketed list in the reply be taken for a tool call.
        """
        if n_want <= 0:
            return []
        known = self.store.texts(category)
        n_calls = max(self.keyword_gen_calls, self.min_batch)
        per_call = max(1, -(-n_want // n_calls) + 4)
        seen = {t.strip().lower() for t in known}
        out: List[str] = []
        explorer = self.keyword_explorer or self.explorer
        for start in range(0, n_calls, self.keyword_refill_concurrency):
            group = range(start, min(start + self.keyword_refill_concurrency, n_calls))
            # Every call in a group is built before any of them runs, so they all
            # carry the same avoid list -- which is exactly the batched behaviour,
            # and why a group of one is what lets call k+1 see call k.
            users = [(self.prompts.keyword_user.format(
                k=per_call, desc=self.category_desc[category])
                + self._avoid_note(known, out,
                                   '\nDo NOT repeat any of these already-used topics: ')
                + f'\n(batch {self._next_nonce()}-{i})') for i in group]
            prompts = [{
                'messages': [{'role': 'system', 'content': self.prompts.keyword_system},
                             {'role': 'user', 'content': u}],
            } for u in users]
            for user, reply in zip(users, explorer(prompts,
                                                  sampling_params=self.keyword_params)):
                text = assistant_text(reply)
                parsed = parse_keyword_list(text)
                fresh = []
                for kw in parsed:
                    key = kw.lower()
                    if key not in seen:
                        seen.add(key)
                        fresh.append(kw)
                out.extend(fresh)
                if self.keyword_sink is not None:
                    # Full text, both sides. The one question this dump exists to
                    # answer -- did the model disobey the format, or does the parser
                    # reject what it produced -- cannot be answered from a count.
                    record = {
                        'category': category,
                        'prompt': user,
                        'reply': text,
                        'stop_reason': reply.get('stop_reason'),
                        'truncated': bool(reply.get('truncated')),
                        'parsed': parsed,
                        'n_parsed': len(parsed),
                        'n_new': len(fresh),
                    }
                    with self._sink_lock:
                        self.keyword_sink(record)
        with self._kw_lock:
            self.rng.shuffle(out)
        return out[:n_want]

    # How many phrases the 'do not repeat these' line may quote in total. There
    # has to be a ceiling in both directions: too few and a serial refill stops
    # seeing what it just said, too many and the model runs out of room to obey.
    # Measured on armA2ser, where this refill's own output went in uncapped: with
    # 130 quoted the eighth call was still answering normally, with 150 it started
    # inventing -- 'îRAPIÓN holistic replace', 'ซะ subspace cutter map limit', 10
    # of 480 phrases that run. 100 sits below where that began.
    _AVOID_TOTAL = 100

    def _next_nonce(self) -> int:
        """A number no other call gets, so two prompts are never byte-identical.

        Shared across categories, which refill at the same time: two threads
        reading the counter together would send the same prompt twice and halve
        the diversity with nothing to show that it happened.
        """
        with self._kw_lock:
            self._nonce += 1
            return self._nonce

    def _avoid_note(self, older: List[str], fresh: List[str], lead: str) -> str:
        """The 'do not repeat these' line, newest first, capped at ``_AVOID_TOTAL``.

        What this refill has just produced comes first and evicts older entries
        rather than the reverse -- the calls run one at a time so that each can
        avoid what the ones before it said, and dropping those would undo it. Past
        the cap the oldest of *this refill's* phrases are what falls off, which is
        also the least costly thing to drop: the model has already moved away from
        them.
        """
        fresh_shown = list(fresh)[-self._AVOID_TOTAL:]
        room = max(0, self._AVOID_TOTAL - len(fresh_shown))
        with self._kw_lock:
            shown = older if len(older) <= room else self.rng.sample(older, room)
        avoid = fresh_shown + list(shown)
        return lead + ', '.join(avoid) if avoid else ''



    # ------------------------------------------------------------ feedback

    def expand_hard_keywords(self) -> int:
        """Brainstorm more topics in families that produced the hardest tasks."""
        if self.store is None or not self._hard or not hasattr(self.prompts, 'keyword_expand_user'):
            return 0
        self.prompts.require('keyword_expand_user')
        hard = self._hard[:32]
        self.rng.shuffle(hard)
        reqs = list(hard)
        while len(reqs) < self.min_batch:
            reqs.append(hard[len(reqs) % len(hard)])
        self._nonce += 1
        prompts = [{
            'messages': [
                {'role': 'system', 'content': self.prompts.keyword_system},
                {'role': 'user',
                 'content': self.prompts.keyword_expand_user.format(kw=kw, m=8)
                 + f'\n(batch {self._nonce}-{i})'},
            ],
        } for i, (_c, kw) in enumerate(reqs)]
        added = 0
        explorer = self.keyword_explorer or self.explorer
        for (cat, kw), reply in zip(reqs, explorer(prompts,
                                                   sampling_params=self.keyword_params)):
            text = assistant_text(reply)
            parsed = parse_keyword_list(text)
            added += self.store.add(cat, parsed, source='expand', parent=kw)
            if self.keyword_sink is not None:
                self.keyword_sink({
                    'category': cat, 'parent': kw, 'reply': text,
                    'stop_reason': reply.get('stop_reason'),
                    'truncated': bool(reply.get('truncated')),
                    'parsed': parsed, 'n_parsed': len(parsed),
                })
        logger.info(f'[AgenticChallenger] expanded {len(hard)} hard keyword(s) -> '
                    f'+{added} same-domain topics')
        return added
