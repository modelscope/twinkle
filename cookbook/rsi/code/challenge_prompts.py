# Copyright (c) ModelScope Contributors. All rights reserved.
"""Prompts for the code challenger.

Every string the model sees during self-play for code tasks, in one file,
because the prompt *is* the experiment: two runs that differ here are not
comparable, and a run has to be able to say which wording produced its data.
:class:`twinkle_agentic.challenger.CodePrompts` holds no defaults for exactly
that reason.

Text carried over verbatim from the previous ``rsi_challenge.py``, including the
findings recorded next to it, so numbers from earlier runs stay comparable.
"""
from twinkle_agentic.challenger import CodePrompts

# Categories of the keyword bank. One keyword is drawn from each per proposal,
# so the challenger has to bridge an algorithm, a computing concept and a
# real-world domain instead of falling back on palindromes and bracket matching.
CATEGORIES = ('algorithm', 'computer', 'noncs')

CATEGORY_DESC = {
    'algorithm': 'algorithmic techniques and paradigms (e.g. dynamic programming, binary '
                 'search, union-find, Dijkstra, backtracking, segment trees, greedy, '
                 'divide and conquer, sliding window ...)',
    'computer': 'computer-science / computing concepts that are NOT algorithms per se '
                '(e.g. hash maps, tries, LRU cache, bitsets, regular expressions, base '
                'conversion, finite state machines, serialization, parsing, memoization ...)',
    'noncs': 'real-world domains OUTSIDE computer science, used to give a problem flavor '
             '(e.g. biology, finance, chemistry, logistics, music, cooking, sports, '
             'astronomy, geography, linguistics ...)',
}

# The output contract. It names the four keys parse_challenge() reads back, and
# the "do NOT write the expected value" line is what makes the ground truth come
# from execution rather than from the model's own guess about its own code.
CHALLENGER_SYSTEM = (
    'You design self-contained Python coding problems for training another model.\n'
    'A good problem: (1) is solvable from its statement ALONE with no external files, '
    'network, images, or hidden context; (2) has ONE clear entry function; (3) is '
    'deterministic (same input -> same output), no randomness, no wall-clock, no threads; '
    '(4) is neither trivial nor impossible for a mid-size model.\n'
    'You will also write the reference solution. We will EXECUTE it to obtain the '
    'ground-truth outputs, so your solution must be correct and runnable as-is.\n'
    'Return ONLY one JSON object, no prose around it, with keys:\n'
    '  "problem":  the statement shown to the solver (describe the function name, its '
    'inputs and expected behavior; do NOT include the solution).\n'
    '  "solution": the reference implementation as plain Python source (no markdown fence).\n'
    '  "entry":    the entry function name.\n'
    '  "checks":   a list of 3-6 Python expressions calling the entry function on concrete '
    'inputs (e.g. "solve([1,2,3])"); each must be evaluable after running the solution. '
    'Do NOT write the expected value — we compute it by running your solution.'
)

FROM_SCRATCH = (
    'Create ONE new Python coding problem now. Vary the topic freely '
    '(strings, arrays, math, greedy, DP, parsing, simulation ...).'
)

FROM_SEED = (
    'Here is a seed problem. Create ONE NEW problem that is a meaningful VARIANT of it '
    '(change the twist, constraints, or data shape — not just renaming), keeping it '
    'self-contained and deterministic.\n\n[seed]\n{seed}'
)

FROM_KEYWORDS = (
    'Create ONE new Python coding problem now. Draw inspiration from the following '
    'topic(s) and combine them creatively into a single coherent problem:\n{keywords}\n'
    'You may use each topic directly or bend it loosely; combine with any data shape '
    '(strings, arrays, grids, trees, numbers, parsing, simulation ...). Make it require '
    'real thought, not a one-liner, and keep it self-contained and deterministic.'
)

# Seed AND keywords together. The seed is deliberately framed as inspiration only,
# not as something to produce a variant of: the point is to pull the generated
# problems toward the shape of public benchmark items (short statement, one plain
# task) while the keywords keep supplying topical variety.
FROM_SEED_KEYWORDS = (
    'Create ONE new Python coding problem now. Use the problem below only as a '
    'STARTING POINT for inspiration — you do NOT have to keep its task, and the new '
    'problem does NOT need to be a variant of it.\n\n[inspiration]\n{seed}\n\n'
    'Also draw on the following topic(s), combining them into a single coherent '
    'problem:\n{keywords}\n'
    'Make it require real thought, not a one-liner, and keep it self-contained and '
    'deterministic.'
)

# ── two-step proposing ──────────────────────────────────────────────────────
# The difficulty comes from adding a layer on top of a real, runnable reference
# solution, not from imagining a hard problem outright; splitting into two calls
# (write the harder code, THEN describe it) keeps the statement and the ground
# truth consistent, which a single call does not. Measured on 40 MBPP seeds
# against the single-call seed+keywords prompt: kept-rate 25% vs 15%,
# constant-answer problems 4 vs 7, similarity to the seed 0.42.
TWO_STEP_SYSTEM = 'You are an expert Python programmer.'

TWO_STEP_SOLUTION = (
    'Below is a coding problem and its reference solution.\n\n'
    '[problem]\n{seed}\n\n[reference solution]\n{code}\n\n'
    'Write a MORE COMPLEX Python function that keeps the idea of the reference solution '
    'as one step and builds a harder computation around it (extra pass, different data '
    'structure, an added rule), in the direction of these topic(s):\n{keywords}\n'
    'Requirements: deterministic, self-contained, no randomness, no I/O, one clear entry '
    'function. Output ONLY the code in a single ```python block, no explanation.'
)

# Showing the seed here pulls the wording back toward the MBPP task family
# (similarity 0.32 -> 0.42). The solution is NOT taken from this JSON -- the
# challenger overwrites it with the code the first call produced, so the ground
# truth matches what was actually executed.
TWO_STEP_PROBLEM = (
    'Here is a Python function.\n\n```python\n{code}\n```\n\n'
    'It was written as a harder follow-up to this exercise:\n\n[original exercise]\n'
    '{seed}\n\nand it was pushed in the direction of these topic(s):\n{keywords}\n\n'
    'Write the problem statement that the function above is the answer to, as if it were '
    'a coding exercise in the same series as the original: name the entry function, '
    'describe its inputs and the exact behaviour expected, and do NOT reveal the '
    'implementation. Phrase it as plainly and briefly as the original exercise.\n'
    'Return ONLY one JSON object, no prose around it, with keys:\n'
    '  "problem":  the statement shown to the solver.\n'
    '  "entry":    the entry function name.\n'
    '  "checks":   a list of 3-6 Python expressions calling the entry function on '
    'concrete inputs; each must be evaluable after running the function above. Do NOT '
    'write the expected value.\n'
    'The "solution" is already known, so do not include it.'
)

# ── keyword bank ────────────────────────────────────────────────────────────
KEYWORD_SYSTEM = 'You brainstorm diverse topics for a Python coding-problem generator.'

KEYWORD_USER = (
    'List {k} DISTINCT and SPECIFIC topics from this category: {desc}\n'
    'Be creative and concrete; avoid vague umbrella words. '
    'Return ONLY a JSON array of short strings, nothing else.'
)

KEYWORD_EXPAND_USER = (
    'The topic "{kw}" turned out to seed genuinely HARD problems. List {m} MORE distinct, '
    'specific topics in the SAME family/domain as "{kw}" that could seed similarly '
    'challenging Python problems. Return ONLY a JSON array of short strings, nothing else.'
)

# ── solver ──────────────────────────────────────────────────────────────────
# Used both to measure difficulty and, as the system half, as the system prompt
# of the task that gets stored: training against a different one than the
# difficulty measurement used would make the measurement mean nothing.
SOLVER_SYSTEM = 'You are an expert Python programmer.'

SOLVER_USER = (
    '{problem}\n\n'
    'Write the complete Python solution. Put the final code in a single ```python fenced '
    'block. Define the exact function name required by the problem.'
)


def code_prompts() -> CodePrompts:
    """Assemble the strings above into the object the challenger takes."""
    return CodePrompts(
        system=CHALLENGER_SYSTEM,
        from_scratch=FROM_SCRATCH,
        from_seed=FROM_SEED,
        from_keywords=FROM_KEYWORDS,
        from_seed_keywords=FROM_SEED_KEYWORDS,
        two_step_system=TWO_STEP_SYSTEM,
        two_step_solution=TWO_STEP_SOLUTION,
        two_step_problem=TWO_STEP_PROBLEM,
        keyword_system=KEYWORD_SYSTEM,
        keyword_user=KEYWORD_USER,
        keyword_expand_user=KEYWORD_EXPAND_USER,
        solver_system=SOLVER_SYSTEM,
        solver_user=SOLVER_USER,
    )
