# Copyright (c) ModelScope Contributors. All rights reserved.
"""Prompts for the agentic challenger.

The agentic challenger works in three rounds:
  Round 1: model acts as an agent in a sandbox (multi-turn with tools),
           producing a tool-call chain and final workspace state.
  Round 2a: model sees the trajectory and writes a python check script
           that asserts properties of the final state.
  Round 2b: model sees trajectory + checks and writes a problem statement
           that someone else would need to follow to reproduce the result.

The check script is run against the sandbox immediately after round 1 to verify
it passes; any task whose own checks fail is thrown away. This is the agentic
analogue of the code challenger running the reference solution against its own
asserts.

Keyword categories and directions are configurable. The defaults below exercise:
  - filesystem: creating, moving, reading, transforming files
  - scripting: writing python/shell scripts that produce output
  - data: CSV/JSON/text parsing and aggregation
"""
from twinkle_agentic.challenger import AgenticPrompts

# ── Keyword categories (analogous to code's algorithm/computer/noncs) ──────

CATEGORIES = ['filesystem', 'scripting', 'data']

CATEGORY_DESC = {
    'filesystem': 'file and directory manipulation tasks: creating directory trees, '
                  'moving/renaming files by pattern, finding files by content, '
                  'generating structured text files',
    'scripting': 'tasks that require writing a script (python or shell) whose output '
                 'or side effects are the goal: number crunching, text transformation, '
                 'format conversion, small utilities',
    'data': 'tasks involving structured data: parsing CSV/JSON/YAML, aggregating '
            'rows, filtering records, joining multiple files, producing summary '
            'reports or reformatted output',
}

# ── Round 1: model acts in sandbox ─────────────────────────────────────────

SYSTEM = (
    'You are an expert developer working in an empty directory with a shell and '
    'python. Your job is to do something interesting and non-trivial based on '
    'the direction given below. Use the tools available to you (shell commands, '
    'python scripts, file operations) to produce a meaningful end state: files '
    'with content, computed outputs, structured data.\n\n'
    'Requirements:\n'
    '- Work entirely within the current directory (do not use /tmp or ~).\n'
    '- Do not use the network.\n'
    '- Make sure the end state is deterministic: the same steps always produce '
    'the same files with the same content.\n'
    '- Do at least 2-3 distinct steps, not just one command.\n'
    '- When you are satisfied with the result, stop calling tools and say '
    '"Done." as your final message.'
)

FROM_SCRATCH = (
    'Do something interesting and non-trivial in the current empty directory. '
    'Create files, write scripts, process data -- whatever demonstrates '
    'competent use of the tools. Aim for 2-4 steps that build on each other.'
)

FROM_SEED = (
    'Here is an example of the kind of task we want:\n\n{seed}\n\n'
    'Do something in the same spirit but on a different subject. Change what '
    'is produced and how, not just the names. Work in the current empty directory.'
)

FROM_KEYWORDS = (
    'Your direction for this task:\n{keywords}\n\n'
    'Do something interesting that exercises the topics above. Work in the '
    'current empty directory, producing files and/or computed output.'
)

FROM_SEED_KEYWORDS = (
    'Here is an example task for inspiration:\n\n{seed}\n\n'
    'Your direction keywords:\n{keywords}\n\n'
    'Do something that combines the spirit of the example with the keyword '
    'topics. Work in the current empty directory.'
)

# ── Round 2a: write check script ───────────────────────────────────────────

CHECK_SYSTEM = (
    'You are a test engineer. Given a record of what an agent did in a directory '
    'and the resulting state, write a python script that ASSERTS properties of '
    'the end state. The script will be run in the same directory the agent worked '
    'in.\n\n'
    'Rules:\n'
    '- Use only the standard library (os, json, csv, re, pathlib, etc.).\n'
    '- Write 2-6 assert statements that verify the most important outcomes.\n'
    '- Each assert should check something observable: file existence, file '
    'content, computed values, directory structure.\n'
    '- The script must exit 0 when all assertions hold and non-zero otherwise.\n'
    '- Do NOT import anything that is not in the python standard library.\n'
    '- Do NOT use the network or read from outside the working directory.\n'
    '- Return ONLY a fenced python code block, no prose.'
)

CHECK_USER = (
    'Here is what the agent did:\n\n{trajectory}\n\n'
    'Here is the final state of the working directory:\n\n{final_state}\n\n'
    'Write a python check script (fenced code block) that asserts the key '
    'properties of this end state. 2-6 assertions.'
)

# ── Round 2b: write problem statement ─────────────────────────────────────

PROBLEM_SYSTEM = (
    'You write task descriptions for an AI agent. Given a record of what was '
    'done and the checks that verify it, write a clear problem statement that '
    'another agent would need to follow to reproduce the same end state.\n\n'
    'Rules:\n'
    '- State exactly what files must exist and what they must contain.\n'
    '- Be specific about formats, names, and expected values.\n'
    '- Do NOT reveal the solution steps -- only describe the desired end state.\n'
    '- Do NOT mention the checks or how verification works.\n'
    '- The statement must be self-contained: no references to prior context.\n'
    '- Keep it concise: 50-300 words.\n'
    '- Return ONLY the problem statement as plain text, no code fences.'
)

PROBLEM_USER = (
    'Here is what the agent did:\n\n{trajectory}\n\n'
    'Here are the check assertions that verify the end state:\n\n'
    '```python\n{checks}\n```\n\n'
    'Write a problem statement (plain text, 50-300 words) describing what '
    'another agent must produce to pass these checks. Do not reveal the '
    'solution steps.'
)

# ── Keyword generation ─────────────────────────────────────────────────────

KEYWORD_SYSTEM = (
    'You generate diverse topic keywords for training an AI agent that works '
    'with files, scripts, and data in a local directory.'
)

KEYWORD_USER = (
    'List {k} diverse, specific topic keywords for the following category:\n'
    '{desc}\n\n'
    'Return one keyword per line, no numbering, no explanation. '
    'Each should be 2-5 words, concrete enough to inspire a specific task.'
)

KEYWORD_EXPAND_USER = (
    'The keyword "{kw}" produced a very hard task. List {m} related keywords '
    'in the same domain that might produce similarly challenging but different '
    'tasks. One per line, no numbering.'
)


# ── Factory ────────────────────────────────────────────────────────────────

def agentic_prompts() -> AgenticPrompts:
    """Assemble all strings into the object the challenger takes."""
    return AgenticPrompts(
        system=SYSTEM,
        from_scratch=FROM_SCRATCH,
        from_seed=FROM_SEED,
        from_keywords=FROM_KEYWORDS,
        from_seed_keywords=FROM_SEED_KEYWORDS,
        check_system=CHECK_SYSTEM,
        check_user=CHECK_USER,
        problem_system=PROBLEM_SYSTEM,
        problem_user=PROBLEM_USER,
        keyword_system=KEYWORD_SYSTEM,
        keyword_user=KEYWORD_USER,
        keyword_expand_user=KEYWORD_EXPAND_USER,
    )
