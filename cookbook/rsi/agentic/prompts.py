# Copyright (c) ModelScope Contributors. All rights reserved.
"""Prompts for the agentic challenger.

One conversation, three stages:
  Stage 1: the model acts as an agent in a sandbox (multi-turn with tools),
           producing a tool-call chain and a final workspace state, and stops
           calling tools.
  Stage 2: a user message carrying the real workspace listing is appended to that
           same conversation, asking for a python check script.
  Stage 3: another user message asks for the problem statement.

Stages 2 and 3 are appended rather than sent as fresh calls, so the whole chain
is one sample whose every assistant turn can be trained on. That is also why
their rules live in user messages: a conversation has one system message, and it
was already spent on stage 1.

The check script is run against the sandbox immediately after stage 2 to verify
it passes; any task whose own checks fail is thrown away.

Keyword categories are configurable; the framework (``KeywordStore`` + draw/combine
logic) is category-agnostic. A proposal draws one entry from each category, and the
three are facets of ONE task so combining them yields a single non-trivial
computation:
  - transform: the computation the task turns on
  - domain: the material it runs over (data AND code/compilation)
  - edge_case: the twist that makes a naive solution fail
"""
from twinkle_agentic.challenger import AgenticPrompts

# ── Keyword categories (framework-agnostic; only the content is scenario-specific) ─

CATEGORIES = ['transform', 'domain', 'edge_case']

# Each category gives three examples, then pushes away from them, then pins the
# answer to what the sandbox can actually build and read back.
_LEAVE_THE_EXAMPLES = (
    '. These three are only to show the form of an answer -- do NOT stay '
    'near them; name things from as many different areas of computer '
    'engineering as you can')
_PINNED_TO_CONTAINER = (
    ', but only material that can actually be BUILT and read back inside a Linux '
    'container that has python (numpy, pandas, pillow, scipy, sympy, networkx, '
    'openpyxl, xlsxwriter, pypdf, pymupdf, pdfplumber, python-docx, python-pptx, '
    'reportlab, lxml, pyarrow, matplotlib), sqlite3, ffmpeg, imagemagick, git, '
    'jq, tar/zip/7z, poppler-utils and pip -- and NO compiler, no GPU, no docker, '
    'no hardware devices. Name a FILE FORMAT or a DATA STRUCTURE, never a device '
    'or a service')
_PINNED_COMPUTATION = (
    ', but only computations that run in that same container: not compiling, not '
    'flashing firmware, not driving hardware')

CATEGORY_DESC = {
    'transform': 'a specific, non-trivial transformation the solver must COMPUTE '
                 'rather than copy -- the answer is derived, never stated. For '
                 'example: solve a system of equations with sympy, decode a '
                 'binary format, find a shortest path' + _LEAVE_THE_EXAMPLES
                 + _PINNED_COMPUTATION,
    'domain': 'the kind of material the task operates on. For example: WAV audio '
              'files, a SQLite database, PNG images' + _LEAVE_THE_EXAMPLES
              + _PINNED_TO_CONTAINER,
    'edge_case': 'a twist that makes a naive or copy-the-statement solution fail '
                 'and forces careful handling. For example: floating-point '
                 'rounding, byte order, cycles in a tree' + _LEAVE_THE_EXAMPLES,
}

# ── Stage 1: model acts in sandbox ─────────────────────────────────────────
# Only shell_executor and python_executor exist -- there is no directory-listing
# tool -- so the prompt names ``ls -R`` explicitly and spells tool names in full.
# One tool call per message: the sampler stops generation at ``</tool_call>``, and
# a reply that plans many calls but is cut after the first would be trained on a
# reasoning that does not match what happened.

SYSTEM = (
    'You are an expert developer working in an empty directory with a shell and '
    'python. Your job is to build something complex and realistic, one tool call '
    'at a time, based on the direction given below. Use '
    'the tools available to you (shell commands, python scripts, file '
    'operations) to produce a meaningful end state: files '
    'with content, computed outputs, structured data.\n\n'
    'Requirements:\n'
    '- Work entirely within the current directory (do not use /tmp or ~).\n'
    '- Do not use the network: no downloads, no web requests, and no installing '
    'packages (no pip install, no apt install). The sandbox has no internet, so '
    'any such call wastes a turn; build only with the Python standard library and '
    'the packages already installed (numpy, pandas, matplotlib, scikit-learn, '
    'pyarrow, and other common data libraries).\n'
    '- Make sure the end state is deterministic: the same steps always produce '
    'the same files with the same content.\n'
    '- Make exactly ONE tool call per message, then read what it returned before '
    'choosing the next one. Take as many turns as the work needs.\n'
    '- Verify your own work before finishing: list the directory and read back '
    'what you wrote. A file you meant to create but did not is worse than a '
    'smaller result, because the task built from this state will be impossible.\n'
    '- To see what is in the directory, run shell_executor with "ls -R", which '
    'shows files and directories at every depth.\n'
    '- When you are satisfied with the result, stop calling tools and say '
    '"Done." as your final message.'
)

FROM_SCRATCH = (
    'Build something complex and realistic in the current empty directory. '
    'Create files, write scripts, '
    'process data -- whatever demonstrates '
    'competent use of the tools. Take as many turns as the work needs, each '
    'building on the last.'
)

FROM_SEED = (
    'Here is an example of the kind of task we want:\n\n{seed}\n\n'
    'Build something in the same spirit but on a '
    'different subject, equally complex and realistic. Change what '
    'is produced and how, not just the names. Work in the current empty directory.'
)

FROM_KEYWORDS = (
    'Your direction for this task:\n{keywords}\n\n'
    'Build something complex and realistic that exercises the topics above. '
    'Work in the '
    'current empty directory, producing files and/or computed output.'
)

FROM_SEED_KEYWORDS = (
    'Here is an example task for inspiration:\n\n{seed}\n\n'
    'Your direction keywords:\n{keywords}\n\n'
    'Build something complex and realistic that combines the spirit of the '
    'example with the keyword topics. '
    'Work in the current empty directory.'
)

# ── Stage 2: write the check script ────────────────────────────────────────
# Appended to the episode as a user message once the model stops calling tools,
# so it says "you": the same conversation did the work. brittle_check_reason() in
# challenger/agentic.py rejects size/checksum/source-text asserts on the syntax
# tree and sends the script back through the rewrite path.

CHECK_FOLLOWUP = (
    'Now write a python script that ASSERTS properties of the state you just '
    'produced. It will be run in the same directory you worked in.\n\n'
    'Here is the actual final state of that directory: first every file as '
    '"path size-in-bytes", then the contents of each one. This listing is the '
    'ground truth, not your account of what you did. Assert only about paths '
    'that appear in it, and only about content you can read here. If it is empty '
    'or shows nothing worth testing, say UNTESTABLE and write no code.\n\n'
    '{final_state}\n\n'
    'Rules:\n'
    '- 2-6 asserts, standard library only.\n'
    '- Make the check ROBUST and BROAD: it must pass for ANY correct '
    'reproduction of this state, and fail only for one that got the work wrong. '
    'Assert meaning, not form -- that a file exists, that it parses, that a value '
    'or a row read out of it is right, that an expected substring is present.\n'
    '- Do NOT pin exact bytes: no file sizes, no checksums, no asserting that a '
    'whole file equals one exact string, no timestamps, no script source text. A '
    'different correct solution writes different bytes and would fail such a '
    'check even though it is right.\n'
    '- Do NOT constrain the directory as a whole: never assert the exact number '
    'of files, or that no other files exist. Check only the files that carry the '
    'result and ignore the rest.\n'
    '- Never write down a number you did not read above -- do not recompute a '
    'mean, a count or a checksum in your head.\n'
    '- A file shown truncated has more content than you can see: assert about the '
    'part you were shown, not its end or its length.\n'
    '- Still discriminating: what you keep must fail for a directory that does '
    'not hold this state. Robust does not mean empty.\n'
    '- Exit 0 when every assertion holds, non-zero otherwise.\n'
    '- Do NOT call any tool now. Return ONLY a fenced python code block, no '
    'prose.'
)

# ── Stage 2b: the one chance to fix a check that did not pass ──────────────

CHECK_RETRY_FOLLOWUP = (
    'That script does not pass. Running it in that directory gave:\n\n'
    '{error}\n\n'
    'Nothing has changed in the directory; this is what it holds:\n\n'
    '{final_state}\n\n'
    'Rewrite the script so that it passes. Where your assertion and this listing '
    'disagree, the listing is what is there and the assertion is what is wrong -- '
    'fix the assertion, do not assert something new that you still cannot read '
    'here. Drop an assertion you cannot make true instead of weakening every one '
    'of them; what stays must still fail for a directory that does not hold this '
    'state.\n\n'
    'Same rules as before: standard library only, 2-6 asserts, no file sizes, '
    'checksums, timestamps, script source text, whole-file exact-string '
    'equality, or claims about the exact set of files in the directory. Keep it '
    'robust -- it must pass for any correct reproduction of this state -- yet '
    'still fail for a directory that does not hold it. Exit 0 exactly when the '
    'state is right. Do NOT call any tool now. Return ONLY a fenced python code '
    'block, no prose.'
)


# ── Keyword generation ─────────────────────────────────────────────────────
# ``parse_keyword_list`` reads a JSON array and returns nothing when it cannot
# find one, so these must ask for a JSON array; keep them in step with the parser.

KEYWORD_SYSTEM = (
    'You generate diverse topic keywords for training an AI agent that does '
    'computer engineering work in a Linux sandbox: writing and running programs, '
    'building, testing and debugging software, processing and analysing data, '
    'and administering files and the system.'
)

KEYWORD_USER = (
    'List {k} diverse, specific topic keywords for the following category:\n'
    '{desc}\n\n'
    'Each should be 2-5 words, concrete enough to inspire a specific task. '
    'Return ONLY a JSON array of short strings, nothing else.'
)

KEYWORD_EXPAND_USER = (
    'The keyword "{kw}" produced a very hard task. List {m} related keywords '
    'in the same domain that might produce similarly challenging but different '
    'tasks. Return ONLY a JSON array of short strings, nothing else.'
)


# ── Arm C: cap what one episode may build ──────────────────────────────────
# A cap on volume only: the failure it targets is a smaller model running out of
# tokens writing many files, and the thing that must survive is the computation.

BUILD_SIZE_CAP = (
    '\n- Keep the result SMALL: at most {n} files in total, counting inputs, '
    'scripts and outputs. No python package (no __init__.py, no importable '
    'module tree), no command-line interface with subcommands. Depth, not '
    'volume: one non-trivial computation done properly on a small input beats '
    'many files. A task built from this state has to be finishable by a smaller '
    'model in about twenty tool calls.'
)

# ── Stage 3: the statement gives the rules, never the computed answer ────────
# The end state is split in two: input data verbatim (it is not the answer), and
# everything derived given as the rule that produces it -- otherwise the only way
# to state what a derived file must contain is to quote the computed answer.

PROBLEM_FOLLOWUP_RULES_ONLY = (
    'Your checks pass on the state you produced. Now write the task description '
    'another AI agent would be given to reproduce that same end state.\n\n'
    'That agent starts in an EMPTY directory and sees nothing but your '
    'statement: every file that must be there at the end has to be created by '
    'it.\n\n'
    'Give the two halves differently:\n'
    '- INPUT data, the raw material nothing was computed from yet: verbatim, '
    'exact filenames and exact contents, so it can be written byte for byte.\n'
    '- Everything DERIVED from it -- computed values, aggregates, orderings, '
    'resolved references, reports: only the RULE that produces it. Name the '
    'output file and its format, say how each part follows from the input, and '
    'never state the resulting value. Not as an example, not in a sample of the '
    'output. A statement that writes out what you computed can be satisfied by '
    'copying it, and then it measures typing.\n\n'
    'Rules:\n'
    '- Be specific about formats, filenames and layout.\n'
    '- Say what must be true of the result, not which commands to run.\n'
    '- Do NOT mention the checks or how verification works.\n'
    '- Self-contained: no reference to this conversation or to anything the '
    'reader cannot see.\n'
    '- 300 words or less, not counting input data quoted verbatim.\n'
    '- Do NOT call any tool now. Return ONLY the problem statement as plain '
    'text, no code fences.'
)


# ── Factory ────────────────────────────────────────────────────────────────

def agentic_prompts(max_build_files: int = 0) -> AgenticPrompts:
    """Assemble all strings into the object the challenger takes.

    ``max_build_files`` is the one knob: when > 0 it appends BUILD_SIZE_CAP to the
    system prompt.
    """
    system = SYSTEM
    if max_build_files > 0:
        system = system + BUILD_SIZE_CAP.format(n=max_build_files)
    return AgenticPrompts(
        system=system,
        from_scratch=FROM_SCRATCH,
        from_seed=FROM_SEED,
        from_keywords=FROM_KEYWORDS,
        from_seed_keywords=FROM_SEED_KEYWORDS,
        check_followup=CHECK_FOLLOWUP,
        check_retry_followup=CHECK_RETRY_FOLLOWUP,
        problem_followup=PROBLEM_FOLLOWUP_RULES_ONLY,
        keyword_system=KEYWORD_SYSTEM,
        keyword_user=KEYWORD_USER,
        keyword_expand_user=KEYWORD_EXPAND_USER,
    )
