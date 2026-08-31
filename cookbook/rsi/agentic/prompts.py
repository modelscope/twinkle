# Copyright (c) ModelScope Contributors. All rights reserved.
"""Every string this pipeline sends to a model.

One proposal is one conversation with three stages:

  1. the model acts in a sandbox, one tool call per reply, until it stops calling
     tools (``SYSTEM`` + ``FROM_KEYWORDS``);
  2. a user message carrying the real workspace listing asks for a python check
     script (``CHECK_FOLLOWUP``, once more via ``CHECK_RETRY_FOLLOWUP`` if it does
     not pass);
  3. a user message asks for the problem statement (``PROBLEM_FOLLOWUP``).

Stage 1 runs on the local model and is what gets trained. Stages 2 and 3 are
appended to a copy of that same conversation and answered by the API model, so the
check script and the statement are written with the whole build history in view
without adding untrained tokens to the sample.

The texts below are unchanged from the pipeline this replaced; the comments
keep the measurements that decided their wording, because a prompt whose numbers are
lost is a prompt nobody can edit safely.
"""

# ── Keyword categories ─────────────────────────────────────────────────────
# A proposal draws one entry from each category, and the three are facets of ONE
# task, so combining them yields a single non-trivial computation:
#   transform: the computation the task turns on
#   domain:    the material it runs over
#   edge_case: the twist that makes a naive solution fail

CATEGORIES = ['transform', 'domain', 'edge_case']

# Each category gives three examples, then pushes away from them, then pins the
# answer to what the sandbox can actually build and read back.
#
# The three pins below were each added against a measured miss rate. Counted over
# the 1344 keywords iterations 1-7 put in keywords.jsonl, scored by the regexes in
# .temp/prune_keywords.py so the numbers can be reproduced (they are lower bounds --
# a phrase can be wrong without matching):
#   transform  517 entries, 37% miss: 31% named an ACTIVITY on a running system
#              rather than a computation ("Debug memory leaks in multi-threaded
#              applications", "Monitor system load metrics", "Swap Space
#              Configuration") and 10% needed hardware or kernel access
#   domain     503 entries, 17% miss: 13% named what someone DOES rather than what
#              it is done to ("File format conversion", "Binary data parsing",
#              "network namespace isolation"), despite the existing rule already
#              saying "a FILE FORMAT or a DATA STRUCTURE, never a device"; 4%
#              named a device. Hand-reading a sample puts this category higher than
#              the regex does -- "Optimize server performance" is a domain entry
#              and matches nothing -- so 17% is the floor, not the estimate
#   edge_case  324 entries, 24% needed real hardware or a kernel subsystem
#              ("USB device enumeration delay", "Linux bridge MAC addresses")
# The edge_case number had a plain cause: this category was the only one with no
# container pin at all, so it was free to name devices.
#
# Why this matters downstream: a keyword the container cannot honour does not
# produce a hard task, it produces a pretend one. Across iterations 1-7, 42-70%
# of statements (mean 53%) described themselves as simulating or synthesising
# their own subject matter, which is what "analyse TCP congestion" collapses into
# when there is no TCP stack to look at. The task then tests whether the solver
# can follow a spec for generating fake data.
#
# Not measured: whether these three additions actually lower those rates. They
# are worded to name the failure rather than restate the rule, because the
# existing domain pin shows a rule the generator agrees with and ignores.
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
    'or a service, and never an ACTIVITY carried out on material: "binary data '
    'parsing", "file format conversion" and "traffic analysis" all name something '
    'a person does, not something that sits in a file waiting to be read')
_PINNED_COMPUTATION = (
    ', but only computations that run in that same container: not compiling, not '
    'flashing firmware, not driving hardware. It has to be a FUNCTION of data '
    'that can sit in a file -- given the input there is one right answer, and a '
    'script can recompute it and check it. An activity carried out on a live '
    'system is not one: "debug X", "monitor X", "detect X in real time", '
    '"configure X" and "X strategy" have no answer to check, so name the '
    'calculation instead ("reconstruct the allocation timeline from a heap trace" '
    'rather than "debug memory leaks")')
# edge_case had no pin before, and 18% of what it produced needed a device. A
# twist is only usable if it survives being written down in a file: the solver
# starts in an empty directory and can only be handed data.
_PINNED_EDGE = (
    ', and only a twist that can be REPRODUCED from data in a file: a property of '
    'the input or of the arithmetic over it. Not the behaviour of a device, a '
    'kernel subsystem, a real clock, a network peer or another process -- those '
    'cannot be put in the solver\'s empty directory, so a task built on them can '
    'only pretend')

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
                 'rounding, byte order, cycles in a tree' + _LEAVE_THE_EXAMPLES
                 + _PINNED_EDGE,
}

# ── Keyword generation ─────────────────────────────────────────────────────
# ``parse_keyword_list`` reads a JSON array and returns nothing when it cannot find
# one, so these must ask for a JSON array; keep them in step with the parser.

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

# This prompt writes most of the bank, and until now it was the only one with no
# category rules in it. Of the 1344 keywords iterations 1-7 produced, 960 (71%) came
# from here and 384 from the refill above -- and 31% of the expanded ones break their
# category's rules against 14% of the generated ones. The mechanism is visible in the
# data: asked for keywords related to "Deduce network protocol versions", a legitimate
# computation over captured bytes, it returned "Analyze network traffic patterns",
# "Troubleshoot DNS resolution issues", "Debug TCP/IP stack issues", "Review firewall
# rule sets" and "Monitor honeypot logs" -- each a step further from anything a check
# script can verify. One good keyword decays into eight bad ones, and those eight are
# what later iterations draw from.
#
# So the category description goes in, and with it a sentence saying that being
# related to the parent does not excuse leaving the category. That second part is
# load-bearing: every parent here was chosen for being HARD, and a keyword can be
# hard precisely because the sandbox cannot honour it, in which case following it
# faithfully is the wrong move.
#
# Measured after the change, over iteration 9's 18 refill calls (all via the API,
# ``keyword_gen.jsonl`` 'via' field): 1 of 105 accepted keywords breaks its category's
# rules, against 24% of the bank iterations 1-7 built, and that one is a false positive
# of the scorer ("Amdahl's law speedup bound from parallel workload profile", flagged
# on the noun "profile"). The wording works.
#
# What it broke: 39 of the 144 keywords the model returned (27%) were silently dropped
# by ``parse_keyword_list``, which keeps only strings of 60 characters or less. All but
# one were transform -- five of its six calls came back with a median length of 65-98
# characters, one with all eight over the cap and nothing left. Cause is in this file:
# KEYWORD_USER says "2-5 words" and this prompt only said "short strings", so the
# instruction to name a calculation rather than an activity ("reconstruct the
# allocation timeline from a heap trace") was followed at sentence length. domain came
# back at a 4-23 character median and edge_case at 32-42, both well clear. The cap is
# named here in characters because it is a silent filter in library code: a keyword
# over it does not warn, it just never exists.

KEYWORD_EXPAND_USER = (
    'The keyword "{kw}" produced a very hard task. List {m} related keywords '
    'that might produce similarly challenging but different tasks.\n\n'
    'They belong to this category, whose rules bind them exactly as they bound '
    'the keyword above:\n{desc}\n\n'
    'Being related to "{kw}" does not exempt them. If that keyword itself sits '
    'outside these rules -- and it may, since it was picked only for being hard '
    '-- move back towards the rules instead of following it further out.\n\n'
    'Each must be 2-5 words and at most 60 characters: a topic to build a task '
    'around, not a description of the task. "heap free-list reconstruction" is '
    'one; "reconstruct the heap free-list state from a sequenced alloc/free '
    'trace" is a task statement and will be thrown away. Return ONLY a JSON '
    'array of short strings, nothing else.'
)


# ── Stage 1: the model acts in the sandbox ─────────────────────────────────
# Only shell_executor and python_executor exist -- there is no directory-listing
# tool -- so the prompt names ``ls -R`` explicitly and spells tool names in full.
# One tool call per message: the sampler stops generation at ``</tool_call>``, and a
# reply that plans many calls but is cut after the first would be trained on a
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

# A cap on volume only: the failure it targets is a smaller model running out of
# tokens writing many files, and the thing that must survive is the computation.
# Appended to SYSTEM when --max-build-files > 0, which the loop sets to 4.
BUILD_SIZE_CAP = (
    '\n- Keep the result SMALL: at most {n} files in total, counting inputs, '
    'scripts and outputs. No python package (no __init__.py, no importable '
    'module tree), no command-line interface with subcommands. Depth, not '
    'volume: one non-trivial computation done properly on a small input beats '
    'many files. A task built from this state has to be finishable by a smaller '
    'model in about twenty tool calls.'
)

# The three keywords are drawn independently, one per category, with nothing
# checking that they belong together (``draw_keywords`` takes a random unused entry
# from each). So a proposal regularly gets a triple no honest task covers -- iter7
# produced "TCP Congestion Control" + "Geospatial algorithms" + "hash collision",
# and iter1 "Detect memory leaks in real-time" + "Guitar tablature" + "Thread
# stack fragmentation". Told to exercise all three, the model has one way out:
# invent data that stands in for the parts it cannot have, which is how 42-70% of
# statements (mean 53%) across iterations 1-7 came to describe themselves as
# simulating their own subject matter.
#
# The escape hatch below is deliberately not "ignore a keyword": that would lose
# the diversity the draw exists to create, and the keyword bank's used-marks would
# stop describing what was actually built. Demoting one to background keeps the
# draw meaningful while letting the task be about something real.
#
# Not measured: the effect on the simulate rate, and the cost in diversity if the
# model demotes more often than it needs to. Both are visible in the next run --
# the statements are in tasks.jsonl and the draws in groups.jsonl.
FROM_KEYWORDS = (
    'Your direction for this task:\n{keywords}\n\n'
    'Build something complex and realistic that exercises the topics above. '
    'Work in the '
    'current empty directory, producing files and/or computed output.\n\n'
    'Those three are a starting point, not a checklist. If all three can only be '
    'combined by pretending -- generating fake data to stand in for something '
    'this container cannot have, or inventing a scenario no engineer would meet '
    '-- then let ONE of them stay in the background and build a task the other '
    'two support honestly. A real computation over material you actually '
    'constructed is worth more than a simulation that name-checks everything.'
)

# ── Stage 2: write the check script ────────────────────────────────────────
# Appended to the conversation once the model stops calling tools, so it says
# "you": the same conversation did the work. ``brittle_check_reason`` rejects
# size/checksum/source-text asserts on the syntax tree and sends the script back
# through the rewrite path.
#
# The substring rule used to read "that an expected substring is present", next to a
# separate ban on "script source text". Asserting that a .py file contains
# 'def worker():' satisfies the permission and violates the ban, and on 188 tasks
# from run_clean9 the permission won 51% of the time. Merging the two rules and
# adding the subprocess instruction took that to 0% of 50 tasks, with 84% of the new
# checks running a program instead of reading one.
#
# Then shortened from 475 words to 261 by dropping every sentence that argued FOR a
# rule while keeping the rule. Measured on the same 50 workspaces, temperature 1.0:
#              asserts .py source text   runs a program   input data handed over
#   475 words              0%                 86%                  86%
#   261 words              2%                 90%                  89%
# The noise floor from sampling one prompt twice is 2 points on the source-text rate
# and 14 on handover, so nothing moved.

# The rule about DERIVED values was added last, against a case where every other
# rule was satisfied and the check still did not test the task. An iter7 proposal
# specified a hash table with bucket size 100 and chaining for collisions, but the
# key was (latitude + longitude) % 100 on floats, so 1000 coordinates produced 1000
# distinct keys and not one collision ever happened. Its check asserted the bucket
# count, the threshold, the CSV header and the first coordinate -- all true, all
# shell -- and passed with reward 0.986, so the solver trained on a task whose
# stated subject was never exercised.
#
# Measured over the 533 check scripts of iterations 1-7: median 6 asserts (range
# 2-16, 45% outside the 2-6 the rules ask for) and 46% run a program via
# subprocess. So the shortage is not in volume. Note 46% against the 86-90%
# recorded above: those were measured on run_clean9's workspaces, and here the
# build stage usually leaves its outputs on disk already, so reading them is
# legitimate.
#
# Not added, for lack of evidence: a rule against matching a float by its printed
# digits. It looked like a problem from one example ('58.54579654631016: [0]' in
# content) but only 2 of 533 scripts compare floats without a tolerance, and the
# rest already use abs(got - expected) < eps. A rule earns its words here.

CHECK_FOLLOWUP = (
    'Now write a python script that ASSERTS properties of the state you just '
    'produced. It runs in the same directory you worked in.\n\n'
    "Below is that directory's actual final state: every file as "
    '"path size-in-bytes", then each one\'s contents. This is the ground truth, '
    'not your account of what you did. Assert only about paths and content '
    'visible here. If there is nothing worth testing, say UNTESTABLE and write '
    'no code.\n\n'
    '{final_state}\n\n'
    'The solver will be told what its program must DO and writes its own code, '
    'so two correct programs share their behaviour and nothing else.\n\n'
    'Rules:\n'
    '- 2-6 asserts, standard library only.\n'
    '- Assert only about files holding RESULTS. Never about the text of a '
    'program: not a line it contains, not a name it mentions, not its length.\n'
    '- At least one assert must pin a DERIVED value: something no one could '
    'write down without doing the computation -- a total, an ordering, a decoded '
    'field, a solved quantity. Existence of a file, a header row, a column name '
    'and a value copied from the input are all shell: a program that produced '
    'them and got the arithmetic wrong must still fail this script.\n'
    '- If a result only exists once a program runs, RUN it -- '
    'subprocess.run([sys.executable, "thing.py"], capture_output=True, '
    'text=True) -- and assert on what it printed or the files it left.\n'
    '- No exact bytes: no sizes, no checksums, no whole-file equality, no '
    'timestamps.\n'
    '- No claim about the directory as a whole: not the file count, not that '
    'nothing else exists.\n'
    '- Never write a number you did not read above.\n'
    '- A truncated file holds more than you can see: assert about the shown '
    'part, not its end or its length.\n'
    '- Still discriminating: it must fail for a directory that does not hold '
    'this state.\n'
    '- Exit 0 when every assertion holds, non-zero otherwise.\n'
    '- Do NOT call any tool now. Return ONLY a fenced python code block, no '
    'prose.'
)

# ── Stage 2b: the one chance to fix a check that did not pass ──────────────
# "Drop an assertion you cannot make true" and the new DERIVED rule pull against
# each other: the assert most likely to fail here is exactly the derived one, since
# the shell asserts (a path exists, a header matches) were already true when they
# were written. Dropping it is the cheapest way to make the script pass, and it
# lands back at the check that tests nothing. Hence the carve-out below.

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
    'One assertion you may not drop: the one pinning a computed value. If it is '
    'the one that failed, correct it against the listing -- read the value there '
    'and assert that -- because a script left asserting only paths, headers and '
    'input values passes for a program that got the computation wrong.\n\n'
    'Same rules as before: standard library only, 2-6 asserts, no file sizes, '
    'checksums, timestamps, script source text, whole-file exact-string '
    'equality, or claims about the exact set of files in the directory. Keep it '
    'robust -- it must pass for any correct reproduction of this state -- yet '
    'still fail for a directory that does not hold it. Exit 0 exactly when the '
    'state is right. Do NOT call any tool now. Return ONLY a fenced python code '
    'block, no prose.'
)

# ── Stage 3: the statement gives the rules, never the computed answer ──────
# The end state is split in two: input data verbatim (it is not the answer), and
# everything derived given as the rule that produces it -- otherwise the only way to
# state what a derived file must contain is to quote the computed answer.
#
# Stating the split as a rule is not enough: over run_clean9's 154 tasks whose check
# compares against a computed-looking value, 52% of statements carry EVERY one of
# them (mean share 0.72). Listing the values instead of describing them was tried on
# 50 tasks in two forms and both differences sat inside the noise floor (0.059,
# p=0.10 to 0.53), so the wording is unchanged and the leak rate is a known open
# problem. A forbidden list can also hide INPUT data, which makes a task unsolvable,
# and that cost is invisible to every offline metric.
#
# Shortened from 328 words to 236 in the same round as CHECK_FOLLOWUP. Measured on
# the same 50 workspaces, temperature 1.0:
#                  input data handed over   leak   statement words p50
#   328 words               86%             0.57           208
#   236 words               91%             0.64           201
#   236 + short check       86%             0.54           198
# Handover moved 5 points against a 14-point same-prompt spread and the leak 0.07
# against 0.059, p=0.50 -- a length change that cost nothing measurable.

PROBLEM_FOLLOWUP = (
    'Your checks pass on the state you produced. Now write the task description '
    'another AI agent would be given to reproduce that same end state.\n\n'
    'It starts in an EMPTY directory and sees nothing but your statement: every '
    'file that must be there at the end has to be created by it.\n\n'
    'Give the two halves differently:\n'
    '- INPUT data, the raw material nothing was computed from yet: verbatim, '
    'exact filenames and exact contents, so it can be written byte for byte. '
    'Only passive data counts as input -- a CSV, a JSON config, a binary record '
    'file, a text corpus. Source code is NEVER input data: do not quote the '
    'body of any script you wrote.\n'
    '- Everything DERIVED from it -- computed values, aggregates, orderings, '
    'reports: only the RULE that produces it. Name the output file and its '
    'format, say how each part follows from the input, and never state the '
    'resulting value, not even as an example.\n\n'
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
