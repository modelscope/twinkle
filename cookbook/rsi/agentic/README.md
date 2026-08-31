# agentic — RSI self-play where one trajectory is one request

One model plays both roles. It builds something in a sandbox, then writes a task
description for what it built, then tries to redo that task from the description
alone. How often it succeeds is what scores the description: a task the solver
passes sometimes is worth training on, one it always or never passes is not.

This replaced an earlier version of the same method whose difference was
scheduling: there a round of proposals moved through the pipeline as a batch and
every stage waited for the slowest member. Here each trajectory is its own request
from start to finish, and the only place anything waits is the last step, deciding
whether a group of eight is worth keeping. (The old version was retired to
`.temp/agentic_legacy`; nothing here imports from it.)

## Three resources, three queues

| resource | how many at once | who queues on it |
|---|---|---|
| sandbox | `--sandbox-slots` microVMs (32) | one job owns one slot from the workspace clear to its last check |
| vLLM | `enable_continous_work` routes each request to the least busy worker | every build turn and every solver turn, one trajectory per request |
| API | `--api-concurrency` (32) | check scripts, problem statements, the rubric |

There is one FIFO job queue and one thread per sandbox slot, so a slot is never
idle while there is work. A build that finishes hands its statement to eight
solver jobs, releases its slot, and returns — it never waits for its own solvers,
which is what would deadlock a pool against itself. Rubric jobs go to a separate
pool because they need no sandbox.

A batch of one is a first-class vLLM call here. `challenge.py` refuses to start if
the sampler does not advertise `enable_continous_work`, because without it a batch
of one is padded up to the worker count and most of every generation is thrown
away.

## One proposal, three stages

1. **Build.** Local model, sandbox tools, one tool call per reply, up to
   `--max-turns`. This is the trainable part: the trajectory keeps exactly the
   tokens the local model produced.
2. **Check script.** The workspace is read back byte for byte and appended to a
   *copy* of the build conversation; qwen3.8-max writes a python script that
   asserts the end state. It is rejected on the syntax tree if it pins file sizes,
   checksums or a script's source text, then run in the sandbox. One rewrite.
3. **Problem statement.** Same copy, one more API reply: input data verbatim,
   everything derived given as the rule that produces it.

Stages 2 and 3 run on the API so the check and the statement are written with the
whole build history in view without adding untrained tokens to the sample.

## Groups, and the one place things wait

A group is `--group-size` (8) proposals sharing one keyword draw and one prompt.
That is what makes it a GRPO group: a proposal's advantage is its reward minus the
mean over the others answering the same prompt.

- Each proposal's task gets `--solver-rollouts` (8) attempts. `n_pass` is how many
  passed, with the denominator fixed at 8 — a truncated attempt is a failed
  attempt, the same as one whose assertions failed.
- Once all eight builds are in, the eight statements are scored for novelty
  *against each other* plus the closest entries in the task bank. Waiting for all
  eight costs nothing: the slots are held by other groups' jobs the whole time. If
  a statement still has no verdict after `--novelty-tries` (3), the group is
  dropped and its queued solver attempts are skipped.
- **The group is kept when at least one proposal has `n_pass` in `[1, 7]`.** The
  other seven may be anything, including builds that produced no task at all;
  they train with the reward they earned, which for those is 0.
- From a kept group the highest-reward in-band proposal is selected, and its eight
  solver attempts are what the solver side trains on. The unselected proposals'
  attempts were measured and are reported, but not trained on.

Eight kept groups give 64 proposing and 64 solving trajectories: one training step.

## Reward

Proposing side, unchanged from where it was measured:

```
reward = exp(-(n_pass/8 - 0.2)^2 / (2 * 0.3^2)) * (floor + (1-floor) * novelty)
reward = 0                                       when n_pass is 0 or unmeasured
```

The gaussian peaks at a pass rate of 0.2, not 0.5: a proposal only teaches the
solver something when the solver mostly cannot do it yet. The floor at
`n_pass <= 0` is load-bearing — the gaussian at p=0 is 0.801, higher than the
0.607 it gives a proposal half the attempts solve, so without the gate the best
thing a proposer could do is write tasks nobody can finish.

Note that being out of band does not zero the reward. A task everybody solves
still earns about 0.03. Out of band decides whether the task is delivered to the
solver side; it does not zero the proposer's score.

`floor` defaults to 1, which makes the novelty term exactly 1.0 — the score is
still judged and still written to `novelty_scores.jsonl`, it just does not move a
reward. Measured on iter1's 27 proposals: judged against their own siblings 24 of
27 scored exactly 0.0, which is the right answer (a keyword draw produces eight
paraphrases of one task) and also a useless one, since a term constant across the
group contributes nothing after GRPO subtracts the group mean. Labelling each
task's shape on its own instead does separate proposals within a group, but the
label changed between sampled repeats on 10 of 27 statements. `NOVELTY_FLOOR=0.5`
puts it back in.

Solving side: 1.0 if the check exits 0, else 0.0.

## Files

```
challenge.py       collect: the queues, the three job bodies, the group decision
train.py           one GRPO step over what was collected, then overwrite the ckpt
sandbox.py         the sandbox as a resource: clear, snapshot, run a script
prompts.py         every string sent to a model
loop.sh            collect -> train -> collect from the new weights, until killed
episode.py         how an episode is built and scored, shared with eval.py
remote_tool_env.py the transport to one microVM, paired with sandbox_server/
sandbox_server/    the image and the in-sandbox tool server it talks to
eval.py            held-out pass rate on tasks the trainer never saw
split_tasks.py     split a collection's tasks into a train and an eval half
rsi_agent.yaml     the ms-agent config both sides' openings are shaped by
```

Output under `--out-dir`:

```
trajs/*.npz            input_ids / labels / logprobs
trajs/index.jsonl      one line per trained trajectory: side, group, reward, messages
groups.jsonl           one line per decided group, kept or not, and why
tasks.jsonl            the statements and check scripts delivered
rejected.jsonl         every build that produced no task, and how its episode ended
solver_attempts.jsonl  every attempt: the check's output and the workspace it left
novelty_scores.jsonl   the rubric, all three dimensions and all nine verdicts
keyword_gen.jsonl      every keyword call, prompt and reply verbatim
keywords.jsonl         the keyword bank, carried between iterations
challenge_metrics.json this collection as numbers: scalars, raw counters, histograms
train_summary.json     what the step actually trained, and what it skipped
```

Only `trajs/` is read again — by `train.py`. The rest is written for reading after
the fact: `solver_attempts.jsonl` is the only thing that answers whether a task at
`n_pass=0` was unsolvable or the solver gave up, and `novelty_scores.jsonl` records
usefulness and complexity, which are scored by the same call but reach no reward.

`challenge_metrics.json` is the exception: `train.py` reads its `scalars` section and
sends it to swanlab together with the training metrics, so one chart carries both
halves of an iteration. It is computed by reading `groups.jsonl` back rather than
from the live objects, so it cannot disagree with the audit file beside it, and the
same function recomputes it for a directory that finished hours ago. Three sections:

* `scalars` — fixed keys, every value a number. What goes up. Includes
  `solve_pass_rate`, the accuracy: passes over every solver attempt that ran. Read
  it as a property of the pair, not of the model — the tasks change every iteration,
  so a rise can be the solver improving or the proposer getting easier, and
  `n_pass_in_band_rate` next to it is what separates those.
* `counts` — the raw counters, dynamic keys and all. `group_dropped:rubric_error`
  exists only in a run where that happened, so these stay in the file and are not
  uploaded: a chart that appears halfway through a run reads as a change in the run.
* `distributions` — the `n_pass`, build-outcome and novelty histograms behind the
  means, because a mean `n_pass` of 4 is a different collection depending on whether
  it came from eights and zeros or from fours.

Nothing is truncated in these files. They are read to check whether a reward was
deserved, which a shortened statement cannot answer.

Everything is in this directory. `sandbox.py` takes its transport from
`remote_tool_env.py`, which is paired with the tool server in `sandbox_server/`,
and the solver's opening from `episode.solver_harness`, which `eval.py` uses too —
so a task's `n_pass` here and its `pass@k` there are measured against one opening.

## Running it

```bash
export E2B_API_KEY=...              # sandbox host
export SANDBOX_API_URL=http://...   # sandbox host address, with port
export LLM_BACKUP_API_KEY=...       # dashscope
ITERATIONS=1 bash cookbook/rsi/agentic/loop.sh
```

Charts land in swanlab project `twinkle-rsi-agentic`, one experiment named after
`TAG`, one step per iteration. `train.py` uploads after saving the checkpoint, so a
swanlab failure costs the charts and not the weights — the numbers are still in
`challenge_metrics.json` and `train_summary.json` either way. Resume is by
`id=TAG`: a second run under the same tag appends to that curve, a new tag starts a
new one. `RSI_SWANLAB_MODE=disabled` turns it off, `RSI_SWANLAB_PROJECT` moves it.

Verified on this machine at swanlab 0.9.2: three separate processes with the same
tag at steps 1, 2, 3 landed on one run (the second and third print `disabled in
resume mode`). Resume works only in `online` mode — in `local` mode each process
made its own run directory instead.

## Settings that shape what gets produced

Every one of these changes either the model's output or how it is scored. The
origin column says where the value came from; nothing marked *inherited* has been
re-measured under this scheduler.

| setting | value | origin |
|---|---|---|
| `--keep-groups` / `--group-size` / `--solver-rollouts` | 8 / 8 / 8 | decided for this pipeline |
| keep rule: ≥1 proposal with `n_pass ∈ [1,7]` | — | decided for this pipeline |
| truncated solver attempt counts as a failure, denominator fixed at 8 | — | decided for this pipeline |
| a build cut off at `--propose-max-tokens` writes no check and no statement | — | restored from the old pipeline, which skipped both stages after a length cut |
| rubric failure after 3 tries drops the whole group | — | decided for this pipeline |
| `--max-build-files` | 4 | inherited: `loop.sh` has passed this since it was added. It is text in the system prompt. |
| `--api-thinking-budget` | 4096 | inherited from the old `loop.sh` |
| `--propose-max-tokens` / `--max-turns` / `--stop-after-stuck-turns` | 8192 / 24 / 2 | inherited |
| `--one-call-per-reply` | on | inherited |
| `--check-retries` / `--check-max-tokens` | 1 / 8192 | inherited |
| `--problem-max-tokens` / `--problem-max-chars` | 4096 / 8192 | inherited |
| `--solver-max-tokens` / `--solver-max-turns` | 8192 / 24 | inherited |
| temperature / top_p, both sides | 1.0 / 0.95 | inherited |
| `--novelty-floor` | 1 | decided after measuring: the term was constant across the group at floor 0.5, so it only scaled the whole reward down |
| `--task-bank-refs` / `--novelty-tries` | 5 / 3 | inherited |
| `--keywords-n` / `--keyword-gen-calls` / `--keyword-temp` | 128 / 8 / 1.3 | inherited |
| `--snapshot-max-files` / `-per-file` / `-budget` | 50 / 600 / 6000 | inherited |
| `--sandbox-slots` | 32 | inherited: a probe once held 96, but not reliably for a whole run |
| lr / one optimizer step / `GRPOLoss(epsilon=0.2, beta=0.0)` | 1e-6 | inherited |
| `MICRO_BATCH_SIZE=1`, `padding_free=False` | — | inherited, forced by an OOM at 2 |

Prompt texts are byte-identical to the ones the old pipeline sent — verified
string by string — minus the seed and single-model follow-up strings, which this
pipeline never sends.

## What has been checked, and what has not

Checked offline, `.tmp_analysis/test_agentic.py` — the real scheduler, group
state machine, job bodies, rubric loop and writers against fake vLLM/sandbox/API.
25 checks, all passing: 8 kept groups produce exactly 64 + 64, zero-reward
proposals are still written, only the selected proposal's attempts are, a group
with nothing in band is dropped, a rubric that never returns a verdict drops the
whole group after 3 tries and skips its queued solver jobs, a length-cut build
writes no check and no statement, `solver_attempts.jsonl` has one line per
attempt (400 of them, 243 failures, each with the check's output and the workspace
it left), and `novelty_scores.jsonl` logs all three dimensions on every retry.
`train.py` then reads the same directory back as 16 groups of 8, 128
trajectories, nothing skipped, every group centred by `GRPOAdvantage`.

Two ways the run could hang were found by that test and fixed rather than worked
around: an exception inside the rubric job (whose future nobody reads) left its
group waiting for a verdict forever, and an exception while writing a decided
group's output skipped the launch of its replacement topic. Both now log and let
the run continue, and `run()` additionally stops with a message if it ever goes
quiet without reaching its target.

Not checked: anything requiring a GPU or a sandbox. No end-to-end run has been
done, so there are no wall-clock, keep-rate or `n_pass` numbers under this
scheduler, and none of the inherited settings above have been re-measured.
