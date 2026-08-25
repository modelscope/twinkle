# Agentic RSI

Self-play for tool-using agents. The model invents its own tasks by doing them,
then trains on the ones it solves only sometimes.

The solver is an **ms-agent** agent -- shell, filesystem, python, notebook,
todo -- working inside a microVM. When it stops calling tools the episode ends
and the task's check script is run against what it left behind. Checks are
ordinary programs, so the same trajectory always earns the same reward.

## Pipeline

```
challenge.py                          rl.py
┌──────────────────────────┐          ┌────────────────────────┐
│ 1 direction + keywords   │          │ 1 read flows           │
│ 2 model acts in sandbox  │          │ 2 boot sandboxes       │
│ 3 model writes checks    │  flows   │ 3 solver works N turns │
│ 4 run checks (verify)    │ ───────► │ 4 run check_script     │
│ 5 model writes statement │          │ 5 GRPO                 │
│ 6 difficulty filter      │          └────────────────────────┘
└──────────────────────────┘
```

`challenge.py` builds tasks backwards: the end state exists before the question
does, so nothing it produces can be unachievable. The difficulty filter then
drops tasks the solver always passes or always fails -- either way GRPO gets a
zero gradient from the whole group.

## Where things run

```
training host                                sandbox (microVM)
─────────────────────────────                ─────────────────────────────────
MsAgentHarness                               tool_server.py
  system prompt, message shaping               ms-agent ToolManager
  llm: and tools: popped ->                    the real file_system /
  constructs no tool at all                    code_executor / todo_list
                                        ┌──►  GET  /tools   schemas
RemoteMsAgentToolEnv  ── curl over ─────┤     POST /call    dispatch
  forwards tool calls    the sandbox    │
  copies files back      command channel└───  /workspace
```

Two properties this layout is built around:

**Nothing the model emits executes next to the trainer.** The harness has its
`llm` and `tools` sections popped *after* ms-agent merges its own `agent.yaml`
underneath -- omitting them from `rsi_agent.yaml` is not enough, since that
default declares `code_executor` and would otherwise put a live shell executor
on the training host with access to the whole machine.

**The advertised tool contract is read off the code that honours it.** The tool
schemas in the prompt come from `GET /tools` on the sandbox, not from a second
ms-agent next to the trainer. In RL the policy actively exploits whatever the
executor actually does, and any divergence from the production tools would only
surface after deployment.

## Setup

### Environment host

Needs `/dev/kvm` and kernel 6.8+. Builds the template and runs the AgentENV
server.

```bash
sh sandbox_server/install.sh                 # AgentENV + build the template
sh sandbox_server/install.sh --rebuild       # after changing the Dockerfile
sh sandbox_server/install.sh --skip-install  # template only, AgentENV already up

sh sandbox_server/serve.sh                   # foreground, binds 127.0.0.1:8000
NOHUP=1 sh sandbox_server/serve.sh           # background
```

On a restricted network the base image will not resolve from Docker Hub; point it
at a reachable mirror (forwarded to `aenv build --image`, so the Dockerfile stays
untouched):

```bash
BASE_IMAGE=docker.m.daocloud.io/library/python:3.11-slim sh sandbox_server/install.sh
```

`docker.m.daocloud.io` is a third-party Docker Hub proxy -- every sandbox's base
image comes through it. Substitute your own Aliyun accelerator address
(`<id>.mirror.aliyuncs.com`) if you would rather not depend on one.

#### When the template build is too slow to use

On 2026-08-23 three `aenv build` runs on our host failed or ran for hours, and
the cause was download speed rather than the Dockerfile. Measured within one
minute, from inside a sandbox: `deb.debian.org` 33 KB/s, `mirrors.aliyun.com`
5.4 MB/s, and for comparison the host itself 12 MB/s and sandbox disk writes
639 MB/s. apt's package index alone is 9.6MB, so the build sat two hours with no
output -- and since the server logs `template build started` and then nothing
until the build ends, slow is indistinguishable from hung. The Dockerfile now
rewrites `deb.debian.org` to the Aliyun mirror, which should remove the cause.

The path that is verified end to end installs inside a live sandbox and
snapshots it, which needs no template builder and takes about six minutes:

```bash
sh sandbox_server/build_via_sandbox.sh                     # name: twinkle-rsi-msagent
NAME=twinkle-rsi-msagent-v2 sh sandbox_server/build_via_sandbox.sh   # verify first
```

Three things to know about a snapshot:

* it shows up in `aenv snapshot list`, **not** `aenv template list`, but the name
  lives in the same namespace -- `--sandbox-template twinkle-rsi-msagent`
  resolves to it unchanged;
* it keeps the filesystem, not the image config, so the Dockerfile's `ENV
  PYTHONUNBUFFERED=1`, `ENV PIP_INDEX_URL=...` and `WORKDIR /workspace` are gone.
  `build_via_sandbox.sh` writes `/etc/pip.conf` and `/workspace` instead, and
  `remote_tool_env` starts the runtime with `python -u`;
* aenv refuses to rebind an existing name, so replacing an image means deleting
  the old one first. Build under a second name and verify before you do that --
  deleting first cost us four hours with no usable sandbox.

Verify either one from the training host with the boot check below.

The Dockerfile also pins `PIP_INDEX_URL` to an Aliyun mirror for the same
reason -- edit those two lines if your host reaches pypi.org directly.

The image clones **ms-agent from source** (`--depth 1` of `main`), not the pip
release: the tools the policy is trained against are the ones in the repository,
and a released wheel can lag behind it.

Verify a sandbox boots and the runtime comes up before anything else:

```bash
python -c "
import sys; sys.path.insert(0, '.')
from remote_tool_env import RemoteMsAgentToolEnv
e = RemoteMsAgentToolEnv(template='twinkle-rsi-msagent', config_path='rsi_agent.yaml',
                         api_url='http://127.0.0.1:8000')
e.reset()
print([t['function']['name'] for t in e.tool_schemas()])
print(e.step('shell_executor', {'command': 'python -V && pwd'}).observation)
e.close()
"
```

### Step 1 -- generate tasks

```bash
pip install e2b

python challenge.py \
    --keep-target 200 \
    --sandbox-template twinkle-rsi-msagent \
    --sandbox-api-url http://<env-host-ip>:8000 \
    --sampler-gpus 4
```

Writes `output/rsi_agentic/challenge_flows.jsonl`, one task per line:
`{id, query, check_script, n_pass, n_rollouts, keywords, seeded}`.

Also writes `output/rsi_agentic/propose_traj/` -- the rounds that *produced* each
task, kept and rejected alike, as one `.npz` of token ids / labels / logprobs per
attempt plus an `index.jsonl` carrying the text and the outcome. Nothing reads it
yet. It exists because proposing is generation like any other, so those rounds
could be trained on later; rejects are in there on purpose, since a set of
kept-only attempts has no zero-reward half to contrast against. `pass_rate` is
stored raw -- mapping it onto a difficulty score means choosing a target rate,
which is a training decision, not a dump format. Pass `--dump-propose-traj ''`
to turn it off; expect it to dwarf the task file.

Useful flags: `--seed-file` (start from existing trajectories), `--keywords-n 0`
(no keyword bank), `--solver-rollouts` (attempts per task in the difficulty
filter), `--max-turns` (tool-calling turns per episode).

Round 1 is serial -- one episode at a time, workspace cleared in between -- so
`--max-proposals-per-round` trades throughput against how often the estimator
recalibrates.

### Step 2 -- train

```bash
AENV_API_URL=http://<env-host-ip>:8000 \
AENV_TEMPLATE=twinkle-rsi-msagent \
RSI_TASKS=output/rsi_agentic/challenge_flows.jsonl \
    python rl.py --model-id ms://Qwen/Qwen3-4B \
                 --model-gpus 4 --sampler-gpus 4
```

`rl.py` accepts both task formats: `check_script` (from `challenge.py`, scored
by exit status) and structured `checks` (see `tasks.example.jsonl`).

Through a tunnel instead:

```bash
ssh -N -L 8000:127.0.0.1:8000 root@<env-host-ip>
```

## Configuration

`challenge.py` is all command-line flags (`--help`). `rl.py` reads:

| Variable | Default | |
|---|---|---|
| `AENV_API_URL` | `http://127.0.0.1:8000` | AgentENV server |
| `AENV_TEMPLATE` | `twinkle-rsi-msagent` | template or snapshot name to boot from |
| `RSI_TASKS` | `tasks.example.jsonl` | task file |
| `RSI_AGENT_CONFIG` | `rsi_agent.yaml` | uploaded into every sandbox |
| `RSI_SANDBOX_TIMEOUT` | `900` | must outlast an episode plus its checks |
| `RSI_ENV_CONCURRENCY` | `16` | parallel boot / scoring |
| `RSI_MAX_TURNS` | `20` | tool-calling turns per episode |
| `RSI_SCORE_MODE` | `fraction` | or `all_or_nothing`; structured checks only |
| `RSI_KEEP_WORKSPACES` | `0` | keep the files copied out of each sandbox |

Training hyper-parameters come from the CLI, e.g.
`python rl.py --batch-size 2 --num-generations 4 --max-steps 2`.

Sandbox count during training is `batch-size x num-generations`, each ~2GiB. At
the defaults (4 x 8) that is 32 microVMs, so size the environment host
accordingly.

## Files

| File | Role |
|---|---|
| `challenge.py` | task generation: act, write checks, verify, describe, filter |
| `prompts.py` | every string the challenger sends; categories live here too |
| `rl.py` | training loop, episode construction, scoring |
| `remote_tool_env.py` | training-side Env: forwards tool calls, copies files back |
| `rsi_agent.yaml` | ms-agent config -- read by *both* halves |
| `sandbox_server/tool_server.py` | in-sandbox HTTP server owning the ToolManager |
| `sandbox_server/Dockerfile` | template image: ms-agent, ripgrep, ffmpeg, imagemagick, openpyxl/reportlab/pdfplumber |
| `sandbox_server/install.sh` | install AgentENV + build the template |
| `sandbox_server/build_via_sandbox.sh` | build the image as a snapshot of a live sandbox instead |
| `sandbox_server/serve.sh` | start the AgentENV server |
| `tasks.example.jsonl` | hand-written tasks in the structured `checks` format |

The machinery both scripts call lives in `twinkle_agentic.challenger`; only
wiring and prompt text are here.

## Decisions worth knowing

**Round 1 is serial.** Every episode needs an empty workspace and they share one
long-lived sandbox, so proposals cannot overlap -- the second episode would see
the first one's files and its checks would pass for free. The difficulty filter
resets between attempts for the same reason.

**Checks are a python script, not a structured list.** `challenge.py` asks the
model to write asserts against the state it just produced, which is the same
trick the code challenger uses (execute first, capture the result, make that the
ground truth). Exit status is the whole verdict: no partial credit, no judge
model, no drift between rounds.

**`read_file(abbreviate=True)` is withdrawn when no LLM is configured.** That
argument asks an LLM to summarise a file. The sandbox has no API key, so the
tool server drops the unusable `llm` section *and* removes the argument from the
advertised schema -- the model is never offered something that can only fail.
Give `rsi_agent.yaml` a real `llm:` section with a key reachable from the sandbox
to get it back.

**A failed sandbox boot skips the whole training batch.** GRPO groups here are
positional: advantages are taken over consecutive runs of `num_generations`, so
dropping one episode would shift every later group onto the wrong task. There is
no retry -- a boot failure is logged and the step is abandoned.

**No web search.** ms-agent's `web_search` key only provides `fetch_page`
(retrieve a known URL). A real search tool needs `EXA_API_KEY` / `SERPAPI_API_KEY`
and is wired separately; no example task requires one.
