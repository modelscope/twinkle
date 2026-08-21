# Agentic RSI

GRPO on multi-turn tool-using episodes. The solver is an **ms-agent** agent —
shell, filesystem, python, notebook, todo — working inside its own microVM. When
it stops calling tools the episode ends and the task's checks are run against
what it left behind. The checks are ordinary programs, so the same trajectory
always earns the same reward.

## Where things run

```
training host                                sandbox (one microVM per episode)
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
underneath — omitting them from `rsi_agent.yaml` is not enough, since that
default declares `code_executor` and would otherwise put a live shell executor
on the training host with access to the whole machine.

**The advertised tool contract is read off the code that honours it.** The tool
schemas in the prompt come from `GET /tools` on the sandbox, not from a second
ms-agent next to the trainer. A reimplementation of the tools would have been
much less work, but in RL the policy actively exploits whatever the executor
actually does, and any divergence from the production tools would only surface
after deployment.

## Setup

### Environment host

Needs `/dev/kvm` and kernel 6.8+. Builds the template and runs the AgentENV
server (the same server `cookbook/rl/envs` uses; only the template differs).

```bash
sh sandbox_server/install.sh              # AgentENV + build the RSI template
sh sandbox_server/install.sh --rebuild    # after changing the Dockerfile
sh sandbox_server/install.sh --skip-server  # template only, server already up

sh sandbox_server/serve.sh                # foreground, binds 127.0.0.1:8000
NOHUP=1 sh sandbox_server/serve.sh        # background
```

On a restricted network, point the base image at a reachable registry:

```bash
BASE_IMAGE=<your-registry>/library/python:3.11-slim sh sandbox_server/install.sh
```

The image installs **the `ms-agent/` checkout from this repo**, not the pip
release: the training host imports that same working copy, and a tool whose
output differs by one local commit is a train/serve mismatch the policy absorbs
silently. `install.sh` prints the staged commit — keep the host on it.

### Training host

```bash
pip install e2b

AENV_API_URL=http://<env-host-ip>:8000 \
    python rsi_agentic_grpo.py

# or through a tunnel:
ssh -N -L 8000:127.0.0.1:8000 root@<env-host-ip>
python rsi_agentic_grpo.py
```

Verify a sandbox boots and the runtime comes up before launching training:

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

## Configuration

| Variable | Default | |
|---|---|---|
| `AENV_API_URL` | `http://127.0.0.1:8000` | AgentENV server |
| `AENV_TEMPLATE` | `twinkle-rsi-msagent` | template built by `install.sh` |
| `RSI_TASKS` | `tasks.example.jsonl` | task file |
| `RSI_AGENT_CONFIG` | `rsi_agent.yaml` | uploaded into every sandbox |
| `RSI_SANDBOX_TIMEOUT` | `900` | must outlast an episode plus its checks |
| `RSI_ENV_CONCURRENCY` | `16` | parallel boot / scoring |
| `RSI_MAX_TURNS` | `20` | tool-calling turns per episode |
| `RSI_SCORE_MODE` | `fraction` | or `all_or_nothing` |
| `RSI_KEEP_WORKSPACES` | `0` | keep the files copied out of each sandbox |

Training hyper-parameters come from the CLI, e.g.
`python rsi_agentic_grpo.py --batch-size 2 --num-generations 4 --max-steps 2`.

Sandbox count is `batch-size × num-generations`, each ~2GiB. At the defaults
(4 × 8) that is 32 microVMs, so size the environment host accordingly.

## Files

| File | Role |
|---|---|
| `rsi_agentic_grpo.py` | training loop, episode construction, scoring |
| `remote_tool_env.py` | training-side Env: forwards tool calls, copies files back |
| `rsi_agent.yaml` | ms-agent config — read by *both* halves |
| `sandbox_server/tool_server.py` | in-sandbox HTTP server owning the ToolManager |
| `sandbox_server/Dockerfile` | template image: ms-agent, ripgrep, ipykernel |
| `sandbox_server/install.sh` | build the template (delegates server bootstrap) |
| `sandbox_server/serve.sh` | start AgentENV (delegates to `cookbook/rl/envs`) |
| `tasks.example.jsonl` | one task per line: `id`, `query`, `checks` |

## Decisions worth knowing

**`read_file(abbreviate=True)` is withdrawn when no LLM is configured.** That
argument asks an LLM to summarise a file. The sandbox has no API key, so the
tool server drops the unusable `llm` section *and* removes the argument from the
advertised schema — the model is never offered something that can only fail.
Give `rsi_agent.yaml` a real `llm:` section with a key reachable from the
sandbox to get it back.

**A failed sandbox boot skips the whole batch.** GRPO groups here are
positional: advantages are taken over consecutive runs of `num_generations`, so
dropping one episode would shift every later group onto the wrong task. There is
no retry — a boot failure is logged and the step is abandoned.

**No web search.** ms-agent's `web_search` key only provides `fetch_page`
(retrieve a known URL). A real search tool needs `EXA_API_KEY` / `SERPAPI_API_KEY`
and is wired separately; no example task requires one.

**Checks reach the sandbox two ways.** `file_*` checks read a local directory,
so the episode's files are copied out first (`download_workspace`, capped at 200
files / 1MiB each). `shell` and `python` checks go back into the sandbox through
`env.runner()`, where the interpreter and packages are the ones the agent used.
