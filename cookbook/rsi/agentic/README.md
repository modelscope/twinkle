# agentic — the sandbox host, and the agent that runs inside it

Two things live here, and neither is a training script:

| path | what it is | who reads it |
|---|---|---|
| `sandbox_server/` | how to stand up the machine that hosts the microVMs | you, once per host |
| `rsi_agent.yaml` | the agent's own config | `ms-agent run`, inside each microVM |

The training entry point is `cookbook/rsi/rsi_grpo.py`, one level up. It is what
starts a run; this directory is what a run needs to already exist.

```bash
python cookbook/rsi/rsi_grpo.py \
    --sandbox-template twinkle-rsi-msagent \
    --sandbox-api-url http://<host>:<port> \
    --agent-config cookbook/rsi/agentic/rsi_agent.yaml \
    --agent-endpoint-host <this machine, as the sandbox can reach it>
```

Drop `--agent-config` and the solver runs through twinkle's own loop against the
environment's built-in tools instead — same tasks, same grading, no agent process.
Drop `--sandbox-template` too and the workspaces are local directories, which has
no isolation: fine for a check that is a few asserts, wrong for training a policy
to run commands it wrote itself.

## Which process runs where

```
training host                                    microVM (one per env slot)
-----------------------------------------------  ----------------------------
vLLM sampler ── PolicyEndpoint (HTTP) ◄──────────── ms-agent run
                     │                                  │ shell, python, files
                     └── LedgerBook ── trajectories      └── /workspace
```

The agent is not called turn by turn. It is started, it works, it exits, and what
trains is the requests it made on the way: the endpoint serves them from the live
sampler and reports each one, and the accounts assemble them into trajectories.
Which episode a request belongs to is decided by its API key, minted per episode.

Two consequences worth knowing before the first run:

* **`--agent-endpoint-host` is not optional with `--agent-config`.** Inside the
  microVM `127.0.0.1` is the microVM, so the default bind is an endpoint the agent
  cannot reach. It has to be an address of the training host that the sandbox
  network routes to.
* **The endpoint is in the trainer's own process, on purpose.** Point the agent at
  a model served anywhere else and every generation is off-policy by however far
  the two copies have drifted, with nothing reporting it.

## The sandbox host

Everything under `sandbox_server/` runs on the machine that hosts the microVMs,
not on the trainer. Two commands, once each:

```bash
sh install.sh                  # AgentENV + the template, from Dockerfile
sh install.sh --via=sandbox    # same template, built inside a live sandbox
sh serve.sh                    # the server, plus the reaper
```

`--via` only matters for the network: `aenv build` hands the Dockerfile to a
template builder whose VM downloaded at 33 KB/s here against a sandbox's 5.4
MB/s, so a six-minute build reads as a hung one. `--via=sandbox` installs inside
a live sandbox and snapshots it instead. Same template name either way (`TEMPLATE`,
default `twinkle-rsi-msagent`), which is what `--sandbox-template` refers to. A
snapshot carries the filesystem but not the image config, so `ENV`/`WORKDIR` from
the Dockerfile are replaced by their filesystem equivalents.

The template carries ms-agent itself (`pip install -e /opt/ms-agent` in the
Dockerfile), so `--agent-config` needs no upload and no install step at episode
time: the command the trainer sends is `ms-agent run` against a binary already
there. Editing `rsi_agent.yaml` is a trainer restart, not an image rebuild — the
file is passed in per episode.

`serve.sh` also starts the reaper, and that is not optional housekeeping:
AgentENV *persists* a sandbox when it ends — a closed sandbox is a paused one,
~1GB each — so every episode leaks a gigabyte and a full disk turns into boots
that fail with `No space left on device` and a whole batch scoring zero, which
reads like hard tasks rather than a broken host. It deletes only paused
sandboxes with the template's alias, every `REAP_INTERVAL` seconds (120), logging
to `/tmp/aenv-reap.log`. `REAP=0` turns it off, `REAP_ONLY=1` runs it alone,
`STOP_ONLY=1` stops both.

## rsi_agent.yaml

Every line in it is commented with why it says what it says; read the file rather
than a summary of it. The three that decide whether a run works at all:

* `llm.service: openai` — what makes ms-agent read `OPENAI_BASE_URL` /
  `OPENAI_API_KEY` from the environment, which is how the endpoint and the
  episode's key arrive. Both are left blank in the file on purpose.
* `tools:` — the line-up the model is offered. Declared in full, because a config
  with an `llm:` section no longer inherits ms-agent's own defaults.
* `permission:` — the refusals, relaxed as far as a config can reach. Two of them
  cannot be reached from a config at all, and the file says which and why.

## Known limitation

`rm -rf build/*` and `cp src/* dst/` are refused by ms-agent regardless of this
config: a glob in a write path is denied outright, and so is removal of a direct
child of `/`. So no task can be posed that starts from a directory needing a
clean-up. This was previously patched at runtime by an in-sandbox tool server
twinkle owned and maintained; the agent now runs as released ms-agent, so the fix
belongs upstream.
