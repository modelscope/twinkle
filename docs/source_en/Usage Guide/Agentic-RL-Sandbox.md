# Agentic RL with Sandbox Environments (AgentENV)

This guide shows how to use [AgentENV](https://github.com/kvcache-ai/AgentENV) to give Twinkle **real sandbox environments** for Agentic RL training. Use it when the model must execute code, install packages, run tests or modify files inside a real operating system (tool-integrated reasoning, SWE-style tasks).

Runnable example: `cookbook/rl/agentenv/`.

## When You Need AgentENV

Twinkle offers two kinds of execution environments; pick based on **whether the code is trusted**:

| | OpenEnv (`EnvPool`) | AgentENV (`AgentEnv`) |
|---|---|---|
| Executes in | The Ray worker process | A dedicated Firecracker microVM |
| Isolation | None (shares the training process) | Hardware virtualization |
| Best for | Environment logic you wrote (game rules, graders) | **Untrusted model-generated code** |
| Install packages / mutate the OS | No | Yes, and it is wiped on destroy |
| Startup cost | Function call | ~50ms (snapshot restore) + HTTP RTT |

If the model only emits an answer that your Python function grades, OpenEnv is enough and you can stop reading. **Once the model's output is code that gets executed, use AgentENV.**

## Prerequisite Check

AgentENV relies on KVM hardware virtualization, which imposes hard requirements on the host. Run this on the target machine:

```bash
uname -r                                     # need >= 6.8
ls -l /dev/kvm                               # must exist and be read/write
modinfo ublk_drv >/dev/null && echo ublk-ok  # need the ublk kernel module
```

All three must pass. Typical situations:

- **Bare metal**: usually satisfied out of the box.
- **Cloud VMs / GPU instances**: require nested virtualization, which most providers disable by default.
- **Containers / K8s Pods / managed notebooks**: depend on the **host** kernel and `/dev/kvm`, plus privileged mode and a `/dev` mount. If the host lacks KVM, nothing inside the container can work around it.

If the check fails, deploy the AgentENV server on a machine that passes and reach it over HTTP from the training side (see "Split Deployment").

## Single-Machine Topology

The simplest setup runs the training process and the AgentENV server on the same host:

```
┌────────── One machine (bare metal, GPUs + KVM) ──────────┐
│                                                          │
│  Training process (Ray/torchrun)      agentenv server    │
│  ├─ model    (GPU 0-3)   ──HTTP──>    127.0.0.1:8000     │
│  ├─ sampler  (GPU 4-7)                 └─ Firecracker    │
│  └─ driver: rollout + tool calls          sandboxes      │
└──────────────────────────────────────────────────────────┘
```

A single node does not need AgentENV's gateway/scheduler (those are for multi-node clusters); talk to the server's `:8000` directly.

**Split deployment** (GPU host lacks KVM): install AgentENV on another machine and point `AENV_API_URL` at it — no code changes.

## Step 1: Deploy the AgentENV Server

**Option A — install script (recommended, Ubuntu 24.04)**

```bash
curl -fsSL https://raw.githubusercontent.com/kvcache-ai/AgentENV/main/scripts/install.sh \
  | sudo AENV_HOME_PATH=/data/aenv bash
sudo systemctl start aenv
```

The script creates a dedicated non-root `aenv` account (granted only `CAP_NET_ADMIN`/`CAP_SYS_ADMIN` plus kvm group access), loads the ublk module, downloads runtime assets (Firecracker, kernel), and registers a systemd service. `AENV_HOME_PATH` is the data directory (default `/var/lib/aenv`); point it at a large disk since image layers and snapshots live there.

**Option B — Docker**

```bash
curl -fsSL https://raw.githubusercontent.com/kvcache-ai/AgentENV/main/scripts/docker-setup.sh | sudo bash
docker run -d --privileged -v /dev:/dev -p 8000:8000 ghcr.io/kvcache-ai/aenv-server:latest
```

Docker here is only a **deployment vehicle**; sandboxes are still Firecracker microVMs, so host KVM is still required.

**Verify**

```bash
curl http://127.0.0.1:8000/health     # expect 204
```

## Step 2: Build an Environment Template

A template is the sandbox's factory image: dependencies are pre-installed and frozen into a snapshot, which is what makes ~50ms boots possible. **Bake dependencies into the template** — installing them per episode at training time costs tens of seconds per trajectory and is not acceptable.

Install the CLI and authenticate (the install script already ships `aenv`; use `install-cli.sh` on a remote machine):

```bash
aenv auth
# AENV server URL [http://localhost:8000]: http://127.0.0.1:8000
# API key: dummy          # any non-empty string works for a local deployment
```

Write a Dockerfile (see `cookbook/rl/agentenv/Dockerfile`):

```dockerfile
FROM python:3.11-slim
# Configure a mirror if your network requires one
# ENV PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir numpy sympy
WORKDIR /workspace
```

Build it:

```bash
aenv build cookbook/rl/agentenv/Dockerfile -t twinkle-math --cpu-count 1 --memory-mb 1024
aenv template watch <template-id>      # wait until ready
aenv template list                     # confirm the template exists
```

`aenv build` understands `FROM / RUN / ENV / WORKDIR / USER` (`ENTRYPOINT` becomes the start command; `EXPOSE / VOLUME / LABEL` are ignored). Rebuild only when the Dockerfile changes.

You can also use an image as-is: `aenv pull ubuntu:22.04 --name ubuntu`.

## Step 3: Wire Up the Training Side and Smoke-Test

```bash
pip install e2b        # AgentENV exposes an E2B-compatible API; reuse the official SDK
```

Validate the path with five lines before launching training:

```python
from twinkle_agentic.envs import AgentEnv

env = AgentEnv(template='twinkle-math', api_url='http://127.0.0.1:8000')
print(env.reset().observation)                                  # boots a sandbox
print(env.step('run_command', {'command': 'python -c "print(6*7)"'}).observation)
print(env.step('read_file', {'path': '/etc/os-release'}).observation[:80])
env.close()
```

`AgentEnv` is a stateless HTTP client implementing the standard `Env` interface (`reset`/`step`/`tools`/`close`). It deliberately does **not** use `@remote_class`: sandbox placement, load balancing, pause/resume and failover are all handled server-side by AgentENV, and Ray plays no part in environment scheduling.

Common parameters:

| Parameter | Description |
|---|---|
| `template` | Template name, i.e. the value of `aenv build -t` |
| `api_url` | Server or gateway URL; the `E2B_API_URL` env var also works |
| `sandbox_timeout` | Sandbox idle timeout (s). On expiry AgentENV pauses (not kills) the sandbox and auto-resumes on next access. **Must exceed the longest single trajectory** |
| `command_timeout` | Per-command timeout (s) |
| `setup_commands` | Commands run once after every reset |
| `include_default_tools` | Expose the built-in `run_command`/`write_file`/`read_file`; default `True` |

## Step 4: Run Training

```bash
cd cookbook/rl/agentenv
AENV_API_URL=http://127.0.0.1:8000 AENV_TEMPLATE=twinkle-math sh agentenv_grpo.sh
```

The example is multi-turn GRPO on GSM8K: the model writes Python in the sandbox to compute the result, then calls `submit`; the reward comes from answer correctness. Overridable environment variables:

| Variable | Default | Description |
|---|---|---|
| `AENV_API_URL` | `http://127.0.0.1:8000` | AgentENV endpoint |
| `AENV_TEMPLATE` | `twinkle-math` | Template name |
| `SANDBOX_TIMEOUT` | `600` | Sandbox idle timeout (s) |
| `ENV_CONCURRENCY` | `16` | Driver threads used to create/kill sandboxes concurrently |
| `MAX_TURNS` | `6` | Max tool-calling turns per episode |

Training hyperparameters (`--model-gpus`/`--batch-size`/`--num-generations`, ...) are CLI flags, consistent with the other cookbook examples.

## Customizing Your Task

Three places to change; the AgentENV server needs no modification.

### 1. Tools (`tools.py`)

The AgentENV server does **not** define "tools" — it provides capability primitives (arbitrary command execution, filesystem access, port proxying). "Tools" are a client-side concept, registered on `AgentEnv`:

```python
# Command-template tool (most common)
env.register_command_tool(
    {'type': 'function', 'function': {
        'name': 'run_tests',
        'description': 'Run the task test suite.',
        'parameters': {'type': 'object',
                       'properties': {'test_file': {'type': 'string'}},
                       'required': ['test_file']},
    }},
    'cd /workspace && pytest {test_file} -x -q')     # formatted with tool arguments

# Arbitrary Python handler
def submit(env, arguments):
    env.submitted_answer = str(arguments.get('answer', '')).strip()
    return f'Answer submitted: {env.submitted_answer}'

env.register_tool({'type': 'function', 'function': {'name': 'submit', ...}}, submit)
```

A handler has signature `handler(env, arguments) -> str` (the return value becomes the observation) and can use `env.run_command(...)` and `env.sandbox` (the raw E2B handle, for PTY, file watching and other capabilities). Registering an existing name overrides the built-in tool; `include_default_tools=False` disables the built-ins entirely.

For complex tools, **bake the implementation into the template** and keep the handler a one-liner:

```dockerfile
COPY tools/search.py /opt/tools/search.py   # installed at template build time
```
```python
env.register_command_tool({...}, 'python /opt/tools/search.py {query}')
```

This gives tool implementations versioning, snapshot distribution and zero runtime cost.

### 2. Rewards

Sandboxes do not produce rewards (`AgentEnv.evaluate` returns zeros). Score after the rollout in the training loop, as `extract_rewards` does in the example: read the state your tool handler stashed on `env` (e.g. `env.submitted_answer`) and compare it with the ground truth.

To grade inside the sandbox (e.g. running a test suite), have the tool emit structured output and parse it on the driver side:

```python
out = env.run_command({'command': 'pytest -q --json-report --json-report-file=/tmp/r.json; cat /tmp/r.json'})
```

### 3. Termination

`MultiTurnRollout` terminates when **the model stops emitting tool calls** (or hits `max_turns` / the length cap); it does not read `EnvTool.done`. So the system prompt must explicitly ask the model not to call more tools after submitting, otherwise every episode runs to `max_turns`.

## Capacity Planning

Sandboxes consume **CPU and memory** only, and Firecracker does **not** support GPU passthrough — you cannot run GPU workloads inside a sandbox.

Single-machine estimate:

```
concurrent sandboxes = batch_size × num_generations
memory needed       ≈ concurrent sandboxes × the template's --memory-mb
```

The example defaults to `batch_size=4 × num_generations=8 = 32` concurrent sandboxes at 1GB each, i.e. ~32GB of host memory headroom (on top of GPU memory). If memory is tight, adjust in this order:

1. Lower the template's `--memory-mb` (512MB suffices for many tasks)
2. Reduce `batch_size`
3. Rely on AgentENV's auto-pause: idle sandboxes return memory to the host

Also watch CPU contention: sandboxes compete with dataloader/tokenizer work, so keep `ENV_CONCURRENCY` at or below the number of spare cores.

## Security

For single-machine personal use (server bound to `127.0.0.1`) the risk is low. Once **other people** can reach it:

1. **AgentENV's control plane has no authorization**, and upstream explicitly states it must not be exposed to the public network. Any non-empty API key is accepted and there is no tenant isolation — anyone holding a sandbox id can read, write or destroy that sandbox, and `GET /sandboxes` lists everyone's sandboxes.
2. **Build your own auth/tenant/quota layer** in front of it, keep AgentENV on a private network, and expose only your own API (ideally task-level, so sandboxes never appear in the external contract).
3. **Restrict sandbox egress by default** so model-generated code cannot scan your intranet (SSRF) or abuse bandwidth. Pass a network policy at creation time:

```python
# base_policy: Default | Allow | Deny
{'base_policy': 'Deny',
 'egress': {'allowed_domains': ['pypi.org'],
            'denied_cidrs': ['10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16', '169.254.0.0/16']}}
```

`169.254.0.0/16` must be denied (cloud metadata service). Each sandbox already gets its own network namespace and iptables isolation.

4. Always set `sandbox_timeout` so failed trajectories cannot leave zombie sandboxes holding resources.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Server fails with `/dev/kvm is not accessible` | The runtime account is not in the kvm group, or the host has no KVM. Run `sudo server --setup-host --runtime-user aenv --runtime-group aenv` and restart the service |
| `ublk_drv is not loaded` | `sudo modprobe ublk_drv`; kernels older than 6.8 must be upgraded |
| `ImportError: AgentEnv requires the E2B SDK` | `pip install e2b` |
| Sandbox creation reports a missing template | Check the name with `aenv template list`; confirm the build is ready (`aenv template watch`) |
| `pip install` fails inside the sandbox | Egress blocked by the network policy, or no mirror configured; prefer pre-installing in the template |
| Sandbox becomes unavailable mid-trajectory | `sandbox_timeout` is shorter than the trajectory duration, so it was paused. Increase it |
| Batches are slow to start | Increase `ENV_CONCURRENCY`; make sure dependencies are baked into the template rather than installed at runtime |
| Training skips batches for too few valid trajectories | Most trajectories were filtered for length. Reduce `MAX_TURNS`/`max_tokens`, or trim tool output (observations are already truncated at 32K chars) |

## Related Documents

- End-to-end walkthrough and backend selection: [Agentic RL Best Practices](./Agentic-RL-Best-Practices.md) (includes the remote OpenEnv code task)
- Deployment selection and cross-network access: [Agentic RL Deployment Guide](./Agentic-RL-Deployment.md) (backend matrix, four deployment tiers, SSH port forwarding)
- Component reference: `Components/Agentic/Envs.md` (the `Env` abstraction, `EnvTool`, both OpenEnv modes)
- Multi-turn tool usage: `Components/Agentic/Multi-Turn-Tool-Usage.md`
- Runnable examples: `cookbook/rl/agentenv/` (AgentENV), `cookbook/rl/openenv_code/` (remote OpenEnv counterpart)
- AgentENV upstream docs: <https://kvcache-ai.github.io/AgentENV/>
