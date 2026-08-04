# Agentic RL Deployment and Training

Agentic RL has two parts: the execution environment where the model performs actions, and the GRPO training loop that consumes the resulting trajectories. This guide covers the deployment of each and how they are wired together.

The two are orthogonal: switching execution backends requires only a change to the `CODE_RL_BACKEND` environment variable, and moving to a separate host requires only a change to one URL — the training code stays untouched. Validate the full pipeline on a single machine first, then scale out.

The examples throughout use [`cookbook/rl/envs/`](https://github.com/modelscope/twinkle/tree/main/cookbook/rl/envs): a multi-turn code generation task on the MBPP dataset, where one `train.py` supports both backends. The task, tools, and reward formula are identical; only the location of code execution differs, so differences between experiment curves can be attributed to the execution environment itself.

## Choosing an Execution Backend

The three options differ mainly in isolation level and its memory cost.

| | OpenEnv embedded | OpenEnv server | AgentENV |
|---|---|---|---|
| Where the environment runs | Inside the training process | A standalone HTTP/WebSocket service | A dedicated Firecracker microVM |
| Isolation | None | Process / container | microVM (KVM) |
| Executor | Depends on the environment package | smolagents AST interpreter | Real CPython |
| Memory per environment | KBs | KBs | **~1GB** |
| Files / pip / subprocesses | Depends on the environment package | Not supported | Supported, wiped on teardown |
| Deployment cost | Zero | One `uvicorn` command | Requires a control plane + `/dev/kvm` |

Selection criteria:

**Pure-computation environments** (board games, text games, scoring logic that can be safely evaluated inside the training process) — use embedded OpenEnv, with zero deployment.

**Code execution is required but the standard library suffices** — use OpenEnv server mode. It is three orders of magnitude lighter than a microVM; booting a virtual machine to execute a few lines of `sorted()` is not worth the cost.

**`unittest` + `@patch` is required, or the model needs to write files, install packages, or spawn subprocesses** — AgentENV is the only option. There is an easily overlooked failure mode here: OpenEnv's `coding_env` is built on smolagents' `LocalPythonExecutor`, **an AST interpreter rather than an OS-level sandbox**. It does not handle `decorator_list` at all, so **decorators are silently ignored** — `@patch` has no effect, the test does not fail, and the reward yields a plausible-looking but incorrect number. Silent errors of this kind are harder to locate than crashes. It is suitable for enforcing an import allowlist, not for executing adversarial code.

For a full comparison of backend capabilities, see [Execution Environments](../Components/Agentic/Envs.md).

---

# Part One: Deploying the Execution Environment

## OpenEnv

### Embedded: no deployment required

The environment is instantiated directly inside the training process, with no network hop:

```bash
cd cookbook/rl/multi_turn && python multi_turn_grpo.py
```

When there are many environment instances and CPU becomes the bottleneck, use `EnvPool` to move them onto a dedicated CPU `DeviceGroup`, keeping them off the GPU process's memory and GIL:

```bash
ENV_REMOTE=1 ENV_NUM_WORKERS=8 ENV_POOL_SIZE=64 python multi_turn_grpo.py
```

Sharding only actually happens with `ENV_REMOTE=1`; without it, environments run locally in the driver (zero RPC). `ENV_POOL_SIZE=0` means the number of trajectories is used automatically.

`EnvPool` is only meaningful for embedded OpenEnv. **Do not place `OpenEnvClient` or `AgentEnv` inside an `EnvPool`** — their session / sandbox lifecycle lives on the server side, so having Ray shard them again yields no benefit and merely adds an RPC hop.

### Server mode

The environment host needs no GPU, KVM, or Docker:

```bash
cd cookbook/rl/envs
sh openenv_server/install.sh    # pip install openenv + coding_env from source
sh openenv_server/serve.sh      # 4 workers x 64 sessions = 256 concurrent
```

`coding_env` is a subpackage inside the OpenEnv repository that is not published to PyPI, so it can only be installed from source — `server_app.py` imports `PythonCodeActEnv` and `PyExecutor` from it, and smolagents is pulled in through it as well.

The training host only needs `pip install openenv`, which provides the client classes; `coding_env` is not required there.

### Why not use the upstream server

`serve.sh` starts [the `server_app.py` in this directory](https://github.com/modelscope/twinkle/blob/main/cookbook/rl/envs/openenv_server/server_app.py) rather than the upstream `coding_env.server.app`. Adopting upstream directly runs into three problems:

```python
class ConcurrentCodeEnv(PythonCodeActEnv):
    # Upstream defaults to False; create_app(max_concurrent_envs > 1) then raises
    # ConcurrencyConfigurationError and the server is locked to a single session
    SUPPORTS_CONCURRENT_SESSIONS = True

    def reset(self, **kwargs):
        # The parent reset() rebuilds executor and transform from upstream
        # defaults, so they must be reconfigured afterwards
        observation = super().reset()
        self._configure()
        return observation

    def _configure(self) -> None:
        # Upstream only authorizes "import json"; math / collections, both common
        # in MBPP, would fail outright
        self._executor = PyExecutor(additional_imports=list(ALLOWED_IMPORTS))
        # Upstream's create_safe_coding_transform() overwrites observation.reward
        # with code-style heuristics (-1.0 on seeing open( / import os, +0.1 for
        # short code). This task's reward comes from unit tests; style scores
        # sharing the same channel would only be noise
        self.transform = None
```

Enabling `SUPPORTS_CONCURRENT_SESSIONS` is safe: `create_app` receives the **class** (used as a factory), each WebSocket connection creates a new instance, and both executor and state are instance-private.

Note also that the HTTP `/step` and `/reset` endpoints are unsuitable for multi-turn scenarios: OpenEnv creates a new env per request on these two endpoints and calls `close()` immediately after returning, so no state is retained. Multi-turn episodes must use WebSocket, which `OpenEnvClient` already implements.

### Capacity

```
concurrent session limit = WORKERS x MAX_CONCURRENT_ENVS   # default 4 x 64 = 256
must be >= BATCH_SIZE x NUM_GENERATIONS                    # default 4 x 8 = 32
```

Connections beyond capacity are rejected by the server, which shows up as some trajectories in a rollout batch having observations that are entirely `Error:`. When raising `--batch-size` / `--num-generations`, scale `WORKERS` or `MAX_CONCURRENT_ENVS` accordingly.

## AgentENV

[AgentENV](https://github.com/kvcache-ai/AgentENV) uses Firecracker microVMs to provide real operating-system semantics, which suits tool-integrated reasoning and SWE-style tasks.

### Host prerequisite check

All three of the following must hold:

```bash
uname -r                                     # >= 6.8
ls -l /dev/kvm                               # must exist and be read-writable
modinfo ublk_drv >/dev/null && echo ublk-ok  # the ublk kernel module is required
```

Bare metal generally satisfies these directly. Cloud VMs and GPU instances require nested virtualization to be enabled by the provider, which is off by default in most cases. Managed environments such as containers, K8s Pods, and notebook services depend on the **host's** kernel and `/dev/kvm`, and additionally need privileged mode with `/dev` mounted — when the host does not qualify, no software workaround inside the container can get around it.

When the prerequisites are not met, no code changes are needed: deploy AgentENV separately on a machine that does qualify, and point `AENV_API_URL` at that address from the training side. Firecracker **does not support GPU passthrough**, so GPU workloads cannot run inside the sandbox and the environment host does not need a graphics card.

### Installing the server

```bash
cd cookbook/rl/envs
sh agentenv_server/install.sh
```

The script performs four steps in order: install the server and the `aenv` CLI, provision the host via `server --setup-host` (kvm group, ublk module, udev rules, sysctl), prepare the runtime directories, and build the sandbox template.

Underneath it calls AgentENV's official `install.sh`, which creates a dedicated non-root `aenv` account (granted only `CAP_NET_ADMIN`/`CAP_SYS_ADMIN` and the kvm group), downloads runtime assets such as Firecracker and the guest kernel, and registers a systemd service. The data directory defaults to `/var/lib/aenv` and holds both image layers and snapshots, so pointing it at a large disk is recommended:

```bash
curl -fsSL https://raw.githubusercontent.com/kvcache-ai/AgentENV/main/scripts/install.sh \
  | sudo AENV_HOME_PATH=/data/aenv bash
```

When only the client is needed (creating templates from the training host, or deploying via Docker):

```bash
curl -fsSL https://raw.githubusercontent.com/kvcache-ai/AgentENV/main/scripts/install-cli.sh | sudo bash
```

### Building a sandbox template

A template is the equivalent of a sandbox's factory image: dependencies are pre-installed and frozen into a snapshot, which is what allows a sandbox to start in roughly 50ms. **Dependencies must be baked into the template** rather than installed with `pip install` at training time — repeating a several-tens-of-seconds install on every trajectory is unacceptable overhead.

`install.sh` already builds the template. To rebuild it manually:

```bash
sh agentenv_server/install.sh --rebuild    # deletes the old template first
```

Or use the CLI directly:

```bash
aenv auth
# AENV server URL [http://localhost:8000]: http://127.0.0.1:8000
# API key: dummy          # any non-empty string works for a local deployment

aenv build agentenv_server/Dockerfile -t twinkle-code --cpu-count 1 --memory-mb 1024
aenv template watch <template-id>      # wait for ready
```

`aenv build` supports `FROM / RUN / ENV / WORKDIR / USER` (`ENTRYPOINT` becomes the start command; `EXPOSE / VOLUME / LABEL` are ignored). Both the image pull and the overlaybd conversion happen **on the server**, so the local machine does not need docker. A rebuild is only necessary when the Dockerfile changes. To use an existing image without modification: `aenv pull ubuntu:22.04 --name ubuntu`.

Aliases cannot be rebound. On `alias 'xxx' already points to ...`, run `aenv template delete xxx` first — `--rebuild` wraps exactly this step.

Implementations of complex tools are also best baked into the template, leaving the tool handler as a single call:

```dockerfile
COPY tools/search.py /opt/tools/search.py
```

The tool implementation is then version-controlled, distributed with the snapshot, and free at runtime.

### Starting the service

```bash
sh agentenv_server/serve.sh              # foreground, binds 127.0.0.1:8000
NOHUP=1 sh agentenv_server/serve.sh      # background, logs to /tmp/aenv-server.log
API_ADDR=0.0.0.0:8000 sh agentenv_server/serve.sh   # external; read the security note below first
```

The script stops any running instance before starting, so restarting requires no manual kill.

The server's own default listener is `0.0.0.0:8000`; the script narrows this to loopback. **AgentENV provides no authentication whatsoever**, so a reachable port is equivalent to reachable arbitrary code execution — binding `0.0.0.0` requires a security-group allowlist.

Verify:

```bash
curl -i http://127.0.0.1:8000/health     # expect 204
```

Single-machine setups do not need AgentENV's gateway / scheduler (both target multi-node deployments); connect directly to the server's `:8000`.

### Memory budget

Memory is AgentENV's primary capacity constraint:

```
concurrent sandboxes = BATCH_SIZE x NUM_GENERATIONS
memory required      = concurrent sandboxes x template --memory-mb + 8GB (AgentENV itself + system)
```

At `--memory-mb 1024`, the default 32 concurrent sandboxes need 40GB; during pipeline validation, `2 x 4 = 8` concurrent sandboxes need only 16GB, so there is no need to procure a large-memory machine up front.

When memory is short, adjust in this order: lower the template `--memory-mb` (512MB suffices for most tasks) → reduce `batch_size` → rely on automatic hibernation (idle sandboxes return memory to the host once paused).

CPU contention also needs consideration: sandboxes compete for CPU cores with the dataloader and tokenizer, so `ENV_CONCURRENCY` should not exceed the number of idle cores.

## Networking and Security

twinkle supports two connection methods: **direct HTTP** for the same machine or the same private network / VPC, and **SSH port forwarding** across networks.

The framework does not introduce VPN or NAT-traversal components — those belong to the network infrastructure layer and are unrelated to the training code. All that is visible from twinkle's side is an `http://host:port`.

### Direct HTTP

In enterprise setups, GPU and CPU machines are usually already in the same VPC / IDC / K8s cluster, so no extra network components are needed. `HOST` in `openenv_server/serve.sh` defaults to `0.0.0.0` (in which case the script prints a no-authentication warning), so bind the **private NIC** explicitly:

```bash
HOST=10.0.1.20 sh openenv_server/serve.sh              # OpenEnv
API_ADDR=10.0.1.20:8000 sh agentenv_server/serve.sh    # AgentENV
```

The training side only needs one environment variable changed:

```bash
OPENENV_BASE_URL=http://10.0.1.20:8000 sh run_openenv.sh
AENV_API_URL=http://10.0.1.20:8000     sh run_agentenv.sh
```

Security-group ingress should admit **only the training host's IP/32 or its security-group ID**, open only port 8000, and never use `0.0.0.0/0`. Confirm the binding took effect with `ss -tlnp | grep 8000` — the output must be `10.0.1.20:8000`, not `0.0.0.0:8000`.

The environment service needs a keep-alive mechanism: once it goes down, the entire rollout batch is wasted. This belongs to operations infrastructure, so use whatever process manager you already have. A minimal systemd unit:

```ini
# /etc/systemd/system/openenv-server.service
[Service]
User=openenv
WorkingDirectory=/opt/twinkle/cookbook/rl/envs/openenv_server
Environment=HOST=10.0.1.20 MAX_CONCURRENT_ENVS=64
ExecStart=/bin/sh serve.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

`Restart=always` is the essential line — it covers both a process crash and an OOM kill. Enable it with `sudo systemctl enable --now openenv-server`.

### SSH port forwarding

Across networks, SSH here does more than establish connectivity — it **adds a layer of authentication to a service that has none**, reusing existing SSH keys and login auditing, and it is typically an operations channel already approved within the organization.

Port forwarding operates at the TCP layer and is fully transparent to WebSocket. Verified in practice: the handshake, session state across messages, and 8 concurrent sessions multiplexed over one connection all work correctly.

```bash
# Environment side: loopback only, unreachable from the network
HOST=127.0.0.1 sh openenv_server/serve.sh

# Training side
ssh -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
    -L 8000:127.0.0.1:8000 user@env-host &
OPENENV_BASE_URL=http://127.0.0.1:8000 sh run_openenv.sh
```

`ssh -N` not returning is normal — that blocked state is the tunnel working; add `-f` to move it to the background.

Two addresses are easily confused: `OPENENV_BASE_URL` takes the **local entry point** `127.0.0.1:8000`, while the `127.0.0.1:8000` in `-L` is the target address as resolved **from the environment host's perspective**. The same applies to AgentENV via `AENV_API_URL`.

Long-running jobs must have keep-alive configured: if the forward drops, the entire rollout batch is wasted, so it should not run in an interactive shell:

```bash
autossh -M 0 -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -L 8000:127.0.0.1:8000 user@env-host
```

All traffic is multiplexed over one TCP connection. Each turn transfers only a few KB of code and output; 8 concurrent sessions showed no pressure in practice. At hundreds of concurrent sessions, multiple forwards mapped to different local ports can spread the load.

### Sandbox egress

Restricting what code inside the sandbox can reach prevents model-generated code from scanning the internal network (SSRF) or abusing bandwidth. AgentENV provides a node-level enforced policy in `config/default.toml`, and the defaults are already reasonable:

```toml
[network.egress]
# A sandbox's own egress policy cannot override these
always_denied_cidrs = [
  "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8",
  "169.254.0.0/16",              # <- critical: blocks the cloud metadata service 169.254.169.254
  "172.16.0.0/12", "192.168.0.0/16",
]
```

The `169.254.0.0/16` entry is the most critical one: it blocks cloud provider metadata services, preventing code inside the sandbox from stealing temporary IAM credentials and thereby gaining permissions over the entire cloud account. Standard-library algorithm problems need no outbound access, so the defaults can be used as-is.

To relax this, pass a policy per sandbox:

```python
# base_policy: Default | Allow | Deny
{'base_policy': 'Deny',
 'egress': {'allowed_domains': ['pypi.org', 'files.pythonhosted.org'],
            'denied_cidrs': ['10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16', '169.254.0.0/16']}}
```

One final point deserves emphasis: the API key used by `aenv auth` **provides no authentication**. Any non-empty string passes in a local deployment, and there is no tenant isolation — obtaining any sandbox-id grants read, write, and destroy access to that sandbox, and `GET /sandboxes` lists every sandbox. To expose this as a service, you must build your own authentication / tenancy / quota layer, keep AgentENV confined to a private network, and expose only your own upper-layer API.

---

# Part Two: Wiring Up Training

This part continues with `cookbook/rl/envs/`. Code generation was chosen as the task for two reasons: its reward can be computed objectively from unit tests, with no judge model required; and multi-turn interaction has intrinsic meaning — write, try, fix, submit.

## The action space the model sees

Only two tools are exposed to the model (defined in `_openenv.py` / `_agentenv.py`): `run_python(code)` executes code in the environment, and `submit_solution(code)` submits the final answer.

The second tool deserves separate mention, as it **is not sent to the server**:

```python
def _submit_solution(env, arguments: Dict[str, Any]) -> str:
    code = (arguments.get('code') or '').strip()
    if not code:
        return "Error: 'code' argument is required."
    env.submitted_code = code
    return 'Solution submitted.'
```

This tool is handled locally on the client via `register_tool`, recording the source on the env purely for the training loop to score. This is a general pattern: **the model's actions and the bookkeeping training needs are two separate concerns**, and the latter belongs in a local handler rather than polluting the environment protocol.

The system prompt must describe backend semantics accurately, otherwise the model writes code against the wrong mental model. The two backends are exactly **opposite** on this point:

- OpenEnv session: `The interpreter keeps its state between calls` — a function defined this turn can be called directly in the next. It also lists the module allowlist and declares that there is no file or network access.
- AgentENV: `Each call runs in a FRESH process, so every snippet must be self-contained`.

The AgentENV side additionally sets `include_default_tools=False` to disable the built-in `run_command` / `read_file` / `write_file`, aligning the action space precisely with the task to keep reward attribution clean.

The tool set is a client-side concept. The AgentENV server itself **defines no tools**; it only provides capability primitives: arbitrary command execution, file read/write, and port proxying. Besides `register_tool(schema, handler)`, there is also the command-template form `register_command_tool`:

```python
env.register_command_tool(
    {'type': 'function', 'function': {
        'name': 'run_tests',
        'description': 'Run the task test suite.',
        'parameters': {'type': 'object',
                       'properties': {'test_file': {'type': 'string'}},
                       'required': ['test_file']}}},
    'cd /workspace && pytest {test_file} -x -q')     # formatted with the tool arguments
```

The handler signature is `handler(env, arguments) -> str`, and the return value becomes the observation. Internally it can use `env.run_command(...)` and `env.sandbox` (the raw E2B handle, giving access to PTY, file watching, and similar). Registering under an existing name overrides the built-in tool.

## One env per trajectory

```python
def prepare_trajectories(samples, pool):
    envs = [backend.make_env() for _ in samples]
    # reset() blocks on the network (a WebSocket handshake, or booting a sandbox),
    # so it must be concurrent — otherwise a batch of 32 waits serially
    list(pool.map(lambda env: env.reset(), envs))

    trajectories, tool_managers = [], []
    for sample, env in zip(samples, envs):
        tool_managers.append(ToolManager(EnvTool.from_env(env)))
        trajectories.append({
            'messages': [
                {'role': 'system', 'content': backend.SYSTEM_PROMPT},
                {'role': 'user', 'content': sample['prompt']},
            ],
            'tools': backend.TOOL_SCHEMA,
        })
    return trajectories, tool_managers, envs
```

**One `ToolManager` per trajectory.** It holds a specific env instance, so sharing one would send every trajectory's tool calls to the same session.

**`envs` must be closed in a `finally` block.** They occupy server-side capacity, and leaking them causes subsequent steps to fail for lack of capacity:

```python
try:
    all_trajectories = rollout(expand_prompts, tool_manager=tool_managers)
    total_rewards, pass_rates = extract_rewards(envs, expanded, env_pool)
finally:
    close_envs(envs, env_pool)
```

One more easily overlooked behavior: `MultiTurnRollout` terminates when **the model stops emitting tool calls** (or when `MAX_TURNS` / the length limit is hit); it **does not read** `EnvTool.done`. The system prompt therefore needs an explicit instruction not to call tools after submitting, otherwise episodes run all the way to `MAX_TURNS`.

## Reward: unit tests rather than a judge model

Each MBPP sample ships with a number of `assert` statements. Hidden tests are replayed after the rollout finishes; **the script form is dictated by backend capability, but the reward formula is identical** for both backends.

The AgentENV side is real CPython, so it generates an ordinary script directly, wrapping each assertion in `try` and printing `TESTS_PASSED n total` at the end.

The OpenEnv side executes them one at a time **within the same session**, rewriting `assert X` into `print(X)`. The purpose is diagnosability: the exception raised by a failing `assert` is indistinguishable from a crash inside the solution, whereas printing separates "the assertion evaluated to False" from "the code crashed", and one test raising does not stop the rest from running. The executor itself does support `assert` and `try`; this is not a capability workaround.

The OpenEnv side uses `env.execute()` rather than `env.step()`: `execute()` returns the server's **raw** `StepResult`, exposing structured fields such as `exit_code`, while `step()` returns text rendered for the model and counts toward the episode. Scoring logic should not appear in the model's conversation.

Reward shaping:

```python
rate = passed / total if total else 0.0
if total and passed == total:
    return 1.0, rate                    # all passed
if getattr(env, 'submitted_code', None):
    return 0.1 + 0.4 * rate, rate       # submitted, partially correct -> 0.1 ~ 0.5
return 0.0, rate                        # not submitted
```

The shape guarantees "all correct > partially correct > submitted but all wrong > not submitted", and the 1.0 for all-correct is clearly above the 0.5 ceiling for partial credit — otherwise the model learns to submit a fake implementation that just barely passes the first test.

**Giving the submit action a 0.1 floor** is deliberate: early in training all trajectory rewards are 0, GRPO's within-group advantages are then all 0, and no useful learning signal can be produced. That floor supplies the initial gradient signal.

The sandbox itself produces no reward (`AgentEnv.evaluate` returns 0 by default); scoring happens uniformly in the training loop. To score inside the sandbox, have the tool emit a structured result and parse it on the driver side.

## GRPO group layout

```python
batch = [dataset[(sample_cursor + i) % len(dataset)] for i in range(BATCH_SIZE)]
expanded = [s for s in batch for _ in range(NUM_GENERATIONS)]   # N copies of the same problem, contiguous
...
advantages = advantage_fn(total_rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()
```

Note how `expanded` is written: `for s in batch for _ in range(N)` places the N rollouts of one problem **contiguously** in the list, and `GRPOAdvantage` splits groups according to that layout. Writing `for _ in range(N) for s in batch` misaligns the grouping completely — training appears to proceed normally while learning nothing useful.

## Launching training

Validate the full pipeline at low concurrency first, to save both time and compute cost:

```bash
cd cookbook/rl/envs

# OpenEnv
sh run_openenv.sh --batch-size 2 --num-generations 4 --max-steps 2

# AgentENV
sh run_agentenv.sh --batch-size 2 --num-generations 4 --max-steps 2
```

Concurrency must not be too low, however: the trajectory count (`batch-size x num-generations`) must be >= `--model-gpus` (4 by default). Otherwise too few remain after length filtering and the whole batch is skipped, with the log merely repeating `skipping this batch` — a symptom that looks like a hang rather than an error, and is harder to diagnose than an outright crash. The `2 x 4 = 8` above leaves a 2x margin.

Arguments only take effect when placed **after** the script name (the script runs `python train.py $TRAIN_ARGS "$@"`, and `"$@"` must come last to override).

Scaling up to real training:

```bash
OPENENV_BASE_URL=http://10.0.0.5:8000 MAX_TURNS=8 ENV_CONCURRENCY=32 \
sh run_openenv.sh --batch-size 8 --num-generations 16 --max-steps 500
```

Overridable environment variables:

| Variable | Default | Description |
|---|---|---|
| `MAX_TURNS` | `6` | Maximum tool-calling turns per episode |
| `ENV_CONCURRENCY` | `16` | Driver threads for concurrent create / destroy / scoring |
| `OPENENV_BASE_URL` | `http://127.0.0.1:8000` | OpenEnv service address (a load balancer works too) |
| `OPENENV_ENV_NAME` | `coding_env` | Environment package name; determines the client and Action classes |
| `OPENENV_MESSAGE_TIMEOUT_S` | `120` | Per-message timeout |
| `AENV_API_URL` | `http://127.0.0.1:8000` | AgentENV address |
| `AENV_TEMPLATE` | `twinkle-code` | Template name |
| `SANDBOX_TIMEOUT` | `600` | Sandbox idle timeout (seconds) |
| `AENV_COMMAND_TIMEOUT` | `60` | Per-command timeout inside the sandbox (seconds) |

Training hyper-parameters are passed as CLI arguments, with defaults defined in the respective `run_*.sh`. The `TRAIN_ARGS` of both backends must stay in sync, otherwise changes in reward cannot be attributed to the execution backend rather than to differing hyper-parameters.

`AENV_API_URL` cannot be replaced by `E2B_API_URL` — setting only the latter silently falls back to the default.

## Monitoring metrics

The log prints one line per step:

```
[Step 0] {'train/code_acc': 0.031, 'train/test_pass_rate': 0.208, 'train/avg_reward': 0.145, ...}
```

Task metrics:

| Metric | Meaning and interpretation |
|---|---|
| `train/code_acc` | pass@1, the fraction where all hidden tests pass. This is the metric that actually needs to improve |
| `train/test_pass_rate` | Mean per-test pass rate. Smoother than `code_acc`; watch this one first in the early phase |
| `train/avg_reward` | Mean of the shaped reward. If it rises while `code_acc` does not, the model is farming the 0.1 submit floor |
| `train/avg_turns` | Mean turn count. Sitting near `MAX_TURNS` means the model often exhausts its turns without submitting |
| `train/max_turns` / `train/min_turns` | Turn distribution within the batch. `max_turns` pinned at the limit means some trajectories are truncated; `min_turns` near 1 means trajectories often submit or give up on the first turn |

Policy health (registered via `model.add_metric('GRPOMetric', ...)`):

| Metric | Meaning and interpretation |
|---|---|
| `train/approx_kl` | KL between the old and new policy (Schulman K3 estimator). A sudden spike precedes policy collapse |
| `train/clip_ratio` | PPO clipping trigger rate (with `_low` / `_high` broken out by direction). Persistently high means single-step updates are too aggressive |
| `train/token_kl_max` / `train/token_ratio_max` | Per-token KL / probability-ratio extremes, used to localize a collapse |
| `train/policy_confidence` | `exp(mean_logp)`, the policy's mean confidence |

`GRPOMetric`'s `epsilon` must match `set_loss('GRPOLoss', epsilon=...)`, otherwise the clipping threshold reflected in `clip_ratio` is not the one actually in effect. `train/entropy` only appears with `GRPOLoss(entropy_coef > 0)`, but that option adds an entropy bonus to the loss and **thereby changes the training objective**, so it should not be enabled merely to observe a metric.

## Porting to your own task

Reusing this skeleton typically requires changes in four places; the training loop, GRPO configuration, weight synchronization, and metrics all stay as they are:

1. **Dataset**: replace `load_mbpp()` with your own loader, producing `{'prompt', ...fields needed for scoring}`.
2. **Tools**: `TOOL_SCHEMA` and handlers. Server capabilities go through the default action path; client-side bookkeeping uses `register_tool`.
3. **Reward**: `score()` inside `extract_rewards()`. Prefer signals that can be computed objectively (unit tests, exact match, executable validation), and consider a judge model only when no such signal exists.
4. **System prompt**: must describe backend semantics accurately (whether state persists across turns, which modules and permissions are available, the turn budget).

To add a backend: write `_xxx.py` (providing `NAME`, `SYSTEM_PROMPT`, `TOOL_SCHEMA`, `make_env()`, `run_tests()`, `describe()`) along with `run_xxx.sh`, and add the name to `BACKENDS` in `train.py`.

---

# Troubleshooting

## Pre-flight checklist

- [ ] The service is **not** bound to `0.0.0.0` (confirm with `ss -tlnp | grep 8000`)
- [ ] Direct private-network access: security-group ingress admits only the training host, not `0.0.0.0/0`
- [ ] Across networks: the environment side binds `127.0.0.1` and is reached only through SSH port forwarding
- [ ] Capacity >= `BATCH_SIZE x NUM_GENERATIONS` (for AgentENV, check memory separately)
- [ ] The environment service has keep-alive configured (systemd) and the SSH forward has keep-alive (autossh), rather than running in an interactive shell
- [ ] The full pipeline has been validated with `--batch-size 2 --num-generations 4 --max-steps 2`
- [ ] AgentENV's `always_denied_cidrs` has not been relaxed just to "make the sandbox reachable"

## Environment side

| Symptom | Cause and remedy |
|---|---|
| Some observations in a batch are entirely `Error:` | Insufficient server capacity. Verify `WORKERS x MAX_CONCURRENT_ENVS >= BATCH_SIZE x NUM_GENERATIONS` |
| The model reports `import math` is not allowed | The `ALLOWED_IMPORTS` allowlist did not take effect; confirm `_configure()` is called again after `reset()` |
| `ConcurrencyConfigurationError` | The upstream `coding_env.server.app` (single session) was started. Use `openenv_server/server_app.py` instead |
| Anomalous ±0.1 / -1.0 values mixed into the reward | The environment's reward transform was not disabled; check `self.transform = None` |
| Timeout / message-wait errors | Raise `OPENENV_MESSAGE_TIMEOUT_S`. The executor itself has three further limits: 30s wall-clock per execution, 10 million operations, and 1 million while-loop iterations, so the value must stay above 30s to be meaningful |
| `Address already in use` | The port is taken. Find the holding process with `ss -tlnp \| grep 8000`, or switch to `PORT=8001` |

## AgentENV side

| Symptom | Cause and remedy |
|---|---|
| `/dev/kvm is not accessible` | The runtime account is not in the kvm group, or the host has no KVM. Run `sudo server --setup-host --runtime-user aenv --runtime-group aenv`, then restart |
| `ublk_drv is not loaded` | Run `sudo modprobe ublk_drv`; kernels older than 6.8 need an upgrade |
| `ImportError: AgentEnv requires the E2B SDK` | `pip install e2b` |
| `Invalid API key format: expected "e2b_"` | Client-side validation in the e2b SDK. `AgentEnv` already sets `E2B_VALIDATE_API_KEY=false` by default, so a persisting error means it was explicitly overridden to `true` |
| `400: template xxx not found` | The template was not created, or the build has not reached ready. Check the state with `aenv template list` |
| `alias 'xxx' already points to ...` | Aliases cannot be rebound; run `aenv template delete xxx` first |
| `pip install` fails inside the sandbox | Egress is blocked by policy, or the current network cannot reach PyPI; pre-install in the template instead |
| A sandbox becomes unavailable mid-trajectory | `SANDBOX_TIMEOUT` is shorter than the trajectory duration and the sandbox was auto-paused. Raise it |
| Batch startup is slow | Raise `ENV_CONCURRENCY`; confirm dependencies are baked into the template rather than installed at runtime |
| Dependencies are re-downloaded on every restart | `AENV_HOME_PATH` was unset and fell back to `/tmp/aenv-test-<uid>/` (the test default in `run-with-capabilities.sh`), which `/tmp` cleanup removes. `serve.sh` pins it to `/var/lib/aenv` |
| `load config ... Permission denied` | A source-built binary hard-codes its build-time path as the default config location, which is unreadable under `/root` after dropping privileges to `aenv`. Point `AENV_CONFIG_PATH` at a copy readable by `aenv`; `serve.sh` already handles this |

## Training side

| Symptom | Cause and remedy |
|---|---|
| Too few valid trajectories, batch skipped | Most trajectories were filtered for being too long. Lower `MAX_TURNS` / `--max-tokens`, or tighten tool output (observations on the AgentENV side are truncated to 32K characters by default); also confirm the trajectory count is >= `--model-gpus` |
| Reward stays at 0 | First run `run_tests` alone against a known-correct solution to confirm the scoring path itself works, then investigate the model |
| The number of envs does not match `--batch-size` | Arguments were placed before `sh run_*.sh` and never reached `"$@"` |

---

# Appendix: Workarounds for Restricted Networks

This section is only relevant when **the default path does not work**, and applies to environments that cannot reach public sources such as Docker Hub, GitHub, and PyPI directly (isolated corporate networks, internal clusters, or regions with egress restrictions). If `install.sh` succeeded and the template is built, skip this section.

The registry addresses and package sources below are examples; substitute the internal mirrors actually reachable from your environment.

## Mirror sources

The default Dockerfile depends on external networks in two places — `FROM python:3.11-slim` and `RUN pip install` — which must be handled separately.

**Base image**: the server only searches the registries listed in `[image.resolver] search_registries`, which defaults to `docker.io` / `ghcr.io`. When neither is reachable, the typical errors are `dial tcp ...: i/o timeout` (timeout) or 401 (authentication required). Writing a fully qualified name bypasses the search list, and **`library/` must not be omitted** (the real path of official images on the Hub is `library/python`):

```bash
BASE_IMAGE=<your-registry>/library/python:3.11-slim sh agentenv_server/install.sh
```

`BASE_IMAGE` is forwarded to `aenv build --image`, overriding `FROM` in the Dockerfile, so the file itself stays untouched.

Probing candidate mirrors for reachability first avoids a wasted build. Both 200 and 401 count as reachable (401 means a token must be fetched first, which is normal), while a timeout means unavailable:

```bash
for R in <registry-1> <registry-2>; do
    printf "%-24s " "$R"
    timeout 15 curl -s -o /dev/null -w '%{http_code}\n' \
        "https://$R/v2/library/python/manifests/3.11-slim"
done
```

A status code alone is not proof of usability: some proxies complete the TLS handshake yet cannot serve a full manifest. To confirm, compare response sizes — a genuine multi-arch index is several KB, whereas an error page is typically under 200 bytes.

Public proxy sites may become unavailable or rate-limited at any time and are unsuitable for long training runs. For production, mirror the images into your own container registry (Harbor, or a managed registry from your cloud provider) on the same private network as the environment host.

**pip**: if the sandbox cannot reach pypi.org, set `ENV PIP_INDEX_URL` in the Dockerfile (the default Dockerfile does not include this line, so it must be added) and it must appear **before** `RUN pip install`.

Use `aenv template watch <id>` to see why a build failed. Two common causes:

- `all registry candidates failed during manifest fetch` — the base image cannot be pulled; substitute a mirror as above.
- `overlaybd-commit ... failed to perform commit(), 2: No such file or directory` — the image was pulled but conversion failed. First confirm `FROM` does not point at the wrong image (large images of tens of GB with hundreds of layers tend to stall here); once that is ruled out, check that overlaybd is fully installed: `/var/lib/aenv/deps/overlaybd/bin/` should contain the create / apply / commit / resize binaries, and `/etc/overlaybd/overlaybd.json` should exist.

The log line `open /root/.regctl/config.json: permission denied` is only a WARN (the server drops privileges to `aenv` and then tries to read root's home directory) and does not affect pulling public images. Configuring private registry credentials, however, requires giving it a writable HOME first:

```bash
sudo install -d -o aenv -g aenv /var/lib/aenv/home
```

## When install.sh cannot download: building from source

`install.sh` depends on only two external endpoints: `api.github.com` for release metadata, and then `github.com/.../releases/download/` for two assets (`aenv-linux-x86_64` at 9.3MB and `aenv-server-linux-x86_64.tar.gz` at 68MB). The latter redirects (302) to `release-assets.githubusercontent.com`, which in some network environments is prone to being RST mid-transfer (`curl: (56) Recv failure` or `(92)`).

Try a proxy first: re-run after `export https_proxy=...`, since `install.sh` uses curl throughout and honors the env proxy. Only without a proxy is the approach in this section necessary.

One point to establish up front: **compiling only solves the Rust binary part.** Besides `server`, the prebuilt tarball also contains a `deps/` directory (firecracker, guest kernel, tools driver, overlaybd, regctl), none of which are build artifacts, so they must be prepared separately. This is where the approach most often runs into trouble.

### Prerequisites

If `static.rust-lang.org` / `crates.io` are unreachable, a usable crates mirror must be configured (`<rust-mirror>` / `<crates-mirror>` are placeholders below; substitute addresses reachable from your environment). **The variables must be set before installing rustup**; reversing the order requires starting over:

```bash
# Write to the profile: AgentENV has a rust-toolchain.toml, so the first cargo
# build pulls the toolchain once more
cat >> ~/.bashrc <<'EOF'
export RUSTUP_DIST_SERVER=<rust-mirror>
export RUSTUP_UPDATE_ROOT=<rust-mirror>/rustup
EOF
. ~/.bashrc

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"

# Most mirrors only support the sparse index, requiring cargo >= 1.68. Write by
# overwriting: appending twice produces a duplicate [source.crates-io] and TOML
# parsing fails outright
tee "${CARGO_HOME:-$HOME/.cargo}/config.toml" >/dev/null <<'EOF'
[source.crates-io]
replace-with = 'mirror'

[source.mirror]
registry = "sparse+<crates-mirror>/"

[net]
retry = 5

[http]
timeout = 60
EOF

sudo apt-get update      # upgrade does not refresh the index; skipping this gives Unable to locate package

# Build dependencies (same as the builder stage of Dockerfile.agentenv). clang /
# libclang-dev are required by uvm-ublk-daemon: it uses bindgen to generate ublk
# kernel bindings, and without them you get Unable to find libclang
sudo apt-get install -y build-essential pkg-config libssl-dev \
     clang libclang-dev libprotobuf-dev protobuf-compiler

# Runtime dependencies (same as the runtime-base stage). libaio1t64 is the
# Ubuntu 24.04 package name; on 22.04 / Debian 12 it is called libaio1
sudo apt-get install -y ca-certificates curl dpkg e2fsprogs iproute2 iptables \
     jq libaio1t64 sudo umoci zstd
```

`protobuf-compiler` must come from the system package: `make ci-deps-protoc` downloads protoc from GitHub Releases, which is precisely the unreachable path. On Debian 13+, `pkg-config` has been renamed `pkgconf`.

Whether the mirror took effect is visible on the first line of `cargo build`, which must read ``Updating `mirror` index``. If it still says `Updating crates.io index`, `config.toml` was not read — common causes are `CARGO_HOME` pointing elsewhere, or `~/.cargo` not existing when the file was written. Note that the `an/yh/anyhow` form in error messages is a sparse path, but cargo >= 1.70 uses sparse by default, so **it cannot be taken as evidence that the mirror is in effect**.

### Building and installing

```bash
git clone https://github.com/kvcache-ai/AgentENV.git && cd AgentENV
cargo build --release -p agentenv --bin server
cargo build --release -p aenv
cargo build --release -p uvm-ublk -p uvm-ublk-daemon

sudo install -m 0755 target/release/aenv   /usr/local/bin/aenv
sudo install -m 0755 target/release/server /usr/local/bin/server
sudo install -D -m 0755 target/release/uvm-ublk-daemon /var/lib/aenv/ublk/uvm-ublk-daemon
sudo install -D -m 0640 config/default.toml /var/lib/aenv/config/config.toml

sudo groupadd --system aenv
sudo useradd --system --gid aenv --home-dir /var/lib/aenv \
     --no-create-home --shell /usr/sbin/nologin aenv
```

The location of that last `config.toml` is especially important: **a source-built binary hard-codes the build-time repository path as the default config location** (`CARGO_MANIFEST_DIR` in `cfg.rs`). Built under `/root/AgentENV`, it looks for `/root/AgentENV/config/default.toml`, which is inaccessible after dropping privileges to `aenv` because `/root` is mode 0700 — hence `Permission denied`. A copy must therefore be placed somewhere `aenv` can read, with `AENV_CONFIG_PATH` pointing at it on startup. `agentenv_server/serve.sh` already handles this.

### Preparing deps

```bash
E="AENV_CONFIG_PATH=/var/lib/aenv/config/config.toml AENV_HOME_PATH=/var/lib/aenv"
sudo env $E /usr/local/bin/server --setup-only
```

When downloads fail, first establish where the files come from — these five URLs are everything it fetches (defined in `config/deps_manifest.toml`):

```
firecracker + cpu-template-helper  https://pub-4ee15c400f554ab7a9eac3f5bc8f53de.r2.dev/firecracker-1.15.1-patch-v1-x86_64.tgz
guest kernel                       https://pub-4ee15c400f554ab7a9eac3f5bc8f53de.r2.dev/vmlinux-6.1.175
regctl                             https://github.com/regclient/regclient/releases/download/v0.11.5/regctl-linux-amd64
overlaybd .deb                     https://github.com/containerd/overlaybd/releases/download/v1.0.18/overlaybd-1.0.18-20260710.cee2186.{target}.deb
tools.ext4                         ghcr.io/zlzgithub-0801/agentenv-tools:0.1.0   (an OCI image; export with regctl/docker)
```

Do not hard-code `{target}`: `overlaybd.rs` reads `/etc/os-release` and substitutes `ubuntu1.<version>.<arch>` automatically. The only part that must be copied verbatim is the `1.0.18-20260710.cee2186` date-plus-git-hash segment.

Download the failing assets on a machine with connectivity, then **pre-place the files** — `download_file` skips the download when the target file already exists and is non-empty, and the target path is the `dest=` value from the failure log:

```bash
# Log: downloading url="https://pub-...r2.dev/firecracker-1.15.1-patch-v1-x86_64.tgz"
#                   dest=/var/lib/aenv/deps/firecracker/1.15.1-patch-v1/firecracker-1.15.1-patch-v1-x86_64.tgz
# scp to the dest= path (keeping the filename identical), then:
sudo chown -R aenv:aenv /var/lib/aenv/deps
sudo env $E /usr/local/bin/server --setup-only
```

This method works for all five dependencies without configuration changes. overlaybd has no local-path switch but responds to the same approach: place the `.deb` under `/var/lib/aenv/deps/overlaybd/downloads/` (filename taken from the last URL segment, with `{target}` already substituted) and `package_url` will never be accessed.

Alternatively, edit `/var/lib/aenv/config/config.toml` to point at the directory holding the files. Note that the `[firecracker]`, `[kernel]`, and `[tools]` **sections already exist**, so the keys must be added inside them; **do not append duplicate sections** — a repeated table causes TOML parsing to fail outright:

```toml
[firecracker]                                      # existing section; boot_args etc. stay as-is
binary_path = "/opt/aenv-assets/firecracker"       # the extracted binary, not the tgz
[kernel]                                           # existing section (empty)
image_path = "/opt/aenv-assets/vmlinux.bin"
[tools]                                             # existing section; control_plane_port stays as-is
version = "0.1.0"                                  # must be paired with drive_path
drive_path = "/opt/aenv-assets/tools.ext4"
```

`/opt/aenv-assets` must be created by hand; it is not a pre-existing directory.

### Starting the service

```bash
sudo env $E /usr/local/bin/server --setup-host --runtime-user aenv --runtime-group aenv
sudo chown -R aenv:aenv /var/lib/aenv
sh cookbook/rl/envs/agentenv_server/serve.sh
```

Several environment variables in `serve.sh` cannot be omitted: `AENV_RUN_USER=aenv` (without it the three-level fallback `SUDO_USER` → repository owner → `aenv` applies, giving unstable results when run directly as root), `AENV_HOME_PATH=/var/lib/aenv` (without it the path falls back to `/tmp/aenv-test-<uid>/`, requiring a few hundred MB to be re-downloaded after cleanup), and `AENV_CONFIG_PATH` (the build-path issue described above).

One more note: the change from `--setup-host` that adds `aenv` to the kvm group does not apply to existing sessions, which is why `run-with-capabilities.sh` re-initializes them via `--init-groups`. Starting the server directly, bypassing that script, fails for lack of kvm permissions.

> The build-from-source path has not been validated end to end, and the reachability of the deps downloads varies by machine.

## Docker deployment

```bash
curl -fsSL https://raw.githubusercontent.com/kvcache-ai/AgentENV/main/scripts/docker-setup.sh | sudo bash
docker run -d --privileged -v /dev:/dev -p 8000:8000 ghcr.io/kvcache-ai/aenv-server:latest
```

Docker here is only a **deployment vehicle**; the sandbox is still a Firecracker microVM and still requires KVM on the host. Pulling the image from `ghcr.io` also goes through GitHub distribution, which may be equally unreachable on a restricted network.

---

## Related Documents

- Component reference: [Execution Environments](../Components/Agentic/Envs.md) (the `Env` abstraction, `EnvTool`, both OpenEnv modes, `EnvPool`)
- Multi-turn tool calling: [Multi-Turn Tool Usage](../Components/Agentic/Multi-Turn-Tool-Usage.md)
- Runnable examples: `cookbook/rl/envs/` (code task, both backends), `cookbook/rl/multi_turn/` (embedded OpenEnv)
- OpenEnv upstream repository: <https://github.com/meta-pytorch/OpenEnv>
- AgentENV official documentation: <https://kvcache-ai.github.io/AgentENV/>
