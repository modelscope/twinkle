# Agentic RL Best Practices

This guide walks through a complete, runnable **remote code-writing task** end to end: the environment runs on its own machine, training runs on the GPU machine, and the only thing connecting them is a URL.

Code writing makes a good first agentic RL task for concrete reasons: the reward can be computed objectively from unit tests, so no judge model is needed; multi-turn interaction is genuinely useful (write → run → fix → submit); and OpenEnv's bundled code environment means almost no environment-side development.

The runnable code lives in [`cookbook/rl/openenv_code/`](https://github.com/modelscope/twinkle/tree/main/cookbook/rl/openenv_code) (remote OpenEnv service) and [`cookbook/rl/agentenv/`](https://github.com/modelscope/twinkle/tree/main/cookbook/rl/agentenv) (Firecracker microVM). The two examples share the **same task, the same tool names and the same reward formula** — only the execution backend differs — so they can be read side by side.

## 1. Pick an Execution Backend

| | **OpenEnv embedded** | **OpenEnv server mode** | **AgentENV** |
|---|---|---|---|
| Adapter | `OpenEnv` + `EnvPool` | `OpenEnvClient` | `AgentEnv` |
| Isolation | None (same process) | Process / container | microVM (KVM) |
| Executor | Depends on the env package | smolagents AST interpreter (`coding_env`) | Real CPython |
| Special hardware | None | None | Requires `/dev/kvm` |
| Runs untrusted code | ❌ | ⚠️ Only within a controlled whitelist | ✅ |
| `assert` / files / `pip` | Depends on the env package | ❌ | ✅ |
| Deployment cost | Zero | One `uvicorn` command | Deploy the AgentENV control plane |

Recommendations:

- **Pure-compute environments** (board games, text games, scoring logic that is safe to evaluate in-process) → embedded `OpenEnv`, sharded with `EnvPool`.
- **Code execution / scaling the env independently of training / conflicting dependencies** → `OpenEnvClient` (the path this guide follows).
- **Genuinely untrusted code** (the model may write files, install packages, spawn processes) → AgentENV; see [Agentic RL with Sandbox Environments](./Agentic-RL-Sandbox.md).

> The key trade-off: OpenEnv's `coding_env` uses smolagents' `LocalPythonExecutor`, which is an **AST interpreter, not an OS-level sandbox**. It is good enough to enforce "only these modules may be imported", but do not use it for genuinely adversarial code. If you need that level of isolation, use AgentENV.

## 2. Environment Side: Serve OpenEnv as a Service

On the environment machine (no GPU, no KVM, no Docker required):

```bash
pip install openenv
pip install -e /path/to/OpenEnv/envs/coding_env   # brings in smolagents

cd cookbook/rl/openenv_code
sh serve.sh        # 4 workers x 64 sessions = 256 concurrent sessions
```

`serve.sh` starts [`server_app.py`](https://github.com/modelscope/twinkle/blob/main/cookbook/rl/openenv_code/server_app.py) from this folder rather than upstream's `coding_env.server.app`. It makes three **necessary** deviations from upstream defaults — copying upstream verbatim will bite you:

```python
class ConcurrentCodeEnv(PythonCodeActEnv):
    # 1. Upstream leaves this False, which makes create_app(max_concurrent_envs > 1)
    #    raise ConcurrencyConfigurationError and caps the server at ONE session.
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._configure()

    def reset(self, **kwargs):
        # The parent's reset() rebuilds the executor and transform with upstream
        # defaults, so re-apply our config afterwards.
        observation = super().reset()
        self._configure()
        return observation

    def _configure(self) -> None:
        # 2. Upstream authorises only `import json`, so math / collections — which
        #    many MBPP solutions need — would fail.
        self._executor = PyExecutor(additional_imports=list(ALLOWED_IMPORTS))
        # 3. Upstream's create_safe_coding_transform() overwrites observation.reward
        #    with code-style heuristics (-1.0 for open( / import os, +0.1 for short
        #    code). This task's reward comes from unit tests, so a style score on
        #    the same channel is pure noise.
        self.transform = None

app = create_app(ConcurrentCodeEnv, CodeAction, CodeObservation,
                 env_name='twinkle_code_env', max_concurrent_envs=MAX_CONCURRENT_ENVS)
```

Flipping `SUPPORTS_CONCURRENT_SESSIONS` is safe here: `create_app` receives the **class** as a factory, so every WebSocket connection gets a fresh instance, and `PythonCodeActEnv.__init__` builds a private executor and state that share nothing.

**Do the capacity math**: concurrent sessions = `WORKERS x MAX_CONCURRENT_ENVS`, and it must be ≥ `BATCH_SIZE x NUM_GENERATIONS` (4 x 8 = 32 with the defaults, against a server capacity of 256). Connections beyond capacity are rejected outright, which shows up as a subset of trajectories in the batch whose observations are all `Error:`.

> The HTTP `/step` and `/reset` endpoints will **not** work for this. Each of those requests builds a fresh env and `close()`s it on the way out, losing all state — they exist for debugging and stateless use. Multi-turn episodes must go over WebSocket, which `OpenEnvClient` handles for you.

## 3. Training Side: Three Pieces of Wiring

### 1. Tools: What the Model Sees

Only two tools are exposed ([`tools.py`](https://github.com/modelscope/twinkle/blob/main/cookbook/rl/openenv_code/tools.py)):

- `run_python(code)` — sent to the server and executed in the session. **The session namespace persists across turns**, so the model can define a function in one turn and call it in the next.
- `submit_solution(code)` — never sent to the server. Registered via `register_tool` and handled **client-side**, recording the final source on the env for the training loop to score.

```python
def _submit_solution(env: OpenEnvClient, arguments: Dict[str, Any]) -> str:
    code = (arguments.get('code') or '').strip()
    if not code:
        return "Error: 'code' argument is required."
    env.submitted_code = code
    return 'Solution submitted.'

def register_tools(env: OpenEnvClient) -> OpenEnvClient:
    env.submitted_code = None
    return env.register_tool(TOOL_SCHEMA[1], _submit_solution)
```

This is a general pattern: **"an action in the environment" and "bookkeeping the trainer needs" are different things**. Keep the latter in a local handler instead of polluting the environment protocol.

The system prompt has to state the backend's semantics precisely, or the model will code against the wrong mental model. The two examples are **opposites** on exactly this point:

- OpenEnv session: `The interpreter keeps its state between calls`, plus the module whitelist and a note that there is no file or network access.
- AgentENV: `Each call runs in a FRESH process, so every snippet must be self-contained`.

### 2. Environment: One Session per Trajectory

```python
def make_env() -> OpenEnvClient:
    env = OpenEnvClient(
        env_name=OPENENV_ENV_NAME,          # 'coding_env'; client + Action classes auto-discovered
        base_url=OPENENV_BASE_URL,          # a load-balancer address works too
        tools=[TOOL_SCHEMA[0]],             # only run_python goes to the server
        message_timeout_s=120.0,            # code execution can be slow
    )
    return register_tools(env)


def prepare_trajectories(samples, pool):
    envs = [make_env() for _ in samples]
    # reset() blocks on the WebSocket handshake plus server-side env creation,
    # so run them concurrently — otherwise a batch of 32 serializes.
    list(pool.map(lambda env: env.reset(), envs))

    trajectories, tool_managers = [], []
    for sample, env in zip(samples, envs):
        tool_managers.append(ToolManager(EnvTool.from_env(env)))
        trajectories.append({
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': sample['prompt']},
            ],
            'tools': TOOL_SCHEMA,
        })
    return trajectories, tool_managers, envs
```

Three things to get right:

1. **Do not wrap `OpenEnvClient` in `EnvPool` / `@remote_class`.** The session's lifetime is owned by the server, so sharding it again through Ray buys nothing and only adds an RPC hop.
2. **One `ToolManager` per trajectory.** A `ToolManager` holds a specific env instance; sharing one would route every trajectory's tool calls into the same session.
3. **Close `envs` in a `finally`.** Sessions occupy server capacity, and leaking them makes later steps fail for want of capacity:

```python
try:
    all_trajectories = rollout(expand_prompts, tool_manager=tool_managers)
    total_rewards, pass_rates = extract_rewards(envs, expanded, env_pool)
finally:
    close_envs(envs, env_pool)
```

### 3. Reward: Use Unit Tests, Not a Judge Model

Every MBPP sample ships a handful of `assert` statements. After the rollout, replay the hidden tests **in the same session**:

```python
def run_tests(env, test_list, setup_code=''):
    total = len(test_list)
    solution = getattr(env, 'submitted_code', None)
    if not solution:
        return 0, total
    if not _ok(env.execute({'code': solution})):     # define the solution in the namespace
        return 0, total
    passed = 0
    for test in test_list:
        expr = test.strip()
        if expr.startswith('assert '):
            expr = expr[len('assert '):]
        result = env.execute({'code': f'print({expr})'})
        if _ok(result) and _last_line(result.observation.stdout) == 'True':
            passed += 1
    return passed, total
```

Two deliberate design choices here:

- **`env.execute()` instead of `env.step()`.** `execute()` returns the server's **raw** `StepResult`, so typed fields such as `exit_code` are readable; `step()` returns text rendered for the model and accumulates episode reward. Scoring logic should not show up in the model's conversation.
- **Rewriting `assert X` into `print(X)` and running one test per step**, rather than submitting one block of asserts. The sandbox is smolagents' AST interpreter, whose support for `assert` / `try` varies across versions while plain expressions are stable; and one step per test isolates a test that raises, so the rest still run (the reward uses the pass rate).
  - Contrast: the AgentENV version runs real CPython, so it simply generates an ordinary script that wraps each assertion in `try/except` and prints `TESTS_PASSED n total`. Same reward, script shape dictated by backend capability.

Reward shaping:

```python
rate = passed / total if total else 0.0
if total and passed == total:
    return 1.0, rate                    # all tests pass
if getattr(env, 'submitted_code', None):
    return 0.1 + 0.4 * rate, rate       # submitted, partially correct -> 0.1 .. 0.5
return 0.0, rate                        # never submitted
```

The shape guarantees "all correct > partially correct > submitted but all wrong > never submitted", and the 1.0 for a full pass is clearly above the 0.5 ceiling for partial credit, so the model cannot profit from shipping a fake implementation that only satisfies the first test. The **0.1 floor for submitting at all** exists to provide gradient signal early on: otherwise every trajectory scores 0, every GRPO group advantage is 0, and nothing is learned.

### 4. GRPO: Group-Relative Advantages

```python
batch = [dataset[(sample_cursor + i) % len(dataset)] for i in range(BATCH_SIZE)]
expanded = [s for s in batch for _ in range(NUM_GENERATIONS)]   # N contiguous copies per problem
...
advantages = advantage_fn(total_rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()
```

Note that `expanded` uses `for s in batch for _ in range(N)`: the N rollouts of one problem are **contiguous** in the list, which is the layout `GRPOAdvantage` slices groups by. Writing `for _ in range(N) for s in batch` misaligns every group — training appears to run but learns nothing.

## 4. Running It

Environment machine:

```bash
cd cookbook/rl/openenv_code
sh serve.sh
```

Training machine (`pip install openenv` is enough; `coding_env` is not needed because the client class comes from `openenv`):

```bash
cd cookbook/rl/openenv_code
OPENENV_BASE_URL=http://<env-host-ip>:8000 sh openenv_code_grpo.sh
```

Both environment variables and CLI arguments can be overridden at invocation:

```bash
OPENENV_BASE_URL=http://10.0.0.5:8000 \
MAX_TURNS=8 ENV_CONCURRENCY=32 \
sh openenv_code_grpo.sh --batch-size 8 --num-generations 16 --max-steps 500
```

After changing `batch-size` / `num-generations`, **scale server capacity to match**: `8 x 16 = 128` concurrent sessions still fits under `serve.sh`'s default 256, but beyond that raise `WORKERS` or `MAX_CONCURRENT_ENVS`.

Key log line:

```
[Step 0] avg_reward=0.145, solve_rate=0.031, test_pass_rate=0.208, avg_turns=4.2
```

| Metric | Meaning and interpretation |
|--------|----------------------------|
| `solve_rate` | Fraction passing every hidden test. This is the metric that must go up |
| `test_pass_rate` | Mean per-test pass rate. Smoother than `solve_rate`; watch it first early on |
| `avg_reward` | Mean shaped reward. If it rises while `solve_rate` does not, the model is farming the 0.1 submission floor |
| `avg_turns` | Mean turns. Pinned at `MAX_TURNS` means the model often burns its budget without submitting |

## 5. Network Hardening (read before any shared deployment)

**Neither OpenEnv nor AgentENV has any authentication.** AgentENV's README states it verbatim — *"AgentENV currently does not support authorization"* — and OpenEnv's `serve.sh` defaults to `--host 0.0.0.0`. Anyone who can reach the port can execute code in your sandbox and exhaust your capacity.

Pick one, easiest first:

| Situation | What to do |
|-----------|------------|
| Training on the same host (simplest) | `HOST=127.0.0.1 sh serve.sh` — never touches the network |
| Across hosts (recommended) | Security-group inbound rule allowing **only the training host's private IP/32**, or its security-group ID. **Never 0.0.0.0/0** |
| Stronger control needed | `iptables`/`nftables` source-IP restriction, or front it with nginx/caddy doing mTLS / token checks |

⚠️ AgentENV's `aenv auth` API key is **not authentication** — any non-empty string works for a local deployment, since the CLI merely requires the field to be non-empty. Do not mistake it for a security control.

For **egress** (where sandboxed code may connect), AgentENV enforces a node-level policy in `config/default.toml`, and the defaults are sensible:

```toml
[network.egress]
# A sandbox's own egress policy cannot override these
always_denied_cidrs = [
  "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8",
  "169.254.0.0/16",              # ← the critical one: blocks cloud metadata (169.254.169.254)
  "172.16.0.0/12", "192.168.0.0/16",
]
```

That `169.254.0.0/16` entry matters most: it blocks the cloud metadata service, preventing sandboxed code from stealing temporary IAM credentials and escalating to your whole cloud account. The task in this guide (standard-library algorithm problems) needs no egress at all, so the defaults apply as-is.

## 6. Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| Some trajectories' observations are all `Error:` | Server session capacity exhausted. Check `WORKERS x MAX_CONCURRENT_ENVS ≥ BATCH_SIZE x NUM_GENERATIONS` |
| `ConcurrencyConfigurationError` | You are serving upstream `coding_env.server.app` (single session). Use this folder's `server_app.py` |
| Odd ±0.1 / -1.0 values in the reward | The env's reward transform is still active; check `self.transform = None` |
| Model reports `import math` is not allowed | The `ALLOWED_IMPORTS` whitelist was lost; make sure `_configure()` is re-applied after `reset()` |
| Timeouts / message-wait errors | Raise `message_timeout_s`. Note the executor enforces three caps of its own: a 30s wall-clock limit per execution (`MAX_EXECUTION_TIME_SECONDS`), 10M operations, and 1M while-iterations — so `message_timeout_s` only helps above 30s |
| Batch skipped for too few valid trajectories | Most trajectories were filtered for length. Lower `MAX_TURNS` / `max_tokens`, or tighten tool output |
| Reward stays at 0 forever | Score a known-correct solution through `run_tests` in isolation first to prove the scoring path works, then suspect the model |

## 7. Porting to Your Own Task

Reusing this skeleton usually means changing four things:

1. **Dataset**: replace `load_mbpp()` with your loader, producing `{'prompt', ...fields needed for scoring}`.
2. **Tools**: `TOOL_SCHEMA` plus handlers. Server-backed capabilities use the default action path; bookkeeping uses `register_tool`.
3. **Reward**: `extract_rewards()`. Prefer objectively computable signals (unit tests, exact match, executable verification); reach for a judge model only when there is none.
4. **System prompt**: it must describe the backend accurately (whether state persists across turns, which modules and permissions exist, the turn budget).

The training loop, GRPO configuration, weight syncing and metrics normally need no changes.

## Related Documents

- Component reference: [Environments](../Components/Agentic/Envs.md) (`Env` abstraction, `EnvTool`, both OpenEnv modes, `EnvPool`)
- Deployment selection and cross-network access: [Agentic RL Deployment Guide](./Agentic-RL-Deployment.md) (backend matrix, four deployment tiers, SSH port forwarding)
- Multi-turn tool calling: [Multi-Turn Tool Usage](../Components/Agentic/Multi-Turn-Tool-Usage.md)
- Strong microVM isolation: [Agentic RL with Sandbox Environments](./Agentic-RL-Sandbox.md)
- Runnable examples: `cookbook/rl/openenv_code/` (remote OpenEnv), `cookbook/rl/agentenv/` (AgentENV microVM)
- OpenEnv upstream repository: <https://github.com/meta-pytorch/OpenEnv>
