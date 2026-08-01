# Agentic RL Deployment Guide

Twinkle assumes nothing about which sandbox you use or what network your machines sit in. This guide is a **lookup table**: pick an execution backend, then pick a deployment tier.

The conclusion that shapes your planning, up front:

> **The two dimensions are orthogonal.** Changing the execution backend means changing code (a different adapter class); changing the network topology means changing one environment variable.
> So **get the flow working on a single machine first — moving to multi-host or across the internet later requires no code changes at all.**

## 1. First: Pick an Execution Backend

Each of the three routes has a runnable example to copy from:

| Your environment needs | Example | Where the env runs | Isolation | Memory per env |
|---|---|---|---|---|
| Pure compute (board games, text games, safely evaluable scoring) | [`cookbook/rl/multi_turn/`](https://github.com/modelscope/twinkle/tree/main/cookbook/rl/multi_turn) | In the training process, sharded across Ray workers by `EnvPool` | None | KBs |
| Code execution; env scaled independently of training | [`cookbook/rl/openenv_code/`](https://github.com/modelscope/twinkle/tree/main/cookbook/rl/openenv_code) | A standalone OpenEnv service (HTTP/WebSocket) | Process / container | KBs |
| Real execution semantics (`unittest`+mock, files, `pip`, subprocesses) | [`cookbook/rl/agentenv/`](https://github.com/modelscope/twinkle/tree/main/cookbook/rl/agentenv) | AgentENV Firecracker microVMs | microVM (KVM) | **~1GB** |

The short version:

- **Tasks solvable with the standard library** → OpenEnv. Three orders of magnitude lighter; no reason to reach for microVMs.
- **Running `unittest` + `@patch`, or the model needs to write files / install packages** → AgentENV is required. OpenEnv's `coding_env` uses smolagents' AST interpreter, where **decorators are silently ignored** (`decorator_list` is referenced nowhere in its source). Mocks stop working without raising, and the reward comes out as a plausible-looking wrong number.
- Full capability comparison: [Environments](../Components/Agentic/Envs.md).

## 2. Second: Pick a Deployment Tier

Four tiers, from single-machine validation to across the public internet. **L2 and L3 share identical training code** — only the network differs.

| Tier | Situation | Environment deployment | Network |
|---|---|---|---|
| **L0** | Get it working | Same process / same host as training | None |
| **L1** | Single node, multi-GPU; move env overhead off the GPU process | `EnvPool` + dedicated CPU workers | None (inside the Ray cluster) |
| **L2** | Env scaled independently / dependency conflicts / isolation needed | Separate host | Private network + security group |
| **L3** | Env host and training host on different networks | Separate host | SSH port forwarding |

### L0 — Single machine

Embedded (the environment is instantiated in the driver, zero network overhead):

```bash
cd cookbook/rl/multi_turn && python multi_turn_grpo.py
```

Or an OpenEnv server bound to loopback, with training on the same host:

```bash
cd cookbook/rl/openenv_code
HOST=127.0.0.1 sh serve.sh &
OPENENV_BASE_URL=http://127.0.0.1:8000 sh openenv_code_grpo.sh \
    --batch-size 2 --num-generations 4 --max-steps 2
```

**Prove the full path at low concurrency before scaling up** — it saves both time and money. But **do not shrink it too far**: the trajectory count (`batch-size × num-generations`) must be ≥ `--model-gpus` (4 by default), otherwise too few survive the length filter and the whole batch is skipped — the log just repeats `skipping this batch`, which looks like a hang rather than an error. The `2×4=8` above leaves 2x headroom.

### L1 — Environments on dedicated CPU workers ("OpenEnv + Ray")

Environments are still instantiated in-process, but `EnvPool` shards N instances onto a separate CPU `DeviceGroup`, off the GPU process's memory and GIL:

```bash
cd cookbook/rl/multi_turn
ENV_REMOTE=1 ENV_NUM_WORKERS=8 ENV_POOL_SIZE=64 python multi_turn_grpo.py
```

| Variable | Effect |
|---|---|
| `ENV_REMOTE=1` | Put envs on a dedicated CPU DeviceGroup; unset runs them in the driver (zero RPC overhead) |
| `ENV_NUM_WORKERS` | Number of CPU workers; each rank becomes one `EnvPool` worker |
| `ENV_POOL_SIZE` | Pool capacity; `0` means auto (the trajectory count) |

⚠️ **This tier applies only to the embedded `OpenEnv`.** Do not put `OpenEnvClient` or `AgentEnv` into an `EnvPool` — their session/sandbox lifetime is owned by the server, so sharding again through Ray buys nothing and only adds an RPC hop.

### L2 — Environment on its own host (private network, the default for most setups)

This is **the recommended default for most setups**: a company's GPU and CPU machines are usually already in the same VPC / datacentre / Kubernetes cluster, and need no VPN at all.

Bind the environment to the **private NIC** (not `0.0.0.0`):

```bash
# OpenEnv
HOST=10.0.1.20 sh serve.sh

# AgentENV: configure its listener to the private IP after installation
```

The training side changes exactly one environment variable:

```bash
OPENENV_BASE_URL=http://10.0.1.20:8000 sh openenv_code_grpo.sh   # OpenEnv
AENV_API_URL=http://10.0.1.20:8000      sh agentenv_grpo.sh      # AgentENV
```

Three things to get right:

| Item | How |
|---|---|
| Narrow the inbound rule | The security group allows **only the training host's IP/32 or its security-group ID**, port 8000 only, never `0.0.0.0/0` |
| Confirm the bind took effect | `ss -tlnp \| grep 8000` must show `10.0.1.20:8000`, not `0.0.0.0:8000` |
| Supervise the service | Use [`deploy/openenv-server.service`](https://github.com/modelscope/twinkle/blob/main/cookbook/rl/deploy/openenv-server.service) — a rollout batch dies with the server |

Traffic inside one VPC never touches the public internet, and with a source-restricted security group **this tier needs no extra networking components at all**.

### L3 — Across networks (training and environment on different networks)

The training commands are identical to L2; only the address changes to the local end of an SSH port forward — see the next section.

## 3. Cross-Network Access: SSH Port Forwarding

Twinkle supports exactly two ways to reach an environment:

| Situation | Connection |
|---|---|
| Same host / same private network (L0–L2) | **Direct HTTP** |
| Training and environment on different networks (L3) | **SSH port forwarding** |

The framework neither ships nor recommends VPN / NAT-traversal components. That layer belongs to your network team's existing infrastructure and is orthogonal to the training code — Twinkle only ever sees an `http://host:port`.

**Why SSH for cross-network**: neither OpenEnv nor AgentENV has **any authentication**, so a reachable port means reachable arbitrary code execution. SSH's role here is not "connecting" but **putting authentication in front of a service that has none** — reusing your existing SSH keys and audited logins, on a channel most companies already approve (a bastion host is exactly this pattern).

SSH port forwarding operates at the TCP layer and is fully transparent to WebSocket — verified: handshake, session state across messages, and 8 concurrent sessions over a single connection all work.

```bash
# Environment side: loopback only, unreachable from the network
HOST=127.0.0.1 sh serve.sh

# Training side: set up the forward, then connect to the local port
ssh -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
    -L 8000:127.0.0.1:8000 user@env-host &
OPENENV_BASE_URL=http://127.0.0.1:8000 sh openenv_code_grpo.sh
```

Two addresses are easy to confuse: `OPENENV_BASE_URL` points at the **local entry** `127.0.0.1:8000`, while the `127.0.0.1:8000` inside `-L` is resolved **from the environment host's point of view**. Same for AgentENV via `AENV_API_URL`.

**Long runs must be supervised**: if the forward drops, the whole rollout batch is lost. Do not leave it in an interactive shell — use autossh:

```bash
autossh -M 0 -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -L 8000:127.0.0.1:8000 user@env-host
```

On throughput: all traffic shares one TCP connection. Each turn carries only a few KB of code and output, and 8 concurrent sessions showed no strain in testing; at hundreds of concurrent sessions, open several forwards on different local ports to spread the load.

> When both sides already sit in the same VPC / datacentre / Kubernetes cluster, L2's direct HTTP is enough and SSH is unnecessary — that covers the large majority of corporate setups.

## 4. Capacity Planning

**OpenEnv server**: concurrent session ceiling = `WORKERS × MAX_CONCURRENT_ENVS`, which must be ≥ `BATCH_SIZE × NUM_GENERATIONS`. `serve.sh` defaults to `4 × 64 = 256` while the default training config needs only `4 × 8 = 32`, leaving ample headroom. Connections beyond capacity are rejected outright, showing up as trajectories whose observations are all `Error:`.

**AgentENV**: memory is the only hard constraint.

```
required memory = concurrent trajectories × per-sandbox memory + 8GB (AgentENV + OS)
concurrent trajectories = BATCH_SIZE × NUM_GENERATIONS
```

With the example's `--memory-mb 1024`: the default 32 concurrent needs ~40GB, while a `2×4=8` concurrent smoke test needs ~16GB. **Do not buy a large machine for the trial phase.**

AgentENV also has hard prerequisites (`/dev/kvm`, kernel 6.8+, `modprobe ublk_drv`, `CAP_SYS_ADMIN`) that container instances typically fail — see [Agentic RL with Sandbox Environments](./Agentic-RL-Sandbox.md).

## 5. Pre-Flight Checklist

- [ ] The service is **not** bound to `0.0.0.0` (verify with `ss -tlnp | grep 8000`)
- [ ] L2: security-group inbound allows only the training host, not `0.0.0.0/0`
- [ ] L3: the environment binds `127.0.0.1` and is reached only through SSH port forwarding
- [ ] Capacity ≥ `BATCH_SIZE × NUM_GENERATIONS` (plus the memory check for AgentENV)
- [ ] The environment service is supervised (systemd) and the SSH forward too (autossh), neither left in an interactive shell
- [ ] The full path was proven with `--batch-size 2 --num-generations 4 --max-steps 2`
- [ ] Trajectory count (`batch-size × num-generations`) ≥ `--model-gpus`, or batches are skipped with only a warning
- [ ] AgentENV's `always_denied_cidrs` was not loosened to "make the sandbox reachable"

## Related Documents

- End-to-end training flow and reward design: [Agentic RL Best Practices](./Agentic-RL-Best-Practices.md)
- AgentENV single-machine deployment in detail: [Agentic RL with Sandbox Environments](./Agentic-RL-Sandbox.md)
- Adapter and API reference: [Environments](../Components/Agentic/Envs.md)
- Deployment templates: `cookbook/rl/deploy/`
