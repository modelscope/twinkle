# Deployment templates for Agentic RL

Network and supervision templates for the three runnable examples in
`cookbook/rl/`. Full reasoning and the selection matrix live in the
[Agentic RL Deployment Guide](../../../docs/source_en/Usage%20Guide/Agentic-RL-Deployment.md).

## Which example do I run?

| Need | Example | Environment runs |
|------|---------|------------------|
| Lightweight compute env (games, pure scoring) | `../multi_turn/` | In-process, sharded across Ray workers via `EnvPool` |
| Code execution, env scaled independently of training | `../openenv_code/` | A remote OpenEnv server (HTTP/WebSocket) |
| Real execution semantics (`unittest` + mock, files, `pip`) | `../agentenv/` | Firecracker microVMs via AgentENV |

Switching examples means changing the adapter class in code. Changing **where**
the environment runs is configuration only — one environment variable
(`OPENENV_BASE_URL` or `AENV_API_URL`).

## Which network setup?

Twinkle supports exactly two ways to reach an environment: **direct HTTP** and
**SSH port forwarding**. It ships no VPN / NAT-traversal tooling — that belongs to
your network team's existing infrastructure, and Twinkle only ever sees an
`http://host:port`.

| Topology | What to do |
|----------|------------|
| Training and env on one host | Bind to localhost: `HOST=127.0.0.1 sh serve.sh`. Nothing else to configure. |
| Separate hosts, same private network / VPC | **The default for most setups.** Bind to the private NIC; security-group inbound allows only the training host. |
| Across networks | Bind to `127.0.0.1` and forward the port over SSH: `ssh -N -L 8000:127.0.0.1:8000 user@env-host`. Point `OPENENV_BASE_URL` at the local entry. |

## Files

| File | Purpose |
|------|---------|
| `openenv-server.service` | systemd unit keeping the OpenEnv server alive across crashes and reboots. Install on the **environment** host. |

## Two things that bite people

**Neither OpenEnv nor AgentENV has any authentication.** Whoever can reach the
port can execute arbitrary code and consume your capacity. Never bind to
`0.0.0.0` on a reachable interface — restrict it with a security group, or keep
it on `127.0.0.1` and let SSH be the authentication layer.

**A dropped connection loses the whole rollout batch.** Supervise the server
with `openenv-server.service`, and if you use SSH forwarding, run it under
`autossh -M 0 -N -o ServerAliveInterval=30 ...` rather than in an interactive
shell.
