# Code RL (MBPP)

One training script, two environments: openenv and agentenv.

MBPP dataset, the model writes Python functions: it calls `run_python` to try code in a sandbox, calls `submit_solution`, and the trainer scores it against hidden tests. Qwen3.5-4B + LoRA, GRPO, up to 6 tool-calling turns.

Concepts, deployment, tuning and troubleshooting live in **[Agentic RL Deployment and Training](../../../docs/source_en/Usage%20Guide/Agentic-RL-Deployment-and-Training.md)**. This file only lists the commands.

## Choosing a backend

| | `openenv` | `agentenv` |
|---|---|---|
| Where code runs | A session on an OpenEnv server | One Firecracker microVM per trajectory |
| Interpreter | AST interpreter (`coding_env`) | Real CPython |
| Memory per env | KBs | ~1GB |
| Environment host | An ordinary CPU machine, can be the training host | Needs `/dev/kvm`, kernel 6.8+ |

`openenv` is enough for MBPP. Use `agentenv` when you need `unittest` + `@patch`, file writes, or pip installs.

Either way the training host has 8 GPUs (`--model-gpus 4` + `--sampler-gpus 4`).

## Run: openenv

Environment host:

```bash
sh openenv_server/install.sh        # pip install openenv + coding_env from source

HOST=127.0.0.1 sh openenv_server/serve.sh          # training on this same host
# HOST=10.0.1.20 sh openenv_server/serve.sh        # across hosts: bind the private NIC
```

Training host:

```bash
pip install openenv
sh run_openenv.sh
# across hosts: OPENENV_BASE_URL=http://10.0.1.20:8000 sh run_openenv.sh
```

## Run: agentenv

On the environment host, install the server and build the template (once; rebuild only when `Dockerfile` changes):

```bash
sh agentenv_server/install.sh                    # install + provision host + build template
sh agentenv_server/install.sh --rebuild          # delete the old template and rebuild
```

On a restricted network neither the base image nor pip resolves; point `BASE_IMAGE` at a reachable registry (it is forwarded to `aenv build --image`, so `Dockerfile` stays untouched):

```bash
BASE_IMAGE=<your-registry>/library/python:3.11-slim sh agentenv_server/install.sh
```

Mirror options and build-failure troubleshooting are in the [appendix](../../../docs/source_en/Usage%20Guide/Agentic-RL-Deployment-and-Training.md) of the deployment guide.

Start the server:

```bash
sh agentenv_server/serve.sh                      # foreground, binds 127.0.0.1:8000
NOHUP=1 sh agentenv_server/serve.sh             # background
```

Training host:

```bash
pip install e2b
# over HTTP directly:
AENV_API_URL=http://<env-host-ip>:8000 sh run_agentenv.sh
# or through an SSH tunnel:
ssh -N -L 8000:127.0.0.1:8000 root@ip-of-agentenv
# then, in another terminal
sh run_agentenv.sh
```

> Verify a sandbox boots before launching training:
>
> ```bash
> python -c "
> from twinkle_agentic.envs import AgentEnv
> e = AgentEnv(template='twinkle-code', api_url='http://127.0.0.1:8000')
> e.reset(); print('sandbox ok')
> print(e.run_command({'command': 'python -c \"import numpy, sympy; print(numpy.__version__)\"'}))
> "
> ```

## Arguments

Command-line arguments are forwarded to `train.py` and override the `TRAIN_ARGS` defaults in `run_*.sh`:

```bash
sh run_openenv.sh --max-steps 500 --batch-size 8
```

Smoke test. `batch-size × num-generations` **must be ≥ `--model-gpus`**, otherwise every batch is dropped by the length filter with only a warning:

```bash
sh run_openenv.sh --batch-size 2 --num-generations 4 --max-steps 2
```

`agentenv` memory = `batch-size × num-generations × 1GB + 8GB`, i.e. ~40GB at the default 32 concurrent sandboxes.

## Files

| File | Role |
|---|---|
| `train.py` | Training logic, backend-agnostic |
| `_openenv.py` `_agentenv.py` | env construction, prompt, tools, hidden-test replay (the `_` prefix keeps them from shadowing the same-named pip packages) |
| `openenv_server/` `agentenv_server/` | Per-backend `install.sh` (one-time setup) and `serve.sh` |
| `run_openenv.sh` `run_agentenv.sh` | Launch commands and training hyper-parameters (keep both `TRAIN_ARGS` in sync) |

To add a backend: write `_xxx.py` (`NAME`, `SYSTEM_PROMPT`, `TOOL_SCHEMA`, `make_env()`, `run_tests()`, `describe()`) and a `run_xxx.sh`, then add the name to `BACKENDS` in `train.py`.
