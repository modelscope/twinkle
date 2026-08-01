# Code RL (MBPP)

一份训练脚本，两种执行后端。[English](#english) | 中文

## 任务

MBPP 数据集，模型写 Python 函数。每条轨迹：模型调 `run_python` 在沙箱里试代码，调 `submit_solution` 提交，训练侧用隐藏测试打分。

reward：全部通过 `1.0`；提交了但没全对 `0.1 + 0.4 × 通过率`；没提交 `0.0`。

Qwen3.5-4B + LoRA，GRPO，最多 6 轮工具调用。

## 两个后端

| | `openenv` | `agentenv` |
|---|---|---|
| 代码在哪跑 | OpenEnv 服务的一个 session | 一条轨迹一个 Firecracker microVM |
| 解释器 | AST 解释器（`coding_env`） | 真 CPython |
| 单 env 内存 | KB 级 | ~1GB |
| `run_python` 间状态 | 保留 | 每次新进程 |
| 文件 / 网络 / pip | 不支持 | 支持 |
| async / 装饰器 | 不支持（装饰器静默失效） | 支持 |

MBPP 用 `openenv` 够。需要 `unittest` + `@patch`、写文件、装包时用 `agentenv`。

## 资源需求

训练机：8 卡（`--model-gpus 4` + `--sampler-gpus 4`）。显存占用待实测补充。

环境机：

| 后端 | 要求 |
|---|---|
| `openenv` | 普通 CPU 机器，2 核 4G。可与训练机同机 |
| `agentenv` | 裸金属或支持嵌套虚拟化的实例，需 `/dev/kvm`、内核 6.8+、Ubuntu 24.04。容器实例通常不满足 |

`agentenv` 内存 = `batch-size × num-generations × 1GB + 8GB`，默认 32 并发约 40GB。

## 运行：openenv

环境机：

```bash
pip install openenv

# coding_env 是 OpenEnv 仓库里的子包（包名 openenv-coding_env），不在 PyPI 上，
# 只能从源码装。serve.sh 起的 server_app.py 从它 import PythonCodeActEnv 和
# PyExecutor；它的依赖里带 smolagents，也就是上表那个 AST 解释器的实现。
git clone https://github.com/huggingface/OpenEnv.git
pip install -e OpenEnv/envs/coding_env

HOST=127.0.0.1 sh serve.sh
```

训练机：

```bash
pip install openenv
sh run_openenv.sh
```

跨机时环境侧绑内网 IP（`HOST=10.0.1.20 sh serve.sh`），训练侧 `OPENENV_BASE_URL=http://10.0.1.20:8000 sh run_openenv.sh`。

## 运行：agentenv

环境机装服务端（Ubuntu 24.04，同时装上 `aenv` CLI）：

```bash
curl -fsSL https://raw.githubusercontent.com/kvcache-ai/AgentENV/main/scripts/install.sh | sudo bash
sudo systemctl start aenv
```

单独装 CLI（走 Docker 部署，或 CLI 与服务端不同机时）：

```bash
curl -fsSL https://raw.githubusercontent.com/kvcache-ai/AgentENV/main/scripts/install-cli.sh | bash
```

认证并构建模板（一次，改 `Dockerfile` 才需重建）：

```bash
aenv auth      # URL 填服务端地址；AgentENV 无认证，API key 随便填
aenv build cookbook/rl/code_rl/Dockerfile -t twinkle-code --cpu-count 1 --memory-mb 1024
aenv template watch <template-id>
```

训练机：

```bash
pip install e2b
AENV_API_URL=http://<环境机IP>:8000 sh run_agentenv.sh
```

## 参数

命令行参数透传给 `train.py`，覆盖 `common_args.sh` 里的默认值：

```bash
sh run_openenv.sh --max-steps 500 --batch-size 8
```

冒烟测试（`batch-size × num-generations` 需 ≥ `--model-gpus`）：

```bash
sh run_openenv.sh --batch-size 2 --num-generations 4 --max-steps 2
```

## 实验记录

待补充。

| backend | 配置 | steps | reward | 备注 |
|---|---|---|---|---|
| | | | | |

## 文件

| 文件 | 作用 |
|---|---|
| `train.py` | 训练逻辑，与后端无关 |
| `backends/openenv.py` `backends/agentenv.py` | env 构造、提示词、工具、隐藏测试回放 |
| `common_args.sh` | 训练超参 |
| `run_openenv.sh` `run_agentenv.sh` | 启动命令 |
| `serve.sh` `server_app.py` | OpenEnv 服务端 |
| `Dockerfile` | AgentENV 沙箱模板 |

加后端：写 `backends/xxx.py`（`NAME`、`SYSTEM_PROMPT`、`TOOL_SCHEMA`、`make_env()`、`run_tests()`、`describe()`）和 `run_xxx.sh`，不改 `train.py`。

跨网络部署见 `docs/source_zh/使用指引/Agentic RL部署与训练.md`。

---

<a name="english"></a>

# Code RL (MBPP) — English

One training script, two execution backends.

## Task

MBPP dataset, the model writes Python functions. Per trajectory: the model calls `run_python` to try code in a sandbox, calls `submit_solution`, and the trainer scores it against hidden tests.

Reward: `1.0` if all tests pass; `0.1 + 0.4 × pass_rate` if submitted but incorrect; `0.0` if never submitted.

Qwen3.5-4B + LoRA, GRPO, up to 6 tool-calling turns.

## Two backends

| | `openenv` | `agentenv` |
|---|---|---|
| Where code runs | A session on an OpenEnv server | One Firecracker microVM per trajectory |
| Interpreter | AST interpreter (`coding_env`) | Real CPython |
| Memory per env | KBs | ~1GB |
| State between `run_python` calls | Persists | Fresh process each call |
| Files / network / pip | No | Yes |
| async / decorators | No (decorators silently ignored) | Yes |

`openenv` is enough for MBPP. Use `agentenv` when you need `unittest` + `@patch`, file writes, or pip installs.

## Resources

Training host: 8 GPUs (`--model-gpus 4` + `--sampler-gpus 4`). Measured memory usage to be filled in.

Environment host:

| Backend | Requirement |
|---|---|
| `openenv` | An ordinary CPU machine, 2 cores / 4GB. Can be the training host itself |
| `agentenv` | Bare metal or nested-virtualisation instance; needs `/dev/kvm`, kernel 6.8+, Ubuntu 24.04. Container instances usually do not qualify |

`agentenv` memory = `batch-size × num-generations × 1GB + 8GB`, i.e. ~40GB at the default 32 concurrent sandboxes.

## Run: openenv

Environment host:

```bash
pip install openenv

# coding_env is a sub-package inside the OpenEnv repo (distribution name
# openenv-coding_env). It is not on PyPI, so it has to be installed from source.
# server_app.py, which serve.sh runs, imports PythonCodeActEnv and PyExecutor
# from it; its dependencies pull in smolagents — the AST interpreter above.
git clone https://github.com/huggingface/OpenEnv.git
pip install -e OpenEnv/envs/coding_env

HOST=127.0.0.1 sh serve.sh
```

Training host:

```bash
pip install openenv
sh run_openenv.sh
```

Across hosts, bind the environment to its private NIC (`HOST=10.0.1.20 sh serve.sh`) and run `OPENENV_BASE_URL=http://10.0.1.20:8000 sh run_openenv.sh`.

## Run: agentenv

Install the server on the environment host (Ubuntu 24.04; this also installs the `aenv` CLI):

```bash
curl -fsSL https://raw.githubusercontent.com/kvcache-ai/AgentENV/main/scripts/install.sh | sudo bash
sudo systemctl start aenv
```

Install the CLI alone (Docker deployment, or CLI on a different machine):

```bash
curl -fsSL https://raw.githubusercontent.com/kvcache-ai/AgentENV/main/scripts/install-cli.sh | bash
```

Authenticate and build the template (once; rebuild only when `Dockerfile` changes):

```bash
aenv auth      # URL is the server address; AgentENV has no authorization, any API key works
aenv build cookbook/rl/code_rl/Dockerfile -t twinkle-code --cpu-count 1 --memory-mb 1024
aenv template watch <template-id>
```

Training host:

```bash
pip install e2b
AENV_API_URL=http://<env-host-ip>:8000 sh run_agentenv.sh
```

## Arguments

Command-line arguments are forwarded to `train.py` and override the defaults in `common_args.sh`:

```bash
sh run_openenv.sh --max-steps 500 --batch-size 8
```

Smoke test (`batch-size × num-generations` must be ≥ `--model-gpus`):

```bash
sh run_openenv.sh --batch-size 2 --num-generations 4 --max-steps 2
```

## Results

To be filled in.

| backend | config | steps | reward | notes |
|---|---|---|---|---|
| | | | | |

## Files

| File | Role |
|---|---|
| `train.py` | Training logic, backend-agnostic |
| `backends/openenv.py`, `backends/agentenv.py` | env construction, prompt, tools, hidden-test replay |
| `common_args.sh` | Training hyper-parameters |
| `run_openenv.sh`, `run_agentenv.sh` | Launch commands |
| `serve.sh`, `server_app.py` | The OpenEnv server |
| `Dockerfile` | The AgentENV sandbox template |

To add a backend: write `backends/xxx.py` (`NAME`, `SYSTEM_PROMPT`, `TOOL_SCHEMA`, `make_env()`, `run_tests()`, `describe()`) and a `run_xxx.sh`. `train.py` stays untouched.

For cross-network deployment see `docs/source_en/Usage Guide/Agentic-RL-Deployment.md`.
