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

`aenv build` 是**服务端**拉镜像并转 overlaybd，本机不需要 docker。别名不可改绑，重建前先 `aenv template delete twinkle-code`。

国内网络下 `Dockerfile` 里的 `FROM python:3.11-slim` 拉不到——服务端默认只搜 `docker.io` / `ghcr.io`，前者超时、后者 401。换成能通的镜像站（**`library/` 不能省**，官方镜像的真实路径是 `library/python`）：

```dockerfile
FROM docker.1panel.live/library/python:3.11-slim
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
```

先在环境机上验，200/401 都算通，超时就换下一个：

```bash
for R in docker.1panel.live docker.m.daocloud.io docker.1ms.run; do
    printf "%-24s " "$R"
    timeout 15 curl -s -o /dev/null -w '%{http_code}\n' \
        "https://$R/v2/library/python/manifests/3.11-slim"
done
```

这些是第三方公益站，随时可能挂或限速；长期用建议转存到自己的 ACR。`PIP_INDEX_URL` 同理必须换，沙箱走官方 pypi 也不通。

训练机：

```bash
pip install e2b
AENV_API_URL=http://<环境机IP>:8000 sh run_agentenv.sh
```

只设 `E2B_API_KEY` / `E2B_API_URL` 无效——必须用 `AENV_API_URL`，否则会静默回落到 `http://127.0.0.1:8000`。

服务端只绑了 `127.0.0.1` 而训练在别的机器上时，用 SSH 隧道（`ssh -N` 不返回是正常的，挂着就是隧道在工作；`-f` 转后台）：

```bash
ssh -f -N -L 8000:127.0.0.1:8000 root@<环境机IP>
```

跑训练前先验沙箱能起，几秒出结果，比等 vLLM 加载快得多：

```bash
python -c "
from twinkle_agentic.envs import AgentEnv
e = AgentEnv(template='twinkle-code', api_url='http://127.0.0.1:8000')
e.reset(); print('sandbox ok')
print(e.run_command({'command': 'python -c \"import numpy, sympy; print(numpy.__version__)\"'}))
"
```

## 参数

命令行参数透传给 `train.py`，覆盖 `common_args.sh` 里的默认值：

```bash
sh run_openenv.sh --max-steps 500 --batch-size 8
```

冒烟测试。`batch-size × num-generations` **必须 ≥ `--model-gpus`**，否则每个 batch 都被长度过滤器静默丢掉、只有一条 warning，比报错难查。参数要放在脚本名**后面**才生效：

```bash
sh run_openenv.sh --batch-size 2 --num-generations 4 --max-steps 2
```

## 排查

| 现象 | 原因 |
|---|---|
| `Invalid API key format: expected "e2b_"` | e2b SDK 的客户端本地校验。`AgentEnv` 已默认设 `E2B_VALIDATE_API_KEY=false`；仍报错说明被显式覆盖成 `true` 了 |
| `400: template twinkle-code not found` | 模板没建，或 build 失败没到 ready。`aenv template list` 看状态 |
| `alias 'twinkle-code' already points to ...` | 别名不可改绑，先 `aenv template delete twinkle-code` |
| `dial tcp ...: i/o timeout` on `registry-1.docker.io` | Docker Hub 不通，`FROM` 换镜像站 |
| `overlaybd-commit ... No such file or directory` | overlaybd 装得不全。查 `/var/lib/aenv/deps/overlaybd/bin/` 是否有 create/apply/commit/resize 四个、`/etc/overlaybd/overlaybd.json` 是否存在 |
| `open /root/.regctl/config.json: permission denied` | server 降权成 `aenv` 却读 root 家目录。拉公开镜像时只是 WARN；配私有 registry 凭据前要给它一个可写 HOME（`install -d -o aenv -g aenv /var/lib/aenv/home` 并设 `HOME=`） |
| env 数量与 `--batch-size` 不符 | 参数加在了 `sh run_*.sh` 之前，没进 `"$@"` |

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

`aenv build` pulls the image and converts it to overlaybd **on the server**; no local docker is needed. Aliases cannot be rebound — run `aenv template delete twinkle-code` before rebuilding.

Behind the Great Firewall, `FROM python:3.11-slim` cannot be resolved: the server only searches `docker.io` / `ghcr.io`, which time out and return 401 respectively. Point `FROM` at a working mirror (**keep `library/`** — the real Hub path for official images is `library/python`):

```dockerfile
FROM docker.1panel.live/library/python:3.11-slim
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
```

Probe from the environment host first; 200 and 401 both mean reachable, a timeout means try the next one:

```bash
for R in docker.1panel.live docker.m.daocloud.io docker.1ms.run; do
    printf "%-24s " "$R"
    timeout 15 curl -s -o /dev/null -w '%{http_code}\n' \
        "https://$R/v2/library/python/manifests/3.11-slim"
done
```

These are third-party community mirrors and may go down or throttle; mirror the image into your own registry for anything long-lived. `PIP_INDEX_URL` needs the same treatment — the sandbox cannot reach pypi.org either.

Training host:

```bash
pip install e2b
AENV_API_URL=http://<env-host-ip>:8000 sh run_agentenv.sh
```

Setting `E2B_API_KEY` / `E2B_API_URL` alone has no effect — use `AENV_API_URL`, otherwise the script silently falls back to `http://127.0.0.1:8000`.

When the server is bound to `127.0.0.1` and training runs elsewhere, use an SSH tunnel (`ssh -N` not returning is normal — that blocking state *is* the tunnel; `-f` backgrounds it):

```bash
ssh -f -N -L 8000:127.0.0.1:8000 root@<env-host-ip>
```

Verify a sandbox boots before launching training — it takes seconds instead of waiting for vLLM to load:

```bash
python -c "
from twinkle_agentic.envs import AgentEnv
e = AgentEnv(template='twinkle-code', api_url='http://127.0.0.1:8000')
e.reset(); print('sandbox ok')
print(e.run_command({'command': 'python -c \"import numpy, sympy; print(numpy.__version__)\"'}))
"
```

## Arguments

Command-line arguments are forwarded to `train.py` and override the defaults in `common_args.sh`:

```bash
sh run_openenv.sh --max-steps 500 --batch-size 8
```

Smoke test. `batch-size × num-generations` **must be ≥ `--model-gpus`**, otherwise every batch is dropped by the length filter with only a warning — harder to diagnose than a crash. Arguments only take effect **after** the script name:

```bash
sh run_openenv.sh --batch-size 2 --num-generations 4 --max-steps 2
```

## Troubleshooting

| Symptom | Cause |
|---|---|
| `Invalid API key format: expected "e2b_"` | Client-side check in the e2b SDK. `AgentEnv` already defaults `E2B_VALIDATE_API_KEY=false`; still failing means it was explicitly set back to `true` |
| `400: template twinkle-code not found` | Template not built, or the build failed before reaching ready. Check `aenv template list` |
| `alias 'twinkle-code' already points to ...` | Aliases cannot be rebound; `aenv template delete twinkle-code` first |
| `dial tcp ...: i/o timeout` on `registry-1.docker.io` | Docker Hub unreachable; point `FROM` at a mirror |
| `overlaybd-commit ... No such file or directory` | Incomplete overlaybd install. Check that `/var/lib/aenv/deps/overlaybd/bin/` holds create/apply/commit/resize and that `/etc/overlaybd/overlaybd.json` exists |
| `open /root/.regctl/config.json: permission denied` | The server dropped to `aenv` but reads root's home. Only a WARN for public images; give it a writable HOME before configuring private registry credentials (`install -d -o aenv -g aenv /var/lib/aenv/home`, then set `HOME=`) |
| env count does not match `--batch-size` | Arguments were placed before `sh run_*.sh` and never reached `"$@"` |

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
