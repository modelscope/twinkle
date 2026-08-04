# Agentic RL 部署与训练

Agentic RL 需要两个部分：供模型执行动作的执行环境，以及消费这些轨迹的 GRPO 训练循环。本文依次介绍两者的部署与对接。

两者相互正交：更换执行后端仅需修改 `CODE_RL_BACKEND` 环境变量，跨机部署仅需修改一个 URL，训练代码无需变动。因此建议先在单机验证完整链路，再扩展到多机。

全文以 [`cookbook/rl/env/`](https://github.com/modelscope/twinkle/tree/main/cookbook/rl/env) 为例：MBPP 数据集上的多轮代码生成任务，同一份 `train.py` 支持两个后端。任务、工具、奖励公式完全一致，仅代码执行位置不同，因此实验曲线的差异可归因于执行环境本身。

## 执行后端的选择

三个选项的差异主要在于隔离级别及其内存代价。

| | OpenEnv 嵌入式 | OpenEnv server | AgentENV |
|---|---|---|---|
| 环境跑在哪 | 训练进程内 | 独立 HTTP/WebSocket 服务 | 独立 Firecracker microVM |
| 隔离 | 无 | 进程 / 容器 | microVM（KVM） |
| 执行器 | 看环境包 | smolagents AST 解释器 | 真 CPython |
| 单环境内存 | KB | KB | **~1GB** |
| 文件 / pip / 子进程 | 看环境包 | 不支持 | 支持，销毁即清零 |
| 部署成本 | 零 | 一条 `uvicorn` | 需部署控制面 + `/dev/kvm` |

选择依据：

**纯计算环境**（棋类、文本游戏、可在训练进程内安全求值的判分逻辑）——用嵌入式 OpenEnv，零部署。

**需要执行代码但标准库已能满足**——用 OpenEnv server 模式。它比 microVM 轻三个数量级，为执行几行 `sorted()` 而启动虚拟机并不划算。

**需要 `unittest` + `@patch`，或模型需要写文件、装包、开子进程**——只能用 AgentENV。此处有一个容易被忽略的失败模式：OpenEnv 的 `coding_env` 底层是 smolagents 的 `LocalPythonExecutor`，**一个 AST 解释器，而非操作系统级沙箱**。它对 `decorator_list` 没有任何处理，**装饰器会被静默忽略**——`@patch` 不生效、测试不报错，reward 产出一个形式正常的错误数值。这类隐形错误比崩溃难以定位。它适用于约束「仅允许 import 白名单」，不适用于执行对抗性代码。

后端能力的完整对比见[执行环境](../组件/Agentic/Envs.md)。

---

# 第一部分：执行环境部署

## OpenEnv

### 嵌入式：无需部署

环境在训练进程里直接实例化，没有网络跳数：

```bash
cd cookbook/rl/multi_turn && python multi_turn_grpo.py
```

环境实例较多、CPU 成为瓶颈时，用 `EnvPool` 将其迁移到独立的 CPU `DeviceGroup`，不占用 GPU 进程的内存与 GIL：

```bash
ENV_REMOTE=1 ENV_NUM_WORKERS=8 ENV_POOL_SIZE=64 python multi_turn_grpo.py
```

`ENV_REMOTE=1` 才会真正分片，不设则在 driver 本地运行（零 RPC）。`ENV_POOL_SIZE=0` 表示自动取轨迹数。

`EnvPool` 仅对嵌入式 OpenEnv 有意义。**不要将 `OpenEnvClient` 或 `AgentEnv` 放入 `EnvPool`**——它们的 session / sandbox 生命周期在服务端，Ray 再分片一次不带来收益，只增加一跳 RPC。

### server 模式

环境机不需要 GPU、KVM 或 Docker：

```bash
cd cookbook/rl/env
sh openenv_server/install.sh    # pip install openenv + 从源码装 coding_env
sh openenv_server/serve.sh      # 4 workers x 64 sessions = 256 并发
```

`coding_env` 是 OpenEnv 仓库内的子包，未发布到 PyPI，只能从源码安装——`server_app.py` 需从其中 import `PythonCodeActEnv` 与 `PyExecutor`，smolagents 也由它引入。

训练机仅需 `pip install openenv`，客户端类由它提供，无需安装 `coding_env`。

### 不使用上游 server 的原因

`serve.sh` 启动的是[本目录下的 `server_app.py`](https://github.com/modelscope/twinkle/blob/main/cookbook/rl/env/openenv_server/server_app.py)，而非上游的 `coding_env.server.app`。直接沿用 upstream 会遇到三个问题：

```python
class ConcurrentCodeEnv(PythonCodeActEnv):
    # 上游默认 False，create_app(max_concurrent_envs > 1) 会直接抛
    # ConcurrencyConfigurationError，服务端被锁成单 session
    SUPPORTS_CONCURRENT_SESSIONS = True

    def reset(self, **kwargs):
        # 父类 reset() 用上游默认值重建 executor 和 transform，之后必须重配
        observation = super().reset()
        self._configure()
        return observation

    def _configure(self) -> None:
        # 上游只授权 import json，MBPP 里常见的 math / collections 会直接失败
        self._executor = PyExecutor(additional_imports=list(ALLOWED_IMPORTS))
        # 上游的 create_safe_coding_transform() 用代码风格启发式覆盖
        # observation.reward（见到 open( / import os 罚 -1.0，短代码奖 +0.1）。
        # 本任务的奖励来自单元测试，风格分挤在同一个通道上只会是噪声
        self.transform = None
```

开启 `SUPPORTS_CONCURRENT_SESSIONS` 是安全的：`create_app` 接收的是**类**（作为 factory 使用），每个 WebSocket 连接新建一个实例，executor 与 state 均为实例私有。

另需说明，HTTP 的 `/step` `/reset` 端点不适用于多轮场景：OpenEnv 的这两个端点每次请求新建一个 env，返回后立即 `close()`，状态不予保留。多轮 episode 必须使用 WebSocket，`OpenEnvClient` 已实现该通路。

### 容量

```
并发 session 上限 = WORKERS x MAX_CONCURRENT_ENVS   # 默认 4 x 64 = 256
必须 ≥ BATCH_SIZE x NUM_GENERATIONS                 # 默认 4 x 8 = 32
```

超出容量的连接会被服务端拒绝，表现为一批 rollout 中部分轨迹的观测全部为 `Error:`。调大 `--batch-size` / `--num-generations` 时须同步扩大 `WORKERS` 或 `MAX_CONCURRENT_ENVS`。

## AgentENV

[AgentENV](https://github.com/kvcache-ai/AgentENV) 用 Firecracker microVM 提供真实的操作系统语义，适合 tool-integrated reasoning 和 SWE 类任务。

### 主机条件核查

以下三项须全部满足：

```bash
uname -r                                     # >= 6.8
ls -l /dev/kvm                               # 必须存在且可读写
modinfo ublk_drv >/dev/null && echo ublk-ok  # 需要 ublk 内核模块
```

裸金属通常直接满足。云上 VM 与 GPU 实例需云厂商开启嵌套虚拟化，多数默认关闭。容器 / K8s Pod / DSW 这类托管环境取决于**宿主机**的内核与 `/dev/kvm`，还需要 privileged 与挂载 `/dev`——宿主机不满足时，容器内没有任何软件手段能绕过。

条件不满足时无需修改代码：在一台满足条件的机器上单独部署 AgentENV，训练侧将 `AENV_API_URL` 指向该地址即可。Firecracker **不支持 GPU 直通**，沙箱内无法运行 GPU 任务，因此环境机不需要显卡。

### 安装服务端

```bash
cd cookbook/rl/env
sh agentenv_server/install.sh
```

该脚本依次完成四项工作：安装 server 与 `aenv` CLI、通过 `server --setup-host` 预置主机（kvm 组、ublk 模块、udev 规则、sysctl）、准备运行时目录、构建沙箱模板。

其底层调用 AgentENV 官方的 `install.sh`，后者创建专用的非 root `aenv` 账户（仅授予 `CAP_NET_ADMIN`/`CAP_SYS_ADMIN` 与 kvm 组），下载 Firecracker、guest kernel 等运行时资源，并注册 systemd 服务。数据目录默认为 `/var/lib/aenv`，镜像层与快照均存放于此，建议指向大容量磁盘：

```bash
curl -fsSL https://raw.githubusercontent.com/kvcache-ai/AgentENV/main/scripts/install.sh \
  | sudo AENV_HOME_PATH=/data/aenv bash
```

仅需客户端时（在训练机上创建模板，或采用 Docker 部署）：

```bash
curl -fsSL https://raw.githubusercontent.com/kvcache-ai/AgentENV/main/scripts/install-cli.sh | sudo bash
```

### 构建沙箱模板

模板相当于沙箱的出厂镜像：依赖预装完毕并固化为快照，沙箱才能达到约 50ms 的启动速度。**依赖必须预置在模板内**，不应在训练时执行 `pip install`——每条轨迹重复安装数十秒的开销不可接受。

`install.sh` 已完成模板构建。手动重建的方式如下：

```bash
sh agentenv_server/install.sh --rebuild    # 先删旧模板
```

也可直接使用 CLI：

```bash
aenv auth
# AENV server URL [http://localhost:8000]: http://127.0.0.1:8000
# API key: dummy          # 本地部署下任意非空字符串均可

aenv build agentenv_server/Dockerfile -t twinkle-code --cpu-count 1 --memory-mb 1024
aenv template watch <template-id>      # 等 ready
```

`aenv build` 支持 `FROM / RUN / ENV / WORKDIR / USER`（`ENTRYPOINT` 转为启动命令，`EXPOSE / VOLUME / LABEL` 忽略）。拉镜像与转 overlaybd 均在**服务端**完成，本机无需 docker。仅当 Dockerfile 变更时才需重建。直接使用现成镜像而不做加工：`aenv pull ubuntu:22.04 --name ubuntu`。

别名不可改绑。报 `alias 'xxx' already points to ...` 时需先执行 `aenv template delete xxx`，`--rebuild` 即封装了这一步。

复杂工具的实现也建议预置在模板内，工具 handler 仅保留一行调用：

```dockerfile
COPY tools/search.py /opt/tools/search.py
```

如此工具实现具备版本管理、随快照分发、运行时零开销。

### 启动服务

```bash
sh agentenv_server/serve.sh              # 前台，绑 127.0.0.1:8000
NOHUP=1 sh agentenv_server/serve.sh      # 后台，日志到 /tmp/aenv-server.log
API_ADDR=0.0.0.0:8000 sh agentenv_server/serve.sh   # 对外，阅读下方安全说明后使用
```

脚本会先停止运行中的实例再启动，因此重启无需手动 kill。

server 自身的默认监听为 `0.0.0.0:8000`，脚本已将其收窄至回环。**AgentENV 不提供任何认证**，端口可达即等同于任意代码执行可达，绑定 `0.0.0.0` 时必须配置安全组白名单。

验证：

```bash
curl -i http://127.0.0.1:8000/health     # 期望 204
```

单机场景无需 AgentENV 的 gateway / scheduler（二者面向多节点部署），直连 server 的 `:8000` 即可。

### 内存预算

内存是 AgentENV 最主要的容量约束：

```
并发沙箱数 = BATCH_SIZE x NUM_GENERATIONS
需要内存   = 并发沙箱数 x 模板 --memory-mb + 8GB（AgentENV 自身 + 系统）
```

按 `--memory-mb 1024` 计算，默认 32 并发需 40GB；验证链路阶段采用 `2 x 4 = 8` 并发仅需 16GB，无需先行采购大内存机器。

内存不足时的调整顺序：降低模板 `--memory-mb`（多数任务 512MB 即可满足）→ 减小 `batch_size` → 依靠自动休眠（空闲沙箱 pause 后内存归还宿主机）。

CPU 争抢同样需要考虑：沙箱与 dataloader/tokenizer 竞争 CPU 核，`ENV_CONCURRENCY` 不应超过空闲核数。

## 网络与安全

twinkle 支持两种连接方式：同机或同一内网/VPC 内使用 **HTTP 直连**，跨网络使用 **SSH 端口转发**。

框架不引入 VPN / NAT 穿透类组件——这些属于网络基础设施层面，与训练代码无关。twinkle 一侧可见的仅是一个 `http://host:port`。

### HTTP 直连

企业环境下 GPU 机与 CPU 机通常已处于同一 VPC / IDC / K8s 集群，无需额外网络组件。`openenv_server/serve.sh` 的 `HOST` 默认为 `0.0.0.0`（此时脚本会打印无认证警告），应显式绑定**内网网卡**：

```bash
HOST=10.0.1.20 sh openenv_server/serve.sh              # OpenEnv
API_ADDR=10.0.1.20:8000 sh agentenv_server/serve.sh    # AgentENV
```

训练侧仅需修改一个环境变量：

```bash
OPENENV_BASE_URL=http://10.0.1.20:8000 sh run_openenv.sh
AENV_API_URL=http://10.0.1.20:8000     sh run_agentenv.sh
```

安全组入方向仅放行**训练机的 IP/32 或其安全组 ID**，端口仅开放 8000，不得使用 `0.0.0.0/0`。用 `ss -tlnp | grep 8000` 确认绑定生效——输出必须为 `10.0.1.20:8000` 而非 `0.0.0.0:8000`。

环境服务需配置保活，[`deploy/openenv-server.service`](https://github.com/modelscope/twinkle/blob/main/cookbook/rl/deploy/openenv-server.service) 可直接使用。服务一旦中断，整批 rollout 随之作废。

### SSH 端口转发

跨网络时，SSH 在此处的作用不仅是建立连接，更是**为零认证的服务补上一层认证**——复用已有的 SSH 密钥与登录审计，且它通常是企业已批准的运维通道。

端口转发工作在 TCP 层，对 WebSocket 完全透明。已实测验证：握手、跨消息的 session 状态、以及 8 个并发 session 复用一条连接，均工作正常。

```bash
# 环境侧：只绑回环，网络上彻底不可达
HOST=127.0.0.1 sh openenv_server/serve.sh

# 训练侧
ssh -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
    -L 8000:127.0.0.1:8000 user@env-host &
OPENENV_BASE_URL=http://127.0.0.1:8000 sh run_openenv.sh
```

`ssh -N` 不返回属于正常现象，该阻塞状态即表示隧道正在工作；需转入后台则加 `-f`。

两个地址容易混淆：`OPENENV_BASE_URL` 填写的是**本地入口** `127.0.0.1:8000`，`-L` 中的 `127.0.0.1:8000` 则是**在环境机视角**下解析的目标地址。AgentENV 同理，对应修改 `AENV_API_URL`。

长时间任务必须配置保活：转发一旦中断，整批 rollout 全部作废，因此不应运行在交互式 shell 中：

```bash
autossh -M 0 -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -L 8000:127.0.0.1:8000 user@env-host
```

所有流量复用一条 TCP 连接。每轮交互仅传输几 KB 代码与输出，实测 8 并发无压力；上百并发时可开多条转发并映射到不同本地端口分流。

### 沙箱出口

限制沙箱内代码的可访问范围，防止模型生成的代码扫描内网（SSRF）或滥用带宽。AgentENV 在 `config/default.toml` 中提供节点级强制策略，默认值已较为合理：

```toml
[network.egress]
# 沙箱自己的 egress policy 覆盖不了这些
always_denied_cidrs = [
  "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8",
  "169.254.0.0/16",              # ← 关键：挡住云元数据服务 169.254.169.254
  "172.16.0.0/12", "192.168.0.0/16",
]
```

`169.254.0.0/16` 这一条最为关键：封禁云厂商元数据服务，防止沙箱内的代码窃取 IAM 临时凭证进而获得整个云账号的权限。标准库算法题无需出网，直接使用默认值即可。

需要放宽时按沙箱粒度传入策略：

```python
# base_policy: Default | Allow | Deny
{'base_policy': 'Deny',
 'egress': {'allowed_domains': ['pypi.org', 'files.pythonhosted.org'],
            'denied_cidrs': ['10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16', '169.254.0.0/16']}}
```

最后需强调一点：`aenv auth` 使用的 API key **不具备认证作用**，本地部署下填写任意非空字符串即可通过，也不存在租户隔离——获得任意 sandbox-id 即可读写、销毁对应沙箱，`GET /sandboxes` 会列出全部沙箱。对外提供服务时，必须自建认证/租户/配额层，将 AgentENV 限定在私网内，仅对外暴露自建的上层 API。


---

# 第二部分：训练对接

本部分继续以 `cookbook/rl/env/` 为例。选择代码生成作为任务有两个原因：其奖励可由单元测试客观计算，无需裁判模型；且多轮交互具备内在语义——编写、试跑、修正、提交。

## 模型可见的动作空间

仅对模型暴露两个工具（定义于 `_openenv.py` / `_agentenv.py`）：`run_python(code)` 在环境中执行代码，`submit_solution(code)` 提交最终答案。

第二个工具值得单独说明，它**不发送给服务端**：

```python
def _submit_solution(env, arguments: Dict[str, Any]) -> str:
    code = (arguments.get('code') or '').strip()
    if not code:
        return "Error: 'code' argument is required."
    env.submitted_code = code
    return 'Solution submitted.'
```

该工具用 `register_tool` 在客户端本地处理，仅将源码记录在 env 上供训练循环打分。这是一个通用模式：**模型的动作与训练所需的记账属于两个层面**，后者应置于本地 handler，不应污染环境协议。

System prompt 必须准确描述后端语义，否则模型会基于错误的心智模型编写代码。两个后端在这一点上正好**相反**：

- OpenEnv session：`The interpreter keeps its state between calls`——本轮定义的函数下轮可直接调用。同时列出可用模块白名单，并声明无文件/网络访问。
- AgentENV：`Each call runs in a FRESH process, so every snippet must be self-contained`。

AgentENV 侧另通过 `include_default_tools=False` 关闭了内置的 `run_command` / `read_file` / `write_file`，使动作空间与任务精确对齐，以保证奖励归因的清晰。

工具集是客户端概念。AgentENV 服务端本身**不定义工具**，仅提供能力原语：任意命令执行、文件读写、端口代理。除 `register_tool(schema, handler)` 之外，还提供命令模板型的 `register_command_tool`：

```python
env.register_command_tool(
    {'type': 'function', 'function': {
        'name': 'run_tests',
        'description': '运行任务测试集。',
        'parameters': {'type': 'object',
                       'properties': {'test_file': {'type': 'string'}},
                       'required': ['test_file']}}},
    'cd /workspace && pytest {test_file} -x -q')     # 用工具参数格式化
```

handler 的签名为 `handler(env, arguments) -> str`，返回值即 observation。内部可使用 `env.run_command(...)` 与 `env.sandbox`（原始 E2B 句柄，可访问 PTY、文件 watch 等能力）。同名注册会覆盖内置工具。

## 轨迹与 env 的一一对应

```python
def prepare_trajectories(samples, pool):
    envs = [backend.make_env() for _ in samples]
    # reset() 阻塞在网络上（WebSocket 握手，或启动一个沙箱），必须并发，
    # 否则一个 batch 32 条会串行等
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

**每条轨迹对应一个 `ToolManager`。** 它持有具体的 env 实例，共享将导致所有轨迹的工具调用打到同一个 session。

**`envs` 必须在 `finally` 中关闭。** 它们占用服务端容量，泄漏会使后续 step 因容量不足而失败：

```python
try:
    all_trajectories = rollout(expand_prompts, tool_manager=tool_managers)
    total_rewards, pass_rates = extract_rewards(envs, expanded, env_pool)
finally:
    close_envs(envs, env_pool)
```

还有一项容易忽略的行为：`MultiTurnRollout` 的终止条件是**模型不再发出工具调用**（或达到 `MAX_TURNS` / 长度上限），它**不读取** `EnvTool.done`。因此 system prompt 中需明确要求「提交后不要再调用工具」，否则回合会一直运行至 `MAX_TURNS`。

## 奖励：基于单元测试而非裁判模型

MBPP 每条样本自带若干 `assert`。rollout 结束后回放隐藏测试，两个后端的**脚本形态由后端能力决定，奖励公式完全相同**。

AgentENV 侧是真 CPython，直接生成一个普通脚本，每条断言用 `try` 包住，最后打印 `TESTS_PASSED n total`。

OpenEnv 侧在**同一个 session** 中逐条执行，并将 `assert X` 改写为 `print(X)`。此举的目的是可诊断性：失败的 `assert` 抛出的异常与解法内部崩溃无法区分，改为 print 即可将「断言求值为 False」与「代码崩溃」区分开，且单条测试抛出异常不影响其余测试继续执行。该执行器本身支持 `assert` / `try`，此处并非能力绕行。

OpenEnv 侧使用 `env.execute()` 而非 `env.step()`：`execute()` 返回服务端**原始** `StepResult`，可读取 `exit_code` 等结构化字段；`step()` 返回的是渲染给模型的文本，且会计入 episode。打分逻辑不应出现在模型的对话中。

奖励整形：

```python
rate = passed / total if total else 0.0
if total and passed == total:
    return 1.0, rate                    # 全对
if getattr(env, 'submitted_code', None):
    return 0.1 + 0.4 * rate, rate       # 提交了，部分对 → 0.1 ~ 0.5
return 0.0, rate                        # 没提交
```

形状上保证「全对 > 部分对 > 提交但全错 > 未提交」，且全对的 1.0 明显高于部分对的上限 0.5，否则模型会学会提交一个「刚好能过第一个测试」的假实现。

**提交动作本身给 0.1 底分**是有意设计：训练早期所有轨迹奖励均为 0，GRPO 的组内优势也全为 0，无法产生有效学习信号。该底分提供了最初的梯度信号。

沙箱本身不产生 reward（`AgentEnv.evaluate` 默认返回 0），打分统一在训练循环中完成。如需在沙箱内判分，则让工具输出结构化结果，再在 driver 侧解析。

## GRPO 的组内布局

```python
batch = [dataset[(sample_cursor + i) % len(dataset)] for i in range(BATCH_SIZE)]
expanded = [s for s in batch for _ in range(NUM_GENERATIONS)]   # 同题连续 N 份
...
advantages = advantage_fn(total_rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()
```

注意 `expanded` 的写法：`for s in batch for _ in range(N)` 使同一道题的 N 条 rollout 在列表中**连续排列**，`GRPOAdvantage` 按该布局切分组。写成 `for _ in range(N) for s in batch` 会使分组完全错位，训练过程表面上正常，实际上无法学到任何有效信号。

## 启动训练

建议先以小并发验证全链路，以节约时间与算力成本：

```bash
cd cookbook/rl/env

# OpenEnv
sh run_openenv.sh --batch-size 2 --num-generations 4 --max-steps 2

# AgentENV
sh run_agentenv.sh --batch-size 2 --num-generations 4 --max-steps 2
```

但并发不宜过低：轨迹数（`batch-size x num-generations`）必须 ≥ `--model-gpus`（默认 4）。否则长度过滤后剩余不足，整批被跳过，日志仅反复输出 `skipping this batch`——这种表现形似卡住而非报错，比明确的崩溃更难定位。上述 `2 x 4 = 8` 预留了 2 倍余量。

参数需置于脚本名**之后**才能生效（脚本内为 `python train.py $TRAIN_ARGS "$@"`，`"$@"` 在后方才能覆盖）。

扩展至正式训练：

```bash
OPENENV_BASE_URL=http://10.0.0.5:8000 MAX_TURNS=8 ENV_CONCURRENCY=32 \
sh run_openenv.sh --batch-size 8 --num-generations 16 --max-steps 500
```

可覆盖的环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `MAX_TURNS` | `6` | 单回合最大工具调用轮数 |
| `ENV_CONCURRENCY` | `16` | driver 并发创建/销毁/打分的线程数 |
| `OPENENV_BASE_URL` | `http://127.0.0.1:8000` | OpenEnv 服务地址（也可以是负载均衡器） |
| `OPENENV_ENV_NAME` | `coding_env` | 环境包名，决定客户端 + Action 类 |
| `OPENENV_MESSAGE_TIMEOUT_S` | `120` | 单条消息超时 |
| `AENV_API_URL` | `http://127.0.0.1:8000` | AgentENV 地址 |
| `AENV_TEMPLATE` | `twinkle-code` | 模板名 |
| `SANDBOX_TIMEOUT` | `600` | 沙箱空闲超时（秒） |
| `AENV_COMMAND_TIMEOUT` | `60` | 沙箱内单条命令超时（秒） |

训练超参通过 CLI 参数传入，默认值定义于各自的 `run_*.sh` 中。两个后端的 `TRAIN_ARGS` 需保持一致，否则奖励的变化无法归因于执行后端而非超参差异。

`AENV_API_URL` 不能用 `E2B_API_URL` 替代——只设后者会静默回落到默认值。

## 监控指标

日志每步输出一行：

```
[Step 0] {'train/code_acc': 0.031, 'train/test_pass_rate': 0.208, 'train/avg_reward': 0.145, ...}
```

任务指标：

| 指标 | 含义与解读 |
|---|---|
| `train/code_acc` | pass@1，全部隐藏测试通过的比例。这是真正需要提升的指标 |
| `train/test_pass_rate` | 平均单测通过率。比 `code_acc` 平滑，早期优先观察其变化 |
| `train/avg_reward` | 整形后的奖励均值。它上涨而 `code_acc` 不涨，说明模型在套取 0.1 的提交底分 |
| `train/avg_turns` | 平均轮数。贴着 `MAX_TURNS` 说明模型经常不提交就用完轮数 |
| `train/max_turns` / `train/min_turns` | 批内轮数分布。`max_turns` 顶到上限说明有轨迹被截断；`min_turns` 接近 1 说明常有轨迹第一轮就提交或放弃 |

策略健康度（由 `model.add_metric('GRPOMetric', ...)` 注册）：

| 指标 | 含义与解读 |
|---|---|
| `train/approx_kl` | 新旧策略的 KL（Schulman K3 估计）。突然飙升是策略崩塌的前兆 |
| `train/clip_ratio` | PPO 裁剪触发比例（另有 `_low` / `_high` 分向）。长期偏高说明单步更新太激进 |
| `train/token_kl_max` / `train/token_ratio_max` | 单 token 的 KL / 概率比极值，用来定位崩塌 |
| `train/policy_confidence` | `exp(mean_logp)`，策略平均置信度 |

`GRPOMetric` 的 `epsilon` 需与 `set_loss('GRPOLoss', epsilon=...)` 保持一致，否则 `clip_ratio` 统计的裁剪阈值与实际生效的不是同一个。`train/entropy` 需 `GRPOLoss(entropy_coef > 0)` 才会出现，但该选项会向 loss 中加入 entropy bonus，**从而改变训练目标**，不应仅为观察指标而开启。

## 迁移到自定义任务

复用该骨架时通常仅需修改四处，训练循环、GRPO 配置、权重同步、指标均无需变动：

1. **数据集**：将 `load_mbpp()` 替换为自定义加载函数，产出 `{'prompt', ...打分所需字段}`。
2. **工具**：`TOOL_SCHEMA` 与 handler。服务端能力走默认 action 通路，客户端记账用 `register_tool`。
3. **奖励**：`extract_rewards()` 中的 `score()`。优先选择可客观计算的信号（单测、精确匹配、可执行校验），无可用信号时再考虑裁判模型。
4. **System prompt**：必须准确描述后端语义（状态是否跨轮保留、有哪些模块和权限、轮数预算）。

新增一个后端：编写 `_xxx.py`（提供 `NAME`、`SYSTEM_PROMPT`、`TOOL_SCHEMA`、`make_env()`、`run_tests()`、`describe()`）与 `run_xxx.sh`，并在 `train.py` 的 `BACKENDS` 中加入名称。

---

# 故障排查

## 上线前检查清单

- [ ] 服务**未**绑在 `0.0.0.0`（`ss -tlnp | grep 8000` 确认）
- [ ] 内网直连：安全组入方向仅放行训练机，而非 `0.0.0.0/0`
- [ ] 跨网络：环境侧绑 `127.0.0.1`，仅经 SSH 端口转发访问
- [ ] 容量 ≥ `BATCH_SIZE x NUM_GENERATIONS`（AgentENV 需另核内存）
- [ ] 环境服务已配保活（systemd），SSH 转发已配保活（autossh），而非运行在交互式 shell 上
- [ ] 已用 `--batch-size 2 --num-generations 4 --max-steps 2` 验证过全链路
- [ ] 未为了「让沙箱联通」而放宽 AgentENV 的 `always_denied_cidrs`

## 环境侧

| 现象 | 原因与处理 |
|---|---|
| 一批轨迹里部分观测全是 `Error:` | 服务端容量不足。核对 `WORKERS x MAX_CONCURRENT_ENVS ≥ BATCH_SIZE x NUM_GENERATIONS` |
| 模型报 `import math` 不被允许 | `ALLOWED_IMPORTS` 白名单未生效；确认 `reset()` 之后重新调用了 `_configure()` |
| `ConcurrencyConfigurationError` | 启动的是上游 `coding_env.server.app`（单 session）。改用 `openenv_server/server_app.py` |
| 奖励里混进 ±0.1 / -1.0 的异常值 | 环境的 reward transform 未关闭，检查 `self.transform = None` |
| 超时 / 消息等待报错 | 调大 `OPENENV_MESSAGE_TIMEOUT_S`。执行器自身另有三重上限：单次执行 30s wall-clock、1000 万次操作、100 万次 while 迭代，因此该值需保留在 30s 以上才有意义 |
| `Address already in use` | 端口被占。`ss -tlnp \| grep 8000` 查占用进程，或换用 `PORT=8001` |

## AgentENV 侧

| 现象 | 原因与处理 |
|---|---|
| `/dev/kvm is not accessible` | 运行账户不在 kvm 组，或宿主机无 KVM。执行 `sudo server --setup-host --runtime-user aenv --runtime-group aenv` 后重启 |
| `ublk_drv is not loaded` | 执行 `sudo modprobe ublk_drv`；内核 < 6.8 需升级 |
| `ImportError: AgentEnv requires the E2B SDK` | `pip install e2b` |
| `Invalid API key format: expected "e2b_"` | e2b SDK 的客户端本地校验。`AgentEnv` 已默认设置 `E2B_VALIDATE_API_KEY=false`，仍报错说明被显式覆盖为 `true` |
| `400: template xxx not found` | 模板未创建，或 build 未达到 ready。用 `aenv template list` 查看状态 |
| `alias 'xxx' already points to ...` | 别名不可改绑，需先执行 `aenv template delete xxx` |
| 沙箱内 `pip install` 失败 | 出口网络被策略拦截，或当前网络不可达 PyPI；改为在模板里预装 |
| 轨迹中途报沙箱不可用 | `SANDBOX_TIMEOUT` 小于轨迹耗时，被自动 pause。需调大 |
| batch 启动很慢 | 调大 `ENV_CONCURRENCY`；确认依赖已预置在模板内而非运行时安装 |
| 每次重启都重新下载依赖 | `AENV_HOME_PATH` 未设置，落到了 `/tmp/aenv-test-<uid>/`（`run-with-capabilities.sh` 的测试默认值），会被 `/tmp` 清理。`serve.sh` 已固定为 `/var/lib/aenv` |
| `load config ... Permission denied` | 源码编译的二进制将编译时路径固化为默认配置位置，降权为 `aenv` 后读不到 `/root`。用 `AENV_CONFIG_PATH` 指向 `aenv` 能读的副本，`serve.sh` 已处理 |

## 训练侧

| 现象 | 原因与处理 |
|---|---|
| 有效轨迹数不足被跳过 | 多数轨迹超长被过滤。调小 `MAX_TURNS` / `--max-tokens`，或收敛工具输出（AgentENV 侧观测默认截断到 32K 字符）；也确认轨迹数 ≥ `--model-gpus` |
| 奖励长期为 0 | 先单独运行 `run_tests` 给一个已知正确解打分，确认打分链路本身可用，再排查模型 |
| env 数量与 `--batch-size` 不符 | 参数加在了 `sh run_*.sh` 之前，未进入 `"$@"` |

---

# 附录：受限网络下的部署变通

本节仅在**默认路径不可用**时需要参考，适用于无法直达 Docker Hub、GitHub、PyPI 等公共源的环境（企业隔离网、内网集群、或存在出网限制的区域）。若 `install.sh` 已执行成功、模板已构建完毕，可跳过本节。

下文的镜像地址与包源均为示例，需替换为当前环境实际可达的内部镜像。

## 镜像源

默认 Dockerfile 有两处依赖外部网络：`FROM python:3.11-slim` 与 `RUN pip install`，两者需分别处理。

**基础镜像**：服务端仅按 `[image.resolver] search_registries` 搜索，默认为 `docker.io` / `ghcr.io`。两者不可达时典型报错为 `dial tcp ...: i/o timeout`（超时）或 401（需认证）。写全限定名可绕开搜索列表，**`library/` 不可省略**（官方镜像在 Hub 上的真实路径为 `library/python`）：

```bash
BASE_IMAGE=<your-registry>/library/python:3.11-slim sh agentenv_server/install.sh
```

`BASE_IMAGE` 会传给 `aenv build --image`，覆盖 Dockerfile 中的 `FROM`，无需修改文件。

建议先探测候选镜像源的可达性，避免无效的 build。200 与 401 均计为可达（401 表示需先获取 token，属于正常），超时则说明不可用：

```bash
for R in <registry-1> <registry-2>; do
    printf "%-24s " "$R"
    timeout 15 curl -s -o /dev/null -w '%{http_code}\n' \
        "https://$R/v2/library/python/manifests/3.11-slim"
done
```

仅返回状态码不足以证明可用：部分代理能完成 TLS 握手却取不到完整 manifest。如需确认，可对比响应体大小——真实的 multi-arch index 为数 KB，而错误页面通常不足 200 字节。

公共代理站点存在随时不可用或限速的可能，不宜用于长期训练。生产环境建议将镜像转存至自建的容器镜像仓库（Harbor、或云厂商提供的托管仓库），置于与环境机同一内网。

**pip**：若沙箱内无法访问 pypi.org，需在 Dockerfile 中设置 `ENV PIP_INDEX_URL`（默认 Dockerfile 未包含该行，需自行添加），且必须写在 `RUN pip install` **之前**。

构建失败时用 `aenv template watch <id>` 查看原因。两个常见原因：

- `all registry candidates failed during manifest fetch`——基础镜像无法拉取，按上述方式更换。
- `overlaybd-commit ... failed to perform commit(), 2: No such file or directory`——镜像已拉取但转换失败。先确认 `FROM` 是否指错（数十 GB、上百层的大镜像容易卡在此处），确认无误后再检查 overlaybd 是否安装完整：`/var/lib/aenv/deps/overlaybd/bin/` 应包含 create / apply / commit / resize 四个二进制，`/etc/overlaybd/overlaybd.json` 应存在。

日志中的 `open /root/.regctl/config.json: permission denied` 仅为 WARN（server 降权为 `aenv` 后去读取 root 家目录），拉取公开镜像不受影响。但配置私有 registry 凭据时，需先为其提供一个可写 HOME：

```bash
sudo install -d -o aenv -g aenv /var/lib/aenv/home
```

## install.sh 下载失败时：源码编译

`install.sh` 仅依赖两个外网端点：`api.github.com` 取 release 元数据，然后从 `github.com/.../releases/download/` 下载两个资产（`aenv-linux-x86_64` 9.3MB、`aenv-server-linux-x86_64.tar.gz` 68MB）。后者会 302 至 `release-assets.githubusercontent.com`，部分网络环境下易在传输中途被 RST（`curl: (56) Recv failure` 或 `(92)`）。

先尝试代理：`export https_proxy=...` 后重跑，`install.sh` 全程使用 curl，识别 env 代理。无代理时才需采用本节方案。

需先明确一点：**编译只能解决 Rust 二进制部分。** 预编译 tarball 中除 `server` 外还包含一个 `deps/`（firecracker、guest kernel、tools 驱动、overlaybd、regctl），这些不在编译产物中，需单独准备。这是本方案最易遭遇问题的环节。

### 前置

若无法直达 `static.rust-lang.org` / `crates.io`，需配置可用的 crates 镜像（下文以 `<rust-mirror>` / `<crates-mirror>` 占位，替换为当前环境可达的地址）。**必须先设置变量再安装 rustup**，顺序颠倒则需重新执行：

```bash
# 写进 profile：AgentENV 有 rust-toolchain.toml，首次 cargo build 会再拉一次工具链
cat >> ~/.bashrc <<'EOF'
export RUSTUP_DIST_SERVER=<rust-mirror>
export RUSTUP_UPDATE_ROOT=<rust-mirror>/rustup
EOF
. ~/.bashrc

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"

# 多数镜像仅支持 sparse 索引，需 cargo >= 1.68。用覆盖写：>> 跟两次会出现重复的
# [source.crates-io]，TOML 直接解析失败
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

sudo apt-get update      # upgrade 不刷索引；漏了会报 Unable to locate package

# 编译依赖（同 Dockerfile.agentenv 的 builder 阶段）。clang / libclang-dev 是
# uvm-ublk-daemon 必需：它用 bindgen 生成 ublk 内核绑定，缺了报 Unable to find libclang
sudo apt-get install -y build-essential pkg-config libssl-dev \
     clang libclang-dev libprotobuf-dev protobuf-compiler

# 运行依赖（同 runtime-base 阶段）。libaio1t64 是 Ubuntu 24.04 的包名，
# 22.04 / Debian 12 上叫 libaio1
sudo apt-get install -y ca-certificates curl dpkg e2fsprogs iproute2 iptables \
     jq libaio1t64 sudo umoci zstd
```

`protobuf-compiler` 必须使用系统包：`make ci-deps-protoc` 会从 GitHub Releases 下载 protoc，而那正是不可达的路径。Debian 13+ 上 `pkg-config` 已改名为 `pkgconf`。

镜像是否生效看 `cargo build` 的第一行，必须为 ``Updating `mirror` index``。仍为 `Updating crates.io index` 则说明 `config.toml` 未被读取——常见原因是 `CARGO_HOME` 指向其他位置，或写入文件时 `~/.cargo` 尚不存在。注意报错中的 `an/yh/anyhow` 是 sparse 路径，但 cargo >= 1.70 默认即使用 sparse，**不能以此作为镜像生效的依据**。

### 编译与安装

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

最后那个 `config.toml` 的位置尤为关键：**源码编译的二进制将编译时的仓库路径固化为默认配置位置**（`cfg.rs` 中的 `CARGO_MANIFEST_DIR`）。若在 `/root/AgentENV` 编译，它会查找 `/root/AgentENV/config/default.toml`，而降权为 `aenv` 后无权访问 `/root`（0700），因此报 `Permission denied`。所以需将配置副本放至 `aenv` 可读的位置，启动时用 `AENV_CONFIG_PATH` 指向该副本。`agentenv_server/serve.sh` 已处理该问题。

### 准备 deps

```bash
E="AENV_CONFIG_PATH=/var/lib/aenv/config/config.toml AENV_HOME_PATH=/var/lib/aenv"
sudo env $E /usr/local/bin/server --setup-only
```

下载失败时，需先确认文件的来源——以下五个 URL 即其需下载的全部内容（定义于 `config/deps_manifest.toml`）：

```
firecracker + cpu-template-helper  https://pub-4ee15c400f554ab7a9eac3f5bc8f53de.r2.dev/firecracker-1.15.1-patch-v1-x86_64.tgz
guest kernel                      https://pub-4ee15c400f554ab7a9eac3f5bc8f53de.r2.dev/vmlinux-6.1.175
regctl                            https://github.com/regclient/regclient/releases/download/v0.11.5/regctl-linux-amd64
overlaybd .deb                    https://github.com/containerd/overlaybd/releases/download/v1.0.18/overlaybd-1.0.18-20260710.cee2186.{target}.deb
tools.ext4                        ghcr.io/zlzgithub-0801/agentenv-tools:0.1.0   （OCI 镜像，需 regctl/docker 导出）
```

`{target}` 不要写死，`overlaybd.rs` 会读取 `/etc/os-release` 自动替换为 `ubuntu1.<version>.<arch>`。真正需要照抄的仅有 `1.0.18-20260710.cee2186` 这一段日期加 git hash。

在可联网的机器上将失败的资源下载完毕，然后**预放文件**——`download_file` 发现目标文件已存在且非空则跳过下载，目标路径即失败日志中的 `dest=`：

```bash
# 日志：downloading url="https://pub-...r2.dev/firecracker-1.15.1-patch-v1-x86_64.tgz"
#                    dest=/var/lib/aenv/deps/firecracker/1.15.1-patch-v1/firecracker-1.15.1-patch-v1-x86_64.tgz
# scp 到 dest= 路径（文件名一模一样），然后：
sudo chown -R aenv:aenv /var/lib/aenv/deps
sudo env $E /usr/local/bin/server --setup-only
```

该方法对五项依赖均适用，无需修改配置。overlaybd 没有本地路径开关，但同样适用此方法：将 `.deb` 放入 `/var/lib/aenv/deps/overlaybd/downloads/`（文件名按 URL 末段，`{target}` 已替换），`package_url` 就不会被访问。

也可修改 `/var/lib/aenv/config/config.toml` 指向存放文件的目录。注意 `[firecracker]`、`[kernel]`、`[tools]` **这三段原本已存在**，需将键加入段内，**不要追加同名段**——TOML 中重复的表会直接解析失败：

```toml
[firecracker]                                      # 已有段，boot_args 等原样保留
binary_path = "/opt/aenv-assets/firecracker"       # 解开 tgz 后的二进制，不是 tgz
[kernel]                                           # 已有段（空）
image_path = "/opt/aenv-assets/vmlinux.bin"
[tools]                                            # 已有段，control_plane_port 原样保留
version = "0.1.0"                                  # 与 drive_path 必须成对
drive_path = "/opt/aenv-assets/tools.ext4"
```

`/opt/aenv-assets` 需自行创建，并非现成目录。

### 启动服务

```bash
sudo env $E /usr/local/bin/server --setup-host --runtime-user aenv --runtime-group aenv
sudo chown -R aenv:aenv /var/lib/aenv
sh cookbook/rl/env/agentenv_server/serve.sh
```

`serve.sh` 中有几个不可省略的环境变量：`AENV_RUN_USER=aenv`（不设则走 `SUDO_USER` → 仓库 owner → `aenv` 三层 fallback，root 直接运行时结果不稳定）、`AENV_HOME_PATH=/var/lib/aenv`（不设则落到 `/tmp/aenv-test-<uid>/`，被清理后需重新下载数百 MB）、`AENV_CONFIG_PATH`（上述的编译路径问题）。

另有一点：`--setup-host` 将 `aenv` 加入 kvm 组的变更对已有会话不生效，需依靠 `run-with-capabilities.sh` 的 `--init-groups` 重新初始化。绕过它直接启动 server 会因无法获得 kvm 权限而失败。

> 源码编译这条路未做端到端实测，deps 的下载可达性因机器而异。

## Docker 部署

```bash
curl -fsSL https://raw.githubusercontent.com/kvcache-ai/AgentENV/main/scripts/docker-setup.sh | sudo bash
docker run -d --privileged -v /dev:/dev -p 8000:8000 ghcr.io/kvcache-ai/aenv-server:latest
```

这里的 Docker 仅作为**部署载体**，沙箱仍为 Firecracker microVM，同样需要宿主机的 KVM。`ghcr.io` 的镜像拉取同样经由 GitHub 分发，在受限网络下可能一样不可达。

---

## 相关文档

- 组件参考：[执行环境](../组件/Agentic/Envs.md)（`Env` 抽象、`EnvTool`、OpenEnv 两种模式、`EnvPool`）
- 多轮工具调用：[多轮工具调用](../组件/Agentic/Multi-Turn-Tool-Usage.md)
- 可运行示例：`cookbook/rl/env/`（代码任务，两个后端）、`cookbook/rl/multi_turn/`（嵌入式 OpenEnv）
- OpenEnv 上游仓库：<https://github.com/meta-pytorch/OpenEnv>
- AgentENV 官方文档：<https://kvcache-ai.github.io/AgentENV/>
