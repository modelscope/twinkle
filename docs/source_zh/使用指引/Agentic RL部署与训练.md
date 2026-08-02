# Agentic RL 部署与训练

Agentic RL 要做两件事：把**执行环境**部署起来，再把**训练**接上去。这两件事是正交的——换执行后端改一个环境变量 `CODE_RL_BACKEND`，换网络拓扑改一个 URL，训练代码一行都不用动。所以先在单机把流程跑通，之后再升级到跨机、跨网络。

可运行示例：[`cookbook/rl/code_rl/`](https://github.com/modelscope/twinkle/tree/main/cookbook/rl/code_rl)。任务是 MBPP 上的多轮代码生成 GRPO，一份 `train.py` 支持两个后端，任务/工具/奖励公式完全相同，只有执行环境不同，可以直接做对照实验。

## 选执行后端

| | **OpenEnv 嵌入式** | **OpenEnv server 模式** | **AgentENV** |
|---|---|---|---|
| 适配器 | `OpenEnv` + `EnvPool` | `OpenEnvClient` | `AgentEnv` |
| 环境跑在哪 | 训练进程内 | 独立的 HTTP/WebSocket 服务 | 独立的 Firecracker microVM |
| 隔离级别 | 无 | 进程 / 容器 | microVM（KVM 硬件虚拟化） |
| 执行器 | 取决于环境包 | smolagents AST 解释器（`coding_env`） | 真实 CPython |
| 单环境内存 | KB 级 | KB 级 | **1GB 级** |
| 文件 / `pip` / 子进程 | 取决于环境包 | 不支持 | 支持，销毁即清零 |
| 特殊硬件 | 无 | 无 | 需要 `/dev/kvm` |
| 部署成本 | 零 | 一条 `uvicorn` 命令 | 需部署 AgentENV 控制面 |

- **纯计算型环境**（棋类、文本游戏、能在训练进程内安全求值的判分逻辑）→ 嵌入式 `OpenEnv`。
- **要执行代码，且环境需要独立于训练扩缩容** → `OpenEnv` server 模式。轻三个数量级，标准库能解决的任务没必要上 microVM。
- **要跑 `unittest` + `@patch`，或模型需要写文件、装包、开子进程** → 必须 AgentENV。OpenEnv 的 `coding_env` 走 smolagents 的 `LocalPythonExecutor`，那是一个 **AST 解释器，不是操作系统级沙箱**：**装饰器会被静默忽略**（其源码里对 `decorator_list` 没有任何引用），mock 失效且不报错，reward 会算出一个看似正常的错误数值。它足以约束"只允许 import 白名单模块"，但不要用它跑对抗性代码。

后端能力的完整对比见[执行环境](../组件/Agentic/Envs.md)。

---

## 第一部分：部署

### 一、OpenEnv

#### 嵌入式模式（零部署）

环境直接在训练进程里实例化，没有网络开销：

```bash
cd cookbook/rl/multi_turn && python multi_turn_grpo.py
```

环境实例多、CPU 开销大时，用 `EnvPool` 把它们分片到独立的 CPU `DeviceGroup`，不占 GPU 进程的内存和 GIL：

```bash
ENV_REMOTE=1 ENV_NUM_WORKERS=8 ENV_POOL_SIZE=64 python multi_turn_grpo.py
```

| 变量 | 作用 |
|---|---|
| `ENV_REMOTE=1` | 环境放到专属 CPU DeviceGroup；不设则在 driver 本地跑（零 RPC 开销） |
| `ENV_NUM_WORKERS` | CPU worker 数，每个 rank 一个 `EnvPool` worker |
| `ENV_POOL_SIZE` | 池容量，`0` 表示自动取轨迹数 |

`EnvPool` 只适用于嵌入式 `OpenEnv`。不要把 `OpenEnvClient` 或 `AgentEnv` 放进 `EnvPool`——它们的 session / sandbox 生命周期在服务端，Ray 再分片一次不会有任何收益，只多一跳 RPC。

#### server 模式

在环境机器上（不需要 GPU、KVM、Docker）：

```bash
pip install openenv

# coding_env 是 OpenEnv 仓库里的子包（包名 openenv-coding_env），不在 PyPI 上，
# 只能从源码装。serve.sh 起的 server_app.py 从它 import PythonCodeActEnv 和
# PyExecutor；smolagents 也是它的依赖带进来的。
git clone https://github.com/huggingface/OpenEnv.git
pip install -e OpenEnv/envs/coding_env

cd cookbook/rl/code_rl
sh serve.sh                 # 4 workers x 64 sessions = 256 并发 session
HOST=127.0.0.1 sh serve.sh  # 训练在同一台机器时，绑回环不出网
```

训练机只需要 `pip install openenv`，客户端类由它提供，不需要装 `coding_env`。

#### server_app.py 对上游默认值的三处修改

`serve.sh` 起的是本目录下的 [`server_app.py`](https://github.com/modelscope/twinkle/blob/main/cookbook/rl/code_rl/server_app.py)，不是上游的 `coding_env.server.app`。直接照抄 upstream 会踩坑：

```python
class ConcurrentCodeEnv(PythonCodeActEnv):
    # 1. 上游默认 False，会让 create_app(max_concurrent_envs > 1) 直接抛
    #    ConcurrencyConfigurationError，服务端被限制成单 session。
    SUPPORTS_CONCURRENT_SESSIONS = True

    def reset(self, **kwargs):
        # 父类 reset() 会用上游默认值重建 executor 和 transform，之后必须重新配一遍。
        observation = super().reset()
        self._configure()
        return observation

    def _configure(self) -> None:
        # 2. 上游只授权 import json，MBPP 常用的 math / collections 会直接失败。
        self._executor = PyExecutor(additional_imports=list(ALLOWED_IMPORTS))
        # 3. 上游的 create_safe_coding_transform() 会用代码风格启发式覆盖
        #    observation.reward（含 open( / import os 罚 -1.0，短代码奖 +0.1）。
        #    本任务的奖励来自单元测试，同一通道上的风格分只会是噪声。
        self.transform = None
```

打开 `SUPPORTS_CONCURRENT_SESSIONS` 是安全的：`create_app` 拿到的是**类**（作为 factory），每个 WebSocket 连接都会新建一个实例，executor 和 state 都是实例私有的。

只用 HTTP 的 `/step`、`/reset` 端点是不行的——OpenEnv 的这些端点每次请求都新建一个 env、返回后立刻 `close()`，状态全丢。多轮 episode 必须走 WebSocket，`OpenEnvClient` 已经处理好了。

#### server 模式的容量

```
并发 session 上限 = WORKERS x MAX_CONCURRENT_ENVS   # serve.sh 默认 4 x 64 = 256
必须 ≥ BATCH_SIZE x NUM_GENERATIONS                 # 训练默认 4 x 8 = 32
```

超容量的连接会被服务端直接拒绝，表现为一批 rollout 里部分轨迹的观测全是 `Error:`。改大 `--batch-size` / `--num-generations` 后记得同步扩 `WORKERS` 或 `MAX_CONCURRENT_ENVS`。

### 二、AgentENV

[AgentENV](https://github.com/kvcache-ai/AgentENV) 用 Firecracker microVM 提供真实操作系统语义，适合 tool-integrated reasoning、SWE 类任务。

#### 前提条件自检

在目标机器上执行，三项都通过才能继续：

```bash
uname -r                                     # 需要 >= 6.8
ls -l /dev/kvm                               # 必须存在且可读写
modinfo ublk_drv >/dev/null && echo ublk-ok  # 需要 ublk 内核模块
```

- **裸金属机器**：通常直接满足。
- **云上 VM / GPU 实例**：需要云厂商开启嵌套虚拟化，多数默认关闭。
- **容器 / K8s Pod / DSW 等托管环境**：取决于**宿主机**内核与 `/dev/kvm`，且需要 privileged + 挂载 `/dev`。宿主机不满足时，容器内无法通过任何软件手段绕过。

不满足时：找一台满足条件的机器单独部署 AgentENV，训练侧把 `AENV_API_URL` 指过去即可，代码零改动。Firecracker **不支持 GPU 直通**，沙箱里跑不了 GPU 任务，所以环境机不需要 GPU。

#### Step 1：部署 server

方式 A — 安装脚本（推荐，Ubuntu 24.04）：

```bash
curl -fsSL https://raw.githubusercontent.com/kvcache-ai/AgentENV/main/scripts/install.sh \
  | sudo AENV_HOME_PATH=/data/aenv bash
sudo systemctl start aenv
```

脚本会创建专用的非 root `aenv` 账户（仅授予 `CAP_NET_ADMIN`/`CAP_SYS_ADMIN` 和 kvm 组权限）、加载 ublk 模块、下载 Firecracker/内核等运行时资源，并注册 systemd 服务。`AENV_HOME_PATH` 是数据目录（默认 `/var/lib/aenv`），镜像层和快照都放这里，建议指向大容量磁盘。它同时会装上 `aenv` CLI。

它只依赖两个外网点：`api.github.com` 取 release 元数据，然后从 `github.com/.../releases/download/` 下两个资产（`aenv-linux-x86_64` 9.3MB、`aenv-server-linux-x86_64.tar.gz` 68MB）。后者会 302 到 `release-assets.githubusercontent.com`，国内很容易在传输中途被 RST（`curl: (56) Recv failure` 或 `(92)`）。先试代理：`export https_proxy=...` 后重跑脚本，`install.sh` 全程用 curl，认 env 代理。没代理就走下面的方式 B。

方式 B — 源码编译（install.sh 下不动时）

先明确一件事：**编译只能解决 Rust 二进制那部分**。预编译 tarball 里除了 `server` 还带了一个 `deps/`（firecracker、guest kernel、tools 驱动、overlaybd、regctl），**这些不在编译产物里**，得单独准备。这是这条路最容易踩空的地方。

**1. 前置**

国内直连 `static.rust-lang.org` / `crates.io` 下不动，走阿里云镜像。**先设变量再装 rustup**，反了要重来。

```bash
# 写进 profile：AgentENV 有 rust-toolchain.toml，首次 cargo build 会再拉一次工具链
cat >> ~/.bashrc <<'EOF'
export RUSTUP_DIST_SERVER=https://mirrors.aliyun.com/rustup
export RUSTUP_UPDATE_ROOT=https://mirrors.aliyun.com/rustup/rustup
EOF
. ~/.bashrc

curl --proto '=https' --tlsv1.2 -sSf https://mirrors.aliyun.com/repo/rust/rustup-init.sh | sh
. "$HOME/.cargo/env"

# 阿里云只支持 sparse 索引，需 cargo >= 1.68。用覆盖写：>> 跟两次会出现重复的
# [source.crates-io]，TOML 直接解析失败
tee "${CARGO_HOME:-$HOME/.cargo}/config.toml" >/dev/null <<'EOF'
[source.crates-io]
replace-with = 'aliyun'

[source.aliyun]
registry = "sparse+https://mirrors.aliyun.com/crates.io-index/"

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

# 运行依赖（同 Dockerfile.agentenv 的 runtime-base 阶段）。libaio1t64 是 Ubuntu 24.04
# 的包名，22.04 / Debian 12 上叫 libaio1
sudo apt-get install -y ca-certificates curl dpkg e2fsprogs iproute2 iptables \
     jq libaio1t64 sudo umoci zstd
```

`protobuf-compiler` 必须用系统包：`make ci-deps-protoc` 会从 GitHub Releases 下 protoc，正是堵住的那条路。Debian 13+ 上 `pkg-config` 已改名 `pkgconf`。

镜像生没生效看 `cargo build` 的第一行：必须是 ``Updating `aliyun` index``。若仍是 `Updating crates.io index`，就是 `config.toml` 没被读到（常见原因：`CARGO_HOME` 指向其他位置；写文件时 `~/.cargo` 还不存在）。注意报错里的 `an/yh/anyhow` 是 sparse 路径，但 cargo >= 1.70 默认就走 sparse，**不能拿它当镜像生效的依据**。

**2. 编译与安装**

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

**3. 准备 deps**

```bash
E="AENV_CONFIG_PATH=/var/lib/aenv/config/config.toml AENV_HOME_PATH=/var/lib/aenv"
sudo env $E /usr/local/bin/server --setup-only
```

下不动先弄清文件从哪来——这五个 URL 就是它要下的全部（`config/deps_manifest.toml`）：

```
firecracker + cpu-template-helper  https://pub-4ee15c400f554ab7a9eac3f5bc8f53de.r2.dev/firecracker-1.15.1-patch-v1-x86_64.tgz
guest kernel                      https://pub-4ee15c400f554ab7a9eac3f5bc8f53de.r2.dev/vmlinux-6.1.175
regctl                            https://github.com/regclient/regclient/releases/download/v0.11.5/regctl-linux-amd64
overlaybd .deb                    https://github.com/containerd/overlaybd/releases/download/v1.0.18/overlaybd-1.0.18-20260710.cee2186.{target}.deb
tools.ext4                        ghcr.io/zlzgithub-0801/agentenv-tools:0.1.0   （OCI 镜像，需 regctl/docker 导出）
```

在能联网的机器上把卡住的那个下好，然后两种绕法。

**一、预放文件（推荐，5 样通用，不用改配置）** —— `download_file` 发现目标文件已存在且非空就跳过下载。目标路径就是失败日志里的 `dest=`：

```bash
# 日志：downloading url="https://pub-...r2.dev/firecracker-1.15.1-patch-v1-x86_64.tgz"
#                    dest=/var/lib/aenv/deps/firecracker/1.15.1-patch-v1/firecracker-1.15.1-patch-v1-x86_64.tgz
# scp 到 dest= 路径（文件名必须一模一样），然后：
sudo chown -R aenv:aenv /var/lib/aenv/deps
sudo env $E /usr/local/bin/server --setup-only
```

overlaybd 没有本地路径开关，但同样吃这一招：`.deb` 放到 `/var/lib/aenv/deps/overlaybd/downloads/` 下（文件名照 URL 末段，`{target}` 已替换），`package_url` 就不会被访问。

**二、改 `/var/lib/aenv/config/config.toml`** 指向你存文件的目录（下面的 `/opt/aenv-assets` 自己建，不是现成的）。这三段原本就有，**把键加进段内，不要追加同名段**（TOML 重复表直接解析失败）。只配当前卡住的那一项：

```toml
[firecracker]                                      # 已有段，boot_args 等原样保留
binary_path = "/opt/aenv-assets/firecracker"       # 注意是解开 tgz 后的二进制，不是 tgz
[kernel]                                           # 已有段（空）
image_path = "/opt/aenv-assets/vmlinux.bin"
[tools]                                            # 已有段，control_plane_port 原样保留
version = "0.1.0"                                  # 与 drive_path 必须成对
drive_path = "/opt/aenv-assets/tools.ext4"
```

或摆进 `deps_path`，版本号须与 `deps_manifest.toml` 逐字一致：

```bash
/var/lib/aenv/deps/firecracker/1.15.1-patch-v1/{firecracker,cpu-template-helper}
/var/lib/aenv/deps/kernel/vmlinux-6.1.175/vmlinux.bin      # 须重命名为 vmlinux.bin
/var/lib/aenv/deps/tools/0.1.0/tools.ext4
/var/lib/aenv/deps/regctl/v0.11.5/regctl

sudo chown -R aenv:aenv /var/lib/aenv/deps
```

**4. 起服务**

```bash
sudo env $E /usr/local/bin/server --setup-host --runtime-user aenv --runtime-group aenv
sudo chown -R aenv:aenv /var/lib/aenv
sudo env $E API_ADDR=127.0.0.1:8000 ./scripts/run-with-capabilities.sh /usr/local/bin/server
```

`/health` 返回 204 后，systemd unit 照 `scripts/install.sh` 第 3 段抄。

> 未端到端实测；deps 下载可达性因机器而异。

方式 C — Docker：

```bash
curl -fsSL https://raw.githubusercontent.com/kvcache-ai/AgentENV/main/scripts/docker-setup.sh | sudo bash
docker run -d --privileged -v /dev:/dev -p 8000:8000 ghcr.io/kvcache-ai/aenv-server:latest
```

这里的 Docker 只是**部署载体**，沙箱依然是 Firecracker microVM，仍然需要宿主机的 KVM。注意 `ghcr.io` 的镜像拉取走的也是 GitHub 的分发，国内可能同样不通，需要配镜像加速。

验证：`curl http://127.0.0.1:8000/health` 期望返回 204。

单机场景不需要 AgentENV 的 gateway / scheduler（那是多节点用的），直接连 server 的 `:8000`。

#### Step 2：构建环境模板

模板是沙箱的"出厂镜像"，把依赖预装并固化成快照，沙箱启动才能做到 ~50ms。**依赖必须烤进模板**，不要在训练时现场 `pip install`——每条轨迹重复装几十秒是不可接受的。

`aenv` 是 Rust 二进制（不是 pip 包）。方式 A 的安装脚本已经带上了；只想装 CLI 用：

```bash
curl -fsSL https://raw.githubusercontent.com/kvcache-ai/AgentENV/main/scripts/install-cli.sh | sudo bash
```

`aenv` 纯客户端，认证到哪就在哪建模板，在训练机上跑 `aenv build` 也可以：

```bash
aenv auth
# AENV server URL [http://localhost:8000]: http://127.0.0.1:8000
# API key: dummy          # 本地部署任意非空字符串即可

aenv build cookbook/rl/code_rl/Dockerfile -t twinkle-code --cpu-count 1 --memory-mb 1024
aenv template watch <template-id>      # 等到 ready
aenv template list                     # 确认模板存在
```

`aenv build` 支持 `FROM / RUN / ENV / WORKDIR / USER`（`ENTRYPOINT` 转为启动命令；`EXPOSE / VOLUME / LABEL` 被忽略）。只在 Dockerfile 变化时才需要重新构建。不加工直接用现成镜像：`aenv pull ubuntu:22.04 --name ubuntu`。

复杂工具的实现也建议烤进模板，工具 handler 只是一行调用：

```dockerfile
COPY tools/search.py /opt/tools/search.py
```

这样工具实现有版本、随快照分发、运行时零开销。

#### Step 3：训练侧冒烟测试

```bash
pip install e2b        # AgentENV 暴露 E2B 兼容 API，客户端复用官方 SDK
```

先用 5 行代码验证链路，不要直接跑训练：

```python
from twinkle_agentic.envs import AgentEnv

env = AgentEnv(template='twinkle-code', api_url='http://127.0.0.1:8000')
print(env.reset().observation)                                  # 沙箱启动
print(env.step('run_command', {'command': 'python -c "print(6*7)"'}).observation)
env.close()
```

`AgentEnv` 是无状态 HTTP 客户端，实现标准 `Env` 接口。它**不使用 `@remote_class`**：沙箱的放置、负载均衡、休眠唤醒、故障转移全部由 AgentENV 服务端负责，Ray 不参与环境调度。常用参数：

| 参数 | 说明 |
|---|---|
| `template` | 模板名，即 `aenv build -t` 的值 |
| `api_url` | server 或 gateway 地址 |
| `sandbox_timeout` | 沙箱空闲超时（秒）。超时后自动 pause（不是 kill），下次访问自动唤醒。**必须大于单条轨迹最长耗时** |
| `command_timeout` | 单条命令超时（秒） |
| `setup_commands` | 每次 reset 后执行的初始化命令 |
| `include_default_tools` | 是否暴露内置的 `run_command`/`write_file`/`read_file`，默认 `True` |

#### 内存预算

内存是唯一硬约束：

```
并发沙箱数 = BATCH_SIZE x NUM_GENERATIONS
需要内存   = 并发沙箱数 x 模板的 --memory-mb + 8GB（AgentENV 自身 + 系统）
```

按 `--memory-mb 1024`：默认 32 并发约需 40GB；先打通流程用 `2 x 4 = 8` 并发只需约 16GB。**试水阶段别买大机器。** 内存不足时的调整顺序：降低模板 `--memory-mb`（多数任务 512MB 够用）→ 减小 `batch_size` → 依赖自动休眠（空闲沙箱 pause 后内存归还宿主机）。

另外注意 CPU 争抢：沙箱与 dataloader/tokenizer 抢 CPU，`ENV_CONCURRENCY` 不宜超过空闲核数。

### 三、网络与安全

#### 两种连接方式

twinkle 只支持两种：

| 场景 | 连接方式 |
|---|---|
| 同机 / 同一内网、同一 VPC | **HTTP 直连** |
| 训练机与环境机跨网络 | **SSH 端口转发** |

框架不提供、也不建议在 twinkle 里引入 VPN / NAT 穿透类组件。那属于网络团队的既有基础设施，与训练代码无关——twinkle 侧看到的永远只是一个 `http://host:port`。

#### HTTP 直连（多数场景的主线）

企业的 GPU 机与 CPU 机通常本来就在同一个 VPC / IDC / K8s 集群里，不需要任何额外网络组件。环境侧绑**内网网卡**（不是 `0.0.0.0`）：

```bash
HOST=10.0.1.20 sh serve.sh          # OpenEnv；AgentENV 同理把监听地址配成内网 IP
```

训练侧只改一个环境变量：

```bash
OPENENV_BASE_URL=http://10.0.1.20:8000 sh run_openenv.sh    # OpenEnv
AENV_API_URL=http://10.0.1.20:8000     sh run_agentenv.sh   # AgentENV
```

| 事项 | 做法 |
|---|---|
| 收窄入方向 | 安全组只放行**训练机的 IP/32 或其安全组 ID**，端口只开 8000，绝不用 `0.0.0.0/0` |
| 确认绑定生效 | `ss -tlnp \| grep 8000`，看到的必须是 `10.0.1.20:8000` 而不是 `0.0.0.0:8000` |
| 服务保活 | 用 [`deploy/openenv-server.service`](https://github.com/modelscope/twinkle/blob/main/cookbook/rl/deploy/openenv-server.service)——rollout 批次会随服务一起死 |

#### SSH 端口转发（跨网络）

**OpenEnv 和 AgentENV 都没有任何认证**，端口可达就等于任意代码执行可达。SSH 在这里的作用不是"连接"，而是**替零认证的服务加一道认证**——复用已有的 SSH 密钥和登录审计，且它通常是企业里早已批准的运维通道。

SSH 端口转发工作在 TCP 层，对 WebSocket 完全透明——已实测：握手、跨消息 session 状态、8 个并发 session 复用一条连接，全部正常。

```bash
# 环境侧：只绑回环，网络上彻底不可达
HOST=127.0.0.1 sh serve.sh

# 训练侧：先建立端口转发，再连本地端口
ssh -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
    -L 8000:127.0.0.1:8000 user@env-host &
OPENENV_BASE_URL=http://127.0.0.1:8000 sh run_openenv.sh
```

两个地址容易搞混：`OPENENV_BASE_URL` 填的是**本地入口** `127.0.0.1:8000`；`-L` 里那个 `127.0.0.1:8000` 是**在环境机视角**解析的目标。AgentENV 同理，改 `AENV_API_URL`。

长期任务必须保活，转发一断整批 rollout 全废。别挂在交互式 shell 里：

```bash
autossh -M 0 -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -L 8000:127.0.0.1:8000 user@env-host
```

所有流量走一条 TCP 连接。每轮交互只有几 KB 代码和输出，实测 8 并发无压力；上百并发时可开多条转发映射到不同本地端口分流。

#### 沙箱出口（egress）

限制沙箱内代码能访问哪里，防止模型生成的代码扫内网（SSRF）或滥用带宽。AgentENV 在 `config/default.toml` 里有节点级强制策略，默认值就不错：

```toml
[network.egress]
# 沙箱自身的 egress policy 无法覆盖这些
always_denied_cidrs = [
  "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8",
  "169.254.0.0/16",              # ← 关键：挡住云元数据服务 169.254.169.254
  "172.16.0.0/12", "192.168.0.0/16",
]
```

`169.254.0.0/16` 这条最重要：封掉云厂商元数据服务，防止沙箱内代码偷 IAM 临时凭证进而拿到整个云账号权限。标准库算法题根本不需要出网，默认值直接用。需要放开时按沙箱粒度传策略：

```python
# base_policy: Default | Allow | Deny
{'base_policy': 'Deny',
 'egress': {'allowed_domains': ['pypi.org', 'mirrors.aliyun.com'],
            'denied_cidrs': ['10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16', '169.254.0.0/16']}}
```

另外：AgentENV 的 `aenv auth` 那个 API key **不是认证**，本地部署填任意非空字符串即可通过，也没有租户隔离——拿到任意 sandbox-id 就能读写、销毁该沙箱，`GET /sandboxes` 会列出所有人的沙箱。要对外提供服务，必须自建认证/租户/配额层，把 AgentENV 锁在私网，只暴露自己的上层 API。

---

## 第二部分：训练

以 `cookbook/rl/code_rl/` 为例。选「写代码」作为任务的原因很直接：奖励可以用单元测试客观计算，不需要裁判模型；多轮交互天然有意义（写 → 试跑 → 修 → 提交）。

### 四、训练侧接线

#### 1. 工具：模型看到什么

只暴露两个工具（`backends/openenv.py` / `backends/agentenv.py`）：

- `run_python(code)` —— 在环境里执行代码。
- `submit_solution(code)` —— **不发给服务端**，用 `register_tool` 在客户端本地处理，只把最终源码记在 env 上，供训练循环打分。

```python
def _submit_solution(env, arguments: Dict[str, Any]) -> str:
    code = (arguments.get('code') or '').strip()
    if not code:
        return "Error: 'code' argument is required."
    env.submitted_code = code
    return 'Solution submitted.'
```

这是一个通用模式：**「模型的动作」和「训练需要的记账」是两件事**，后者放本地 handler，不要污染环境协议。

System prompt 必须准确描述后端语义，否则模型会按错误的心智模型写代码。两个后端在这一点上是**相反的**：

- OpenEnv session：`The interpreter keeps its state between calls`——这一轮定义函数、下一轮可以直接调用；并列出可用模块白名单、声明无文件/网络访问。
- AgentENV：`Each call runs in a FRESH process, so every snippet must be self-contained`。

AgentENV 侧还用 `include_default_tools=False` 关掉了内置的 `run_command`/`read_file`/`write_file`，让动作空间与任务精确对齐，奖励归因才干净。

工具集是客户端概念，AgentENV 服务端本身**不定义"工具"**，它只提供能力原语（任意命令执行、文件读写、端口代理）。除 `register_tool(schema, handler)` 外还有命令模板型的 `register_command_tool`：

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

handler 签名是 `handler(env, arguments) -> str`（返回值即 observation），内部可用 `env.run_command(...)` 和 `env.sandbox`（原始 E2B 句柄，可访问 PTY、文件 watch 等）。同名注册会覆盖内置工具。

#### 2. 环境：一条轨迹一个 env

```python
def prepare_trajectories(samples, pool):
    envs = [backend.make_env() for _ in samples]
    # reset() 阻塞在网络上（WebSocket 握手，或启动一个沙箱），必须并发，
    # 否则一个 batch 32 条会串行等待。
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

三个要点：

1. **不要用 `EnvPool` / `@remote_class` 包 `OpenEnvClient` 或 `AgentEnv`**。session / sandbox 的生命周期在服务端，用 Ray 再分片一次没有收益，只多一跳 RPC。
2. **每条轨迹一个 `ToolManager`**。`ToolManager` 持有具体的 env 实例，共享会让所有轨迹的工具调用打到同一个 session。
3. **`envs` 必须在 `finally` 里关**。它们占着服务端容量，泄漏会让后续 step 因容量不足而失败：

```python
try:
    all_trajectories = rollout(expand_prompts, tool_manager=tool_managers)
    total_rewards, pass_rates = extract_rewards(envs, expanded, env_pool)
finally:
    close_envs(envs, env_pool)
```

`MultiTurnRollout` 的终止条件是**模型不再发出工具调用**（或达到 `MAX_TURNS` / 长度上限），它不读 `EnvTool.done`。所以要在 system prompt 里明确要求"提交后不要再调用工具"，否则回合会跑满 `MAX_TURNS`。

#### 3. 奖励：用单元测试，不要用裁判模型

MBPP 每条样本自带若干 `assert` 断言。rollout 结束后回放隐藏测试，两个后端的**脚本形态由后端能力决定，奖励公式完全相同**：

- **OpenEnv**：在**同一个 session** 里逐条执行，并把 `assert X` 改写成 `print(X)`。这样能把"断言求值为 False"和"代码崩了"区分开（失败的 `assert` 抛出的异常与解法内部崩溃无法区分），且一条测试抛异常不影响其余测试继续跑。该执行器是支持 `assert`/`try` 的，这里是可诊断性的选择，不是能力绕行。
- **AgentENV**：真实 CPython，直接生成一个普通脚本，每条断言用自己的 `try` 包住，最后打印 `TESTS_PASSED n total`。

OpenEnv 侧用 `env.execute()` 而不是 `env.step()`：`execute()` 返回服务端**原始** `StepResult`，可以读 `exit_code` 这类结构化字段；`step()` 返回的是渲染给模型看的文本，并且会计入 episode。打分逻辑不应该出现在模型的对话里。

奖励整形：

```python
rate = passed / total if total else 0.0
if total and passed == total:
    return 1.0, rate                    # 全部通过
if getattr(env, 'submitted_code', None):
    return 0.1 + 0.4 * rate, rate       # 提交了，部分通过 → 0.1 ~ 0.5
return 0.0, rate                        # 没提交
```

形状上保证「全对 > 部分对 > 提交但全错 > 没提交」，且全对的 1.0 明显高于部分对的上限 0.5，避免模型学会"交一个能过第一个测试的假实现"。**提交动作本身给 0.1 底分**，是为了在训练早期给出梯度信号——否则一开始所有轨迹奖励全 0，GRPO 的组内优势全是 0，学不到任何东西。

沙箱本身不产生 reward（`AgentEnv.evaluate` 默认返回 0），打分统一在训练循环里做。需要在沙箱内判分时，让工具输出结构化结果再在 driver 侧解析。

#### 4. GRPO：组内相对优势

```python
batch = [dataset[(sample_cursor + i) % len(dataset)] for i in range(BATCH_SIZE)]
expanded = [s for s in batch for _ in range(NUM_GENERATIONS)]   # 同题连续 N 份
...
advantages = advantage_fn(total_rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()
```

`expanded` 用的是 `for s in batch for _ in range(N)`，同一道题的 N 条 rollout 在列表里**连续排列**，`GRPOAdvantage` 按这个布局切分组。写成 `for _ in range(N) for s in batch` 会让分组完全错位，训练看起来能跑但学不到东西。

### 五、跑起来

先用小并发打通全链路再放大，省时间也省钱：

```bash
cd cookbook/rl/code_rl

# OpenEnv
OPENENV_BASE_URL=http://127.0.0.1:8000 sh run_openenv.sh \
    --batch-size 2 --num-generations 4 --max-steps 2

# AgentENV
AENV_API_URL=http://127.0.0.1:8000 AENV_TEMPLATE=twinkle-code sh run_agentenv.sh \
    --batch-size 2 --num-generations 4 --max-steps 2
```

但别压得太狠：轨迹数（`batch-size x num-generations`）必须 ≥ `--model-gpus`（默认 4），否则长度过滤后剩余不足，整批会被直接跳过——日志只反复刷 `skipping this batch`，看起来像卡住而不是报错。上面的 `2 x 4 = 8` 留了 2 倍余量。

放大到正式训练：

```bash
OPENENV_BASE_URL=http://10.0.0.5:8000 MAX_TURNS=8 ENV_CONCURRENCY=32 \
sh run_openenv.sh --batch-size 8 --num-generations 16 --max-steps 500
```

可覆盖的环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `MAX_TURNS` | `6` | 单回合最大工具调用轮数 |
| `ENV_CONCURRENCY` | `16` | driver 并发创建/销毁/打分的线程数 |
| `OPENENV_BASE_URL` | `http://127.0.0.1:8000` | OpenEnv 服务地址（也可以是负载均衡器地址） |
| `OPENENV_ENV_NAME` | `coding_env` | 环境包名，决定客户端 + Action 类 |
| `OPENENV_MESSAGE_TIMEOUT_S` | `120` | 单条消息超时 |
| `AENV_API_URL` | `http://127.0.0.1:8000` | AgentENV 地址 |
| `AENV_TEMPLATE` | `twinkle-code` | 模板名 |
| `SANDBOX_TIMEOUT` | `600` | 沙箱空闲超时（秒） |
| `AENV_COMMAND_TIMEOUT` | `60` | 沙箱内单条命令超时（秒） |

训练超参（`--model-gpus`/`--batch-size`/`--num-generations` 等）走 CLI 参数，默认值在 `common_args.sh`——两个后端共用同一份，这样奖励的变化才能归因到执行后端而不是超参差异。

#### 指标

日志每步打一行，同时通过 `swanlab.log` 上报（`project='twinkle'`，实验名 `code-rl-<backend>`）：

```
[Step 0] {'train/code_acc': 0.031, 'train/test_pass_rate': 0.208, 'train/avg_reward': 0.145, ...}
```

| 指标 | 含义与解读 |
|---|---|
| `train/code_acc` | pass@1：全部隐藏测试通过的比例。这是真正要涨的指标 |
| `train/test_pass_rate` | 平均单测通过率。比 `code_acc` 平滑，早期先看它动没动 |
| `train/avg_reward` | 整形后的奖励均值。若它涨而 `code_acc` 不涨，说明模型在薅 0.1 的提交底分 |
| `train/avg_turns` | 平均轮数。贴着 `MAX_TURNS` 说明模型经常不提交就用完轮数 |
| `train/max_turns` / `train/min_turns` | 批内轮数分布。`max_turns` 顶到 `MAX_TURNS` 说明有轨迹被截断；`min_turns` 接近 1 说明常有轨迹第一轮就提交或放弃 |
| `train/approx_kl` | 新旧策略的 KL（Schulman K3 估计）。突然飙升是策略崩塌的前兆 |
| `train/clip_ratio` | PPO 裁剪触发比例（另有 `_low` / `_high` 分向）。长期偏高说明单步更新太激进 |
| `train/token_kl_max` / `train/token_ratio_max` | 单 token 的 KL / 概率比极值，用于定位崩塌 |
| `train/policy_confidence` | `exp(mean_logp)`，策略平均置信度 |

这些 GRPO 指标由 `model.add_metric('GRPOMetric', is_training=True, epsilon=...)` 注册，`epsilon` 要与 `set_loss('GRPOLoss', epsilon=...)` 保持一致，否则 `clip_ratio` 统计的裁剪阈值与实际生效的不是同一个。`train/entropy` 需要 `GRPOLoss(entropy_coef > 0)` 才会出现——注意那会往 loss 里加 entropy bonus，**改变训练目标**，不要只为了看指标而打开。

### 六、迁移到你自己的任务

复用示例骨架时通常只需要改四处，训练循环、GRPO 配置、权重同步、指标都不用动：

1. **数据集**：`load_mbpp()` 换成你的加载函数，产出 `{'prompt', ...打分所需字段}`。
2. **工具**：`TOOL_SCHEMA` + handler。服务端能力走默认 action 通路，客户端记账用 `register_tool`。
3. **奖励**：`extract_rewards()` 里的 `score()`。优先选能客观计算的信号（单测、精确匹配、可执行校验），实在没有再上裁判模型。
4. **System prompt**：必须准确描述后端语义（状态是否跨轮保留、有哪些模块/权限、轮数预算）。

### 七、排查

| 现象 | 原因与处理 |
|---|---|
| 一批轨迹里部分观测全是 `Error:` | 服务端容量不足。核对 `WORKERS x MAX_CONCURRENT_ENVS ≥ BATCH_SIZE x NUM_GENERATIONS` |
| `ConcurrencyConfigurationError` | 起的是上游 `coding_env.server.app`（单 session）。用本目录的 `server_app.py` |
| 奖励里混进 ±0.1 / -1.0 的怪值 | 环境的 reward transform 没关掉，检查 `self.transform = None` |
| 模型报 `import math` 不被允许 | `ALLOWED_IMPORTS` 白名单未生效；确认 `reset()` 之后重新调了 `_configure()` |
| 超时 / 消息等待报错 | 调大 `OPENENV_MESSAGE_TIMEOUT_S`。执行器自身另有三重上限：单次执行 30s wall-clock、1000 万次操作、100 万次 while 迭代，所以该值要留在 30s 以上才有意义 |
| AgentENV server 启动报 `/dev/kvm is not accessible` | 运行账户不在 kvm 组，或宿主机无 KVM。执行 `sudo server --setup-host --runtime-user aenv --runtime-group aenv` 后重启服务 |
| 报 `ublk_drv is not loaded` | `sudo modprobe ublk_drv`；内核 < 6.8 需升级内核 |
| `ImportError: AgentEnv requires the E2B SDK` | `pip install e2b` |
| 创建沙箱报模板不存在 | `aenv template list` 确认名字；构建是否 ready（`aenv template watch`） |
| 沙箱内 `pip install` 失败 | 出口网络被策略拦截，或未配 pip 镜像源；建议改为在模板里预装 |
| 轨迹中途报沙箱不可用 | `SANDBOX_TIMEOUT` 小于轨迹耗时，被自动 pause。调大该值 |
| batch 启动很慢 | 调大 `ENV_CONCURRENCY`；确认依赖已烤进模板而非运行时安装 |
| 有效轨迹数不足被跳过 | 多数轨迹超长被过滤。调小 `MAX_TURNS`/`--max-tokens`，或收敛工具输出（观测已默认截断到 32K 字符）；也要确认轨迹数 ≥ `--model-gpus` |
| 奖励长期为 0 | 先单独跑 `run_tests` 对一个已知正确解打分，确认打分链路本身是通的，再怀疑模型 |
| `Address already in use` | 端口被占。`ss -tlnp \| grep 8000` 查占用者，或换 `PORT=8001 sh serve.sh` |

### 上线检查清单

- [ ] 服务**没有**绑在 `0.0.0.0`（`ss -tlnp | grep 8000` 确认）
- [ ] 内网直连：安全组入方向只放行训练机，不是 `0.0.0.0/0`
- [ ] 跨网络：环境侧绑 `127.0.0.1`，只经 SSH 端口转发访问
- [ ] 容量 ≥ `BATCH_SIZE x NUM_GENERATIONS`（AgentENV 再核内存）
- [ ] 环境服务有保活（systemd），SSH 转发有保活（autossh），不是挂在交互式 shell 上
- [ ] 已用 `--batch-size 2 --num-generations 4 --max-steps 2` 打通过全链路
- [ ] 轨迹数（`batch-size x num-generations`）≥ `--model-gpus`
- [ ] 没有为了"让沙箱联通"而放宽 AgentENV 的 `always_denied_cidrs`

## 相关文档

- 组件参考：[执行环境](../组件/Agentic/Envs.md)（`Env` 抽象、`EnvTool`、OpenEnv 两种模式、`EnvPool`）
- 多轮工具调用：[多轮工具调用](../组件/Agentic/Multi-Turn-Tool-Usage.md)
- 部署模板：`cookbook/rl/deploy/`
- 可运行示例：`cookbook/rl/code_rl/`（代码任务，两个后端）、`cookbook/rl/multi_turn/`（嵌入式 OpenEnv）
- OpenEnv 上游仓库：<https://github.com/meta-pytorch/OpenEnv>
- AgentENV 官方文档：<https://kvcache-ai.github.io/AgentENV/>
