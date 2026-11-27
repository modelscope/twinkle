# 3×8 H20 Runtime 部署

线上每台机器使用一个镜像、一条运行命令和一个平台对外端口。Gateway 运行 HTTP 转发和 Redis。Observability 运行 Collector、Mimir、Loki、Tempo 和 Grafana。GPU A 担任 Ray head，A/B/C 共享 checkpoint NAS。

| 机器 | Runtime 运行命令 | 平台端口与内网连接 |
| --- | --- | --- |
| Gateway | `exec /sbin/tini -- /usr/local/bin/twinkle-gateway /dashscope/caches/application/runtime-config/twinkle-gateway.env` | 对外 `9000`，Redis 内网 `6380`，上游为 GPU A 内网 `8000` |
| Observability | `exec /sbin/tini -- /usr/local/bin/twinkle-observability /dashscope/caches/application/runtime-config/twinkle-observability.env` | 对外 Grafana `3000`，Collector 内网 `4317` |
| GPU A | `twinkle-runtime h20-a` | Ray head `6379`，Twinkle Serve `8000`，27B trainer 和 sampler |
| GPU B | `twinkle-runtime h20-b` | Ray worker，35B trainer 和 sampler |
| GPU C | `twinkle-runtime h20-c` | Ray worker，4B/9B/Embedding/Reranker 动态池 |

## 在构建机打包并推送

从仓库根目录执行。脚本用 `sudo docker` 登录仓库，构建和推送 `linux/amd64` 镜像。`all` 会在两个镜像都构建成功后推送。每次发布使用新版本标签，记录推送输出里的镜像 digest。镜像只包含公开代码和模板，Runtime 启动时读取部署现场的私有配置。

```bash
build_script=cookbook/client/server/megatron/deployment/build-and-push-cpu-images.sh
bash "$build_script" all 20261125-1
# 单独构建和推送
bash "$build_script" gateway 20261125-1
bash "$build_script" observability 20261125-1
```

镜像标签为 `twinkle-gateway-<版本>` 和 `twinkle-observability-<版本>`。脚本默认使用阿里云 Alpine 软件包镜像和 BuildKit 层缓存。可用 `ALPINE_MIRROR=` 采用基础镜像自带的软件包源。Docker Hub 镜像层下载慢时，可用 `DOCKERHUB_MIRROR=docker.m.daocloud.io bash "$build_script" all 20261125-1` 走镜像站。

## 创建和调试 CPU Runtime

1. 为 Gateway 和 Observability 分别创建 Runtime 版本，填写已推送的镜像、表中运行命令和端口。Gateway 的 HTTP 探针使用 `/health`，Observability 使用 `/api/health`。在研发部署启用调试模式。平台此时替换启动命令，登录容器后手动执行表中对应命令。
2. 需要使用应用级持久挂载时，在部署扩展配置中填写 `{"cache_volume":{"enabled":true,"scope":"application"}}`。平台在容器内挂载 `/dashscope/caches/application`。Runtime 配置文件放在各自应用卷的 `runtime-config`，只向所需运行用户开放读权限。真实地址和密钥填入部署现场的文件。
3. 从 [Gateway 配置示例](gateway.runtime.env.example) 创建私有配置。在 `TWINKLE_UPSTREAM` 填 GPU A 的内网 `IP:8000`，GPU 到位前可填模拟 HTTP 服务地址。Redis 的数据目录可放在已有 NFS 下的独立子目录，整个目录由一个 Redis 实例写入。两个 Redis 用户的密码分别配置。备份数据后再演练补机。
4. 从 [Observability 配置示例](observability.runtime.env.example) 创建私有配置。使用 Gateway 的内网 Redis 地址和独立只读用户。Observability 镜像以 UID 472 运行，配置对该用户可读，`OBS_DATA_DIR` 可写。调试阶段可用临时目录。生产用可重新挂载、支持 SQLite 和后端本地文件语义的持久卷。 [SQLite 网络文件系统说明](https://www.sqlite.org/useovernet.html)
5. Gateway 用真实或模拟上游验证鉴权头、流式响应及 Redis 重启回读。Observability 用测试 OTLP 标记验证日志、指标、trace 和快照。切出调试模式后，再验证单个关键进程退出时启动命令收尾、平台重拉和数据恢复。

## GPU 和发布顺序

- GPU A/B/C 使用同一个计算镜像 digest。设置各自的 `NODE_IP`，并挂载同一绝对路径到 `TWINKLE_DEFAULT_SAVE_DIR`。B/C 的 `RAY_ADDRESS` 指向 A 的内网 `6379`。A 注入 `TWINKLE_REDIS_URL`、Observability 的 `TWINKLE_OTLP_ENDPOINT` 和 `TWINKLE_TELEMETRY_USER_ID_KEY`。内置 [H20 YAML](../server_config_h20_3x8.yaml) 用节点标签固定模型落点。
- 先启动 Gateway、Observability 和 GPU A，再启动 GPU B/C。A 等到三个节点各注册 8 卡才部署 Twinkle。真实接流前用 SDK 验证鉴权、训练、采样、checkpoint 保存和跨节点读取，同时检查 Grafana 上的对应请求。
- 更新前先停止接新训练，给在途任务留出结果轮询和保存 checkpoint 的时间。按 A、B/C 的顺序停止计算，三台切换到同一新版本后按原顺序启动。CPU 角色通过平台单独更新，保留原数据目录。重建或换机导致地址变化时，更新依赖方的私有配置。
- 保留上一版镜像和 YAML 供回滚。直连灰度失败时，可切回已验证的 Mesh/Adapter 路径。

[观测 Compose](../../../../observability/docker-compose.production.yaml) 用于有 Docker Compose 的开发环境。

## Spectrum GPU 成组验收

在 Linux 构建机用部署分支打包 GPU 镜像。脚本用当前 Git 提交中的文件构建，可以保留本地未提交文件。它实时打印构建日志和实际依赖版本，仅在版本匹配、Swift 契约可加载且依赖检查没有新增冲突后推送。

```bash
git fetch origin server-runtime-deployment:refs/remotes/origin/server-runtime-deployment && git checkout server-runtime-deployment && git pull --ff-only origin server-runtime-deployment && bash cookbook/client/server/megatron/deployment/build-and-push-gpu-image.sh
```

1. 为同一个应用创建 GPU Runtime 版本，使用脚本打印的镜像。运行命令填 `exec /usr/local/bin/twinkle-runtime spectrum-group`，平台端口填 `9000`，探针填 `/healthz`。只将此探针交给平台。把 Serve `8000`、Ray 和 OTLP 留在授权的内网。
2. 创建一个成组部署，基数选 3，组数填 1。扩展配置开启 `{"cache_volume":{"enabled":true,"scope":"application"}}`。[GPU 私有配置模板](gpu-group.runtime.env.example) 中的变量可以填到部署环境变量，也可以放在应用卷的 `runtime-config/twinkle-gpu.env`。Redis URL 和 Grafana 必须指向同一个 Redis 和 key prefix，Grafana 使用独立的快照只读账户。验双机时同时把基数和 `TWINKLE_EXPECTED_RAY_NODES` 改为 2。
3. 先核对平台给的 Rank、Ray 的 24 张卡和三个节点的任务落点，再通过 Head 的 `/readyz` 检查业务路由。默认 [4B 验收配置](../server_config_h20_smoke.yaml) 把 Megatron 放 A，在 B 预热一个 vLLM 实例，C 参与 Ray 组网。使用 [3×8 正式模型配置](../server_config_h20_3x8.yaml) 时显式设置 `TWINKLE_SERVER_CONFIG`，单独验收各模型。
4. 运行客户端训练并对账。训练活跃时查 Grafana 的 **Current trainings**，打开对应 trace 的 `task_queue.execute`，通过 **Allocated GPU utilization** 或 **Allocated GPU memory** 查实际分配的卡。在同一次训练中保存训练和采样 checkpoint，让 B 从共享卷读取采样权重。上传 ModelScope 时只使用获准的私有仓库。

### 复用镜像验证纯代码改动

先核对已运行镜像的包版本满足目标提交的依赖。容器能访问 GitHub 时，保留原镜像，创建新的 Runtime 版本，给每个 Rank 使用下面同一条命令。将 `REVIEWED_COMMIT` 替换为验收过的完整 Git SHA。每个 Rank 将源码放在同一绝对路径，Ray worker 才能沿用 Head 传播的 `PYTHONPATH`。运行日志会打印提交号。依赖改变时使用上面的镜像构建脚本。

```bash
exec bash -c 'set -eu; code="/tmp/twinkle-source/$1"; mkdir -p "$code"; cd "$code"; git init -q; timeout 90 git fetch -q --depth=1 https://github.com/modelscope/twinkle.git "$1"; git checkout -q --detach FETCH_HEAD; test "$(git rev-parse HEAD)" = "$1"; printf "Twinkle source %s\n" "$1"; export PYTHONPATH="$code/src${PYTHONPATH:+:$PYTHONPATH}"; exec bash cookbook/client/server/twinkle-runtime.sh spectrum-group' twinkle REVIEWED_COMMIT
```
