# 3×8 H20 Runtime 部署

线上每台机器使用一个镜像、一条运行命令和一个平台对外端口。CPU 1 运行 HTTP 转发和 Redis。CPU 2 运行 Collector、Mimir、Loki、Tempo 和 Grafana。GPU A 担任 Ray head，A/B/C 共享 checkpoint NAS。

| 机器 | Runtime 运行命令 | 平台端口与内网连接 |
| --- | --- | --- |
| CPU 1 | `exec /sbin/tini -- /usr/local/bin/twinkle-cpu1 /dashscope/caches/application/runtime-config/twinkle-cpu1.env` | 对外 `9000`，Redis 内网 `6380`，上游为 GPU A 内网 `8000` |
| CPU 2 | `exec /sbin/tini -- /usr/local/bin/twinkle-cpu2 /dashscope/caches/application/runtime-config/twinkle-cpu2.env` | 对外 Grafana `3000`，Collector 内网 `4317` |
| GPU A | `twinkle-runtime h20-a` | Ray head `6379`，Twinkle Serve `8000`，27B trainer 和 sampler |
| GPU B | `twinkle-runtime h20-b` | Ray worker，35B trainer 和 sampler |
| GPU C | `twinkle-runtime h20-c` | Ray worker，4B/9B/Embedding/Reranker 动态池 |

## 在构建机打包并推送

从仓库根目录执行。脚本用 `sudo docker` 登录仓库，构建两个 `linux/amd64` 镜像，构建都成功后再推送。每次发布使用新版本标签，记录推送输出里的镜像 digest。镜像只包含公开代码和模板，Runtime 启动时读取部署现场的私有配置。

```bash
bash cookbook/client/server/megatron/deployment/build-and-push-cpu-images.sh 20261125-1
```

使用独立构建上下文时，解压后从根目录运行 `bash build-and-push-cpu-images.sh 20261125-1`。脚本默认使用阿里云 Alpine 软件包镜像和 BuildKit 层缓存。可用 `ALPINE_MIRROR=` 采用基础镜像自带的软件包源。基础镜像下载走构建机 Docker 的镜像仓库配置。

## 创建和调试 CPU Runtime

1. 为 CPU 1 和 CPU 2 分别创建 Runtime 版本，填写已推送的镜像、表中运行命令和端口。CPU 1 的 HTTP 探针使用 `/health`，CPU 2 使用 `/api/health`。在研发部署启用调试模式。平台此时替换启动命令，登录容器后手动执行表中对应命令。
2. 需要使用应用级持久挂载时，在部署扩展配置中填写 `{"cache_volume":{"enabled":true,"scope":"application"}}`。平台在容器内挂载 `/dashscope/caches/application`。Runtime 配置文件放在各自应用卷的 `runtime-config`，只向所需运行用户开放读权限。真实地址和密钥填入部署现场的文件。
3. 从 [CPU 1 配置示例](cpu1.runtime.env.example) 创建私有配置。在 `TWINKLE_UPSTREAM` 填 GPU A 的内网 `IP:8000`，GPU 到位前可填模拟 HTTP 服务地址。Redis 的数据目录可放在已有 NFS 下的独立子目录，整个目录由一个 Redis 实例写入。两个 Redis 用户的密码分别配置。备份数据后再演练补机。
4. 从 [CPU 2 配置示例](cpu2.runtime.env.example) 创建私有配置。使用 CPU 1 的内网 Redis 地址和独立只读用户。CPU 2 镜像以 UID 472 运行，配置对该用户可读，`OBS_DATA_DIR` 可写。调试阶段可用临时目录。生产用可重新挂载、支持 SQLite 和后端本地文件语义的持久卷。 [SQLite 网络文件系统说明](https://www.sqlite.org/useovernet.html)
5. CPU 1 用真实或模拟上游验证鉴权头、流式响应及 Redis 重启回读。CPU 2 用测试 OTLP 标记验证日志、指标、trace 和快照。切出调试模式后，再验证单个关键进程退出时启动命令收尾、平台重拉和数据恢复。

## GPU 和发布顺序

- GPU A/B/C 使用同一个计算镜像 digest。设置各自的 `NODE_IP`，并挂载同一绝对路径到 `TWINKLE_DEFAULT_SAVE_DIR`。B/C 的 `RAY_ADDRESS` 指向 A 的内网 `6379`。A 注入 `TWINKLE_REDIS_URL`、CPU 2 的 `TWINKLE_OTLP_ENDPOINT` 和 `TWINKLE_TELEMETRY_USER_ID_KEY`。内置 [H20 YAML](../server_config_h20_3x8.yaml) 用节点标签固定模型落点。
- 先启动 CPU 1、CPU 2 和 GPU A，再启动 GPU B/C。A 等到三个节点各注册 8 卡才部署 Twinkle。真实接流前用 SDK 验证鉴权、训练、采样、checkpoint 保存和跨节点读取，同时检查 Grafana 上的对应请求。
- 更新前先停止接新训练，给在途任务留出结果轮询和保存 checkpoint 的时间。按 A、B/C 的顺序停止计算，三台切换到同一新版本后按原顺序启动。CPU 角色通过平台单独更新，保留原数据目录。重建或换机导致地址变化时，更新依赖方的私有配置。
- 保留上一版镜像和 YAML 供回滚。直连灰度失败时，可切回已验证的 Mesh/Adapter 路径。

[观测 Compose](../../../../observability/docker-compose.production.yaml) 用于有 Docker Compose 的开发环境。
