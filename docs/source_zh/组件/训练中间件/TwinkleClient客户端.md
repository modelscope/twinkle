# TwinkleClient 客户端

`TwinkleClient` 是与 Twinkle REST API 交互的 Python 客户端，管理会话、训练任务和检查点。

## 初始化

```python
from twinkle_client.manager import TwinkleClient

client = TwinkleClient(
    base_url='http://localhost:8000',   # 或 TWINKLE_SERVER_URL 环境变量
    api_key='your-api-key',             # 或 TWINKLE_SERVER_TOKEN 环境变量
    route_prefix='/twinkle',            # API 路由前缀
    session_heartbeat_interval=10,      # 心跳间隔（秒）
    session_metadata={'user': 'alice'}, # 可选的会话元数据
)
```

初始化时客户端会：
1. 将 `base_url` 和 `api_key` 设置到共享上下文（所有客户端对象自动使用）
2. 创建服务端会话
3. 启动后台心跳线程保持会话活跃

## 健康检查

```python
is_healthy = client.health_check()  # 返回 True/False
capabilities = client.get_server_capabilities()  # 支持的模型
```

## 训练任务

```python
# 列出训练任务
runs = client.list_training_runs(limit=20, offset=0)

# 带分页游标列出
runs, cursor = client.list_training_runs_with_cursor(limit=20)

# 获取特定任务
run = client.get_training_run(run_id='run_abc123')

# 按基础模型查找
qwen_runs = client.find_training_run_by_model('Qwen/Qwen3.5-4B')
```

## 检查点

```python
# 列出训练任务的检查点
checkpoints = client.list_checkpoints(run_id='run_abc123')

# 获取检查点路径
parsed = client.get_checkpoint_path(run_id, checkpoint_id)
# parsed.path         → 文件系统路径
# parsed.twinkle_path → twinkle:// URI

# 获取最新检查点（用于恢复训练）
latest_path = client.get_latest_checkpoint_path(run_id)

# 删除检查点
client.delete_checkpoint(run_id, checkpoint_id)
```

## 容量与权重信息

```python
# LoRA 容量
capacity = client.get_capacity_info()
# capacity.max_loras, capacity.used_loras, capacity.free_loras

# 权重元数据
info = client.get_weights_info('twinkle://run_id/weights/checkpoint')
# info.base_model, info.is_lora, info.lora_rank
```

## 清理

```python
client.close()  # 停止心跳线程（也通过 atexit 自动注册）
```
