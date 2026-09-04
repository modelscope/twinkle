已写好。既然模型目录存在 `tokenizer.json`，会直接从 `/nas/disk1/random-deepseek-v4-4b` 加载，不需要单独配置 `tokenizer_id`。

生成的文件：

- [server_config_dsv4_0731.yaml](/Users/linjiajia/project/twinkle/cookbook/client/server/transformer/server_config_dsv4_0731.yaml)
- [server_dsv4_0731.py](/Users/linjiajia/project/twinkle/cookbook/client/server/transformer/server_dsv4_0731.py)
- [dsv4_multi_lora_sft.py](/Users/linjiajia/project/twinkle/cookbook/client/twinkle/dsv4_multi_lora_sft.py)

配置为：

- 本地模型：`/nas/disk1/random-deepseek-v4-4b`
- 本地数据集：`/model/ljl/dataset/self-cognition.jsonl`
- 2 张 GPU
- Native FSDP2 + EP=2
- `memory_efficient_init: true`
- 两个 LoRA：`tenant_a`、`tenant_b`
- `DeepseekV4Template`
- 两张 GPU 全部用于训练，因此没有启动 vLLM sampler

服务端启动：

```bash
cd /model/ljl/project/remote-git/dsv4_0731/twinkle
export PYTHONPATH="$PWD/src:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0,1 ray start \
  --head \
  --num-gpus=2 \
  --disable-usage-stats \
  --include-dashboard=false

CUDA_VISIBLE_DEVICES=0,1 python3 \
  cookbook/client/server/transformer/server_dsv4_0731.py
```

如果 Ray 集群已经启动，跳过 `ray start`。

另一个终端启动客户端：

```bash
cd /model/ljl/project/remote-git/dsv4_0731/twinkle
export PYTHONPATH="$PWD/src:$PYTHONPATH"

TWINKLE_SERVER_URL=http://127.0.0.1:8000 \
TWINKLE_SERVER_TOKEN=EMPTY_TOKEN \
python3 cookbook/client/twinkle/dsv4_multi_lora_sft.py
```

`DeepseekV4Template` 会优先加载模型目录里的：

```text
tokenizer.json
encoding/encoding_dsv4.py
```

若不存在 `encoding/encoding_dsv4.py`，才回退到 Twinkle 内置 encoding。相关检查共 `31 passed, 1 skipped`；当前改动尚未 commit/push。
