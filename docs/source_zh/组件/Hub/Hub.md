# Hub

Hub 组件提供对模型和数据集仓库的统一访问，同时支持魔搭和 Hugging Face。

```python
from twinkle.hub import Hub

# 从魔搭下载
Hub.download('ms://Qwen/Qwen3.5-4B', local_dir='./models')

# 从 Hugging Face 下载
Hub.download('hf://Qwen/Qwen3.5-4B', local_dir='./models')

# 上传检查点
Hub.upload(local_path='./my-model', repo_id='my-org/my-model', hub='ms')
```

`ms://` 和 `hf://` 前缀决定使用哪个仓库。Hub 自动处理认证、缓存和进度跟踪。

> Hub 被 Dataset、Model 和其他组件内部使用。你也可以直接使用它进行自定义的下载/上传工作流。
