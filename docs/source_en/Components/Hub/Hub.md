# Hub

The Hub component provides unified access to model and dataset hubs, supporting both ModelScope and Hugging Face.

```python
from twinkle.hub import Hub

# Download from ModelScope
Hub.download('ms://Qwen/Qwen3.5-4B', local_dir='./models')

# Download from Hugging Face
Hub.download('hf://Qwen/Qwen3.5-4B', local_dir='./models')

# Upload checkpoints
Hub.upload(local_path='./my-model', repo_id='my-org/my-model', hub='ms')
```

The `ms://` and `hf://` prefixes determine which hub to use. Hub handles authentication, caching, and progress tracking automatically.

> Hub is used internally by Dataset, Model, and other components. You can also use it directly for custom download/upload workflows.
