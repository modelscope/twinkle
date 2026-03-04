# Twinkle Training Service on ModelScope

Alongside the open-source release of the Twinkle framework, we also provide a hosted model training service (Training as a Service) powered by ModelScope's backend infrastructure. Developers can use this service to experience Twinkle's training API for free.

The model currently running on the cluster is [Qwen/Qwen3-30B-A3B-Instruct-2507](https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B-Instruct-2507). Below are the detailed usage instructions:

## Step 1. Register a ModelScope Account and Apply to Join the twinkle-explorers Organization

Developers first need to register as a ModelScope user and apply to join the [Twinkle-Explorers](https://modelscope.cn/organization/twinkle-explorers) organization to obtain access permissions. The current free Serverless training experience is still in beta testing and is only available to users within the organization. You can also use Twinkle✨ by deploying the service locally.

Registration link: https://www.modelscope.cn/

After registering and being approved to join the [Twinkle-Explorers](https://modelscope.cn/organization/twinkle-explorers) organization, obtain your API-Key (i.e., the ModelScope platform access token) from this page: https://www.modelscope.cn/my/access/token.

API endpoint: `base_url="https://www.modelscope.cn/twinkle"`

## Step 2. Review the Cookbook and Customize Development

We strongly recommend that developers review our [cookbook](https://github.com/modelscope/twinkle/tree/main/cookbook/client/tinker) and build upon the training code provided there.

Sample code:

```python
import os
from tqdm import tqdm
from tinker import types
from twinkle_client import init_tinker_client
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.preprocessor import SelfCognitionProcessor
from twinkle.server.tinker.common import input_feature_to_datum

base_model = 'ms://Qwen/Qwen3-30B-A3B-Instruct-2507'
base_url='http://www.modelscope.cn/twinkle'
api_key=os.environ.get('MODELSCOPE_TOKEN')

# Use twinkle dataset to load the data
dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(500)))
dataset.set_template('Template', model_id=base_model, max_length=256)
dataset.map(SelfCognitionProcessor('Twinkle Model', 'ModelScope Team'), load_from_cache_file=False)
dataset.encode(batched=True, load_from_cache_file=False)
dataloader = DataLoader(dataset=dataset, batch_size=8)

# Initialize Tinker client before importing ServiceClient
init_tinker_client()
from tinker import ServiceClient

service_client = ServiceClient(base_url=base_url, api_key=api_key)
training_client = service_client.create_lora_training_client(base_model=base_model[len('ms://'):], rank=16)

# Training loop: use input_feature_to_datum to transfer the input format
for epoch in range(3):
    for step, batch in tqdm(enumerate(dataloader)):
        input_datum = [input_feature_to_datum(input_feature) for input_feature in batch]

        fwdbwd_future = training_client.forward_backward(input_datum, "cross_entropy")
        optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))

        fwdbwd_result = fwdbwd_future.result()
        optim_result = optim_future.result()

    training_client.save_state(f"twinkle-lora-{epoch}").result()
```

> The ModelScope server is tinker-compatible, so use the tinker cookbooks. In the future version, we will support a server works both for twinkle/tinker clients.

Developers can customize datasets, advantage functions, rewards, templates, and more. However, the Loss component is not currently customizable since it needs to be executed on the server side (for security reasons). If you need support for additional Loss functions, you can upload your Loss implementation to ModelHub and contact us via the Q&A group or through an issue to have the corresponding component added to the whitelist.

## Appendix: Supported Training Methods

This model is a text-only model, so multimodal tasks are not currently supported. For text-only tasks, you can train using:

1. Standard PT/SFT training methods, including Agentic training
2. Self-sampling RL algorithms such as GRPO/RLOO
3. Distillation methods like GKD/On-policy. Since the official ModelScope endpoint only supports a single model, the other Teacher/Student model must be prepared by the developer

The current official environment only supports LoRA training, with the following requirements:

1. Maximum rank = 32
2. modules_to_save is not supported
