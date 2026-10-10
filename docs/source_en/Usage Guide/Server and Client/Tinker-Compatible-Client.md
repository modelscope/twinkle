# Tinker Client

The Tinker Client is suitable for scenarios with existing Tinker training code. After initializing with `init_tinker_client`, it patches the Tinker SDK to point to the Twinkle Server. Check the loss compatibility limits below before reusing a training loop.

## Loss compatibility

The bridge accepts `cross_entropy` and `importance_sampling`. `ppo`, `cispo`, and `dro` are not implemented by this bridge and return HTTP 400 before enqueueing or changing gradients; they are never replaced with cross-entropy.

For compatibility with existing Twinkle cookbooks, `importance_sampling` retains Twinkle's **GRPO loss**: PPO-style clipping (`epsilon=0.2` by default), optional KL penalty (`beta=0` by default), and a mean over sequences. This differs from Tinker's [unclipped, token-summed IS objective](https://tinker-docs.thinkingmachines.ai/tinker/losses/importance-sampling/). It is not an exact replacement for Tinker IS.

Twinkle's DPO extension also uses `importance_sampling`, with `ref_logps` on **every** datum and interleaved chosen/rejected examples. Each data-parallel shard must receive complete pairs, including in single-rank training. Ordinary RL examples do not need pairs: batches of 1 or 3 are valid with one data rank, subject to the usual token and rate limits.

## Sampler checkpoint lifetime

`save_weights_for_sampler(name="step-1")` preserves a named checkpoint at
`twinkle://<run>/sampler_weights/step-1`. Other named or unnamed saves do not delete it.
Saving the same name replaces its weights; new sampling requests after the save completes
reload that version, including sampling clients already using the same path.
`latest` is reserved for unnamed saves. Names retain the existing filesystem normalization.

Unnamed saves, including `save_weights_and_get_sampling_client()`, retain only the latest
weights. Their generated paths are temporary and may expire on the next unnamed save.
Use explicit names to retain evaluation snapshots and the existing checkpoint deletion API
to release their disk space. Keep calling `init_tinker_client()`; unpatched SDK path parsing
and checkpoint archive downloads are outside this compatibility bridge.

## Initialization

```python
# Initialize Tinker client before importing ServiceClient
from twinkle import init_tinker_client

init_tinker_client()

# Use ServiceClient directly from tinker
from tinker import ServiceClient

service_client = ServiceClient(
    base_url='http://localhost:8000',                    # Server address
    api_key=os.environ.get('MODELSCOPE_TOKEN')           # Recommended: set to ModelScope Token
)

# Verify connection: List available models on Server
for item in service_client.get_server_capabilities().supported_models:
    print("- " + item.model_name)
```

### What does init_tinker_client do?

When calling `init_tinker_client`, the following operations are automatically executed:

1. **Patch Tinker SDK**: Bypass Tinker's `tinker://` prefix validation, allowing it to connect to standard HTTP addresses
2. **Set Request Headers**: Inject necessary authentication headers such as `X-Ray-Serve-Request-Id` and `Authorization`

After initialization, import `from tinker import ServiceClient` to connect to Twinkle Server. Training code can reuse the supported endpoints subject to the loss compatibility limits above.

## Complete Training Example

> **Note**: `DataLoader` and `Dataset` in Tinker compatible mode only support local `twinkle` imports; `twinkle_client` is not supported.

```python
import os
import numpy as np
from tqdm import tqdm
from tinker import types
from twinkle import init_tinker_client
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.preprocessor import SelfCognitionProcessor
from twinkle.server.model.tinker_datum import input_feature_to_datum

# Step 1: Initialize Tinker client before importing ServiceClient
init_tinker_client()

from tinker import ServiceClient

base_model = 'Qwen/Qwen3.5-4B'
base_url = 'http://localhost:8000'
api_key = 'EMPTY_API_KEY'

# Step 2: Prepare dataset
dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(500)))
dataset.set_template('Qwen3_5Template', model_id=f'ms://{base_model}', max_length=256)
dataset.map(SelfCognitionProcessor('twinkle model', 'ModelScope Team'), load_from_cache_file=False)
dataset.encode(batched=True, load_from_cache_file=False)
dataloader = DataLoader(dataset=dataset, batch_size=8)

# Step 3: Initialize training client
service_client = ServiceClient(base_url=base_url, api_key=api_key)

# Create LoRA training client (rank=16 specifies the LoRA adapter rank)
training_client = service_client.create_lora_training_client(base_model=base_model, rank=16)

# Step 4: Training loop
for epoch in range(3):
    print(f'Epoch {epoch}')
    for step, batch in tqdm(enumerate(dataloader)):
        # Convert Twinkle's InputFeature to Tinker's Datum format
        input_datum = [input_feature_to_datum(input_feature) for input_feature in batch]

        # Send data to Server: forward + backward propagation
        fwdbwd_future = training_client.forward_backward(input_datum, 'cross_entropy')

        # Optimizer step: update model weights with Adam
        optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))

        # Wait for both operations to complete
        fwdbwd_result = fwdbwd_future.result()
        optim_result = optim_future.result()

        # Compute weighted average log-loss per token for monitoring
        logprobs = np.concatenate([output['logprobs'].tolist() for output in fwdbwd_result.loss_fn_outputs])
        weights = np.concatenate([example.loss_fn_inputs['weights'].tolist() for example in input_datum])
        print(f'Loss per token: {-np.dot(logprobs, weights) / weights.sum():.4f}')
        print(f'Training Metrics: {optim_result}')

    # Save a checkpoint after each epoch
    save_future = training_client.save_state(f'twinkle-lora-{epoch}')
    save_result = save_future.result()
    print(f'Saved checkpoint to {save_result.path}')
```

## Inference Sampling

Tinker compatible mode supports inference sampling functionality (Server needs to have Sampler service configured).

### Sampling from Training

After training is complete, you can directly create a sampling client from the training client:

```python
# Save current weights and create sampling client
sampling_client = training_client.save_weights_and_get_sampling_client(name='my-model')

# Prepare inference input
prompt = types.ModelInput.from_ints(tokenizer.encode("English: coffee break\nPig Latin:"))
params = types.SamplingParams(
    max_tokens=20,       # Maximum number of tokens to generate
    temperature=0.0,     # Greedy sampling (deterministic output)
    stop=["\n"]          # Stop when encountering newline
)

# Generate multiple completions
result = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=8).result()

for i, seq in enumerate(result.sequences):
    print(f"{i}: {tokenizer.decode(seq.tokens)}")
```

### Sampling from Checkpoint

You can also load saved checkpoints for inference:

```python
import os
from tinker import types
from twinkle import init_tinker_client
from twinkle.data_format import Message, Trajectory
from twinkle.template import Template

# Initialize Tinker client before importing ServiceClient
init_tinker_client()

from tinker import ServiceClient

base_model = 'Qwen/Qwen3.5-4B'
base_url = 'http://localhost:8000'
api_key = 'EMPTY_API_KEY'

service_client = ServiceClient(base_url=base_url, api_key=api_key)

# Create sampling client from saved checkpoint
sampling_client = service_client.create_sampling_client(
    model_path='twinkle://run_id/weights/checkpoint_name',  # twinkle:// path of the checkpoint
    base_model=base_model
)

# Use Twinkle's Template to build multi-turn dialogue input
template = Template(model_id=f'ms://{base_model}')

trajectory = Trajectory(
    messages=[
        Message(role='system', content='You are a helpful assistant'),
        Message(role='user', content='What is your name?'),
    ]
)

input_feature = template.batch_encode([trajectory], add_generation_prompt=True)[0]
input_ids = input_feature['input_ids'].tolist()

prompt = types.ModelInput.from_ints(input_ids)
params = types.SamplingParams(
    max_tokens=50,       # Maximum number of tokens to generate
    temperature=0.2,     # Low temperature, more focused answers
)

# Generate multiple completions
print('Sampling...')
future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=8)
result = future.result()

# Decode and print each response
print('Responses:')
for i, seq in enumerate(result.sequences):
    print(f'{i}: {repr(template.decode(seq.tokens))}')
```

### Publishing Checkpoint to ModelScope Hub

After training is complete, you can publish checkpoints to ModelScope Hub through the REST client:

```python
rest_client = service_client.create_rest_client()

# Publish checkpoint from tinker path
# Need to set a valid ModelScope token as api_key when initializing the client
rest_client.publish_checkpoint_from_tinker_path(save_result.path).result()
print("Published checkpoint to ModelScope Hub")
```
