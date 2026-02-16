# Tinker Compatible Client

The Tinker Compatible Client is suitable for scenarios with existing Tinker training code. After initializing with `init_tinker_compat_client`, it patches the Tinker SDK to point to the Twinkle Server, **and the rest of the code can directly reuse existing Tinker training code**.

## Initialization

```python
from twinkle_client import init_tinker_compat_client

# Initialize Tinker compatible client
# init_tinker_compat_client automatically patches the Tinker SDK,
# allowing it to connect to Twinkle Server instead of Tinker Server
service_client = init_tinker_compat_client(
    base_url='http://localhost:8000',    # Server address
    api_key='your-api-key'               # Authentication token
)

# Verify connection: List available models on Server
for item in service_client.get_server_capabilities().supported_models:
    print("- " + item.model_name)
```

### What does init_tinker_compat_client do?

When calling `init_tinker_compat_client`, the following operations are automatically executed:

1. **Patch Tinker SDK**: Bypass Tinker's `tinker://` prefix validation, allowing it to connect to standard HTTP addresses
2. **Set Request Headers**: Inject necessary authentication headers such as `serve_multiplexed_model_id` and `Authorization`
3. **Return `ServiceClient`**: Returns a standard Tinker `ServiceClient` object, subsequent operations are completely identical to native Tinker

This means that after initialization, **all existing Tinker training code can be used directly** without any modifications.

## Complete Training Example

```python
import os
import numpy as np
import dotenv
dotenv.load_dotenv('.env')

from tinker import types
from modelscope import AutoTokenizer
from twinkle_client import init_tinker_compat_client

# Step 1: Initialize client (automatically patches Tinker SDK)
service_client = init_tinker_compat_client(
    base_url='http://localhost:8000',
    api_key=os.environ.get('MODELSCOPE_TOKEN')
)

# Step 2: Query existing training runs (optional)
rest_client = service_client.create_rest_client()
response = rest_client.list_training_runs(limit=50).result()
print(f"Found {len(response.training_runs)} training runs")

# Step 3: Create training client
base_model = "Qwen/Qwen2.5-0.5B-Instruct"

# Create new training session
training_client = service_client.create_lora_training_client(
    base_model=base_model
)

# Or resume from checkpoint
# resume_path = "twinkle://run_id/weights/checkpoint_name"
# training_client = service_client.create_training_client_from_state_with_optimizer(path=resume_path)

# Step 4: Prepare training data
examples = [
    {"input": "banana split", "output": "anana-bay plit-say"},
    {"input": "quantum physics", "output": "uantum-qay ysics-phay"},
    {"input": "donut shop", "output": "onut-day op-shay"},
]

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

def process_example(example: dict, tokenizer) -> types.Datum:
    """Convert raw sample to Datum format required by Tinker API.

    Datum contains:
      - model_input: Input token IDs
      - loss_fn_inputs: Target tokens and per-token weights (0=ignore, 1=compute loss)
    """
    prompt = f"English: {example['input']}\nPig Latin:"

    # Prompt part: weight=0, does not participate in loss computation
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)

    # Completion part: weight=1, participates in loss computation
    completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)
    completion_weights = [1] * len(completion_tokens)

    # Concatenate and construct next-token prediction format
    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
    )

processed_examples = [process_example(ex, tokenizer) for ex in examples]

# Step 5: Training loop
for epoch in range(2):
    for batch in range(5):
        # Send training data to Server: forward + backward propagation
        fwdbwd_future = training_client.forward_backward(processed_examples, "cross_entropy")
        # Optimizer update
        optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))

        # Wait for results
        fwdbwd_result = fwdbwd_future.result()
        optim_result = optim_future.result()

        # Calculate weighted average log-loss
        logprobs = np.concatenate([o['logprobs'].tolist() for o in fwdbwd_result.loss_fn_outputs])
        weights = np.concatenate([e.loss_fn_inputs['weights'].tolist() for e in processed_examples])
        print(f"Epoch {epoch}, Batch {batch}: Loss = {-np.dot(logprobs, weights) / weights.sum():.4f}")

    # Save checkpoint every epoch
    save_result = training_client.save_state(f"lora-epoch-{epoch}").result()
    print(f"Saved checkpoint to {save_result.path}")
```

## Using Twinkle Dataset Components

Tinker compatible mode can also leverage Twinkle's dataset components to simplify data preparation instead of manually constructing `Datum`:

```python
from tqdm import tqdm
from tinker import types
from twinkle_client import init_tinker_compat_client
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.preprocessor import SelfCognitionProcessor
from twinkle.server.tinker.common import input_feature_to_datum

base_model = "Qwen/Qwen2.5-0.5B-Instruct"

# Use Twinkle's Dataset component to load and preprocess data
dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(500)))
dataset.set_template('Template', model_id=f'ms://{base_model}', max_length=256)
dataset.map(SelfCognitionProcessor('twinkle model', 'twinkle team'), load_from_cache_file=False)
dataset.encode(batched=True, load_from_cache_file=False)
dataloader = DataLoader(dataset=dataset, batch_size=8)

# Initialize Tinker compatible client
service_client = init_tinker_compat_client(base_url='http://localhost:8000')
training_client = service_client.create_lora_training_client(base_model=base_model, rank=16)

# Training loop: Use input_feature_to_datum to convert data format
for epoch in range(3):
    for step, batch in tqdm(enumerate(dataloader)):
        # Convert Twinkle's InputFeature to Tinker's Datum
        input_datum = [input_feature_to_datum(input_feature) for input_feature in batch]

        fwdbwd_future = training_client.forward_backward(input_datum, "cross_entropy")
        optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))

        fwdbwd_result = fwdbwd_future.result()
        optim_result = optim_future.result()

    training_client.save_state(f"twinkle-lora-{epoch}").result()
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
from tinker import types
from modelscope import AutoTokenizer
from twinkle_client import init_tinker_compat_client

base_model = "Qwen/Qwen2.5-0.5B-Instruct"

# Initialize client
service_client = init_tinker_compat_client(base_url='http://localhost:8000')

# Create sampling client from saved checkpoint
sampling_client = service_client.create_sampling_client(
    model_path="twinkle://run_id/weights/checkpoint_name",  # twinkle:// path of the checkpoint
    base_model=base_model
)

# Prepare inference input
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

# Construct multi-turn dialogue input
inputs = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': 'what is your name?'}
]
input_ids = tokenizer.apply_chat_template(inputs, tokenize=True, add_generation_prompt=True)

prompt = types.ModelInput.from_ints(input_ids)
params = types.SamplingParams(
    max_tokens=50,       # Maximum number of tokens to generate
    temperature=0.2,     # Low temperature, more focused answers
    stop=["\n"]          # Stop when encountering newline
)

# Generate multiple completions
result = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=8).result()

for i, seq in enumerate(result.sequences):
    print(f"{i}: {tokenizer.decode(seq.tokens)}")
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
