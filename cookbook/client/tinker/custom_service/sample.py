# Tinker-Compatible Client - Sampling / Inference Example
#
# This script demonstrates how to use a previously trained LoRA checkpoint
# for text generation (sampling) via the Tinker-compatible client API.
# The server must be running first (see server.py and server_config.yaml).

import os
from tinker import types

from twinkle.data_format import Message, Trajectory
from twinkle.template import Template
from twinkle import init_tinker_client

# Step 1: Initialize Tinker client
init_tinker_client()

from tinker import ServiceClient

# Step 2: Define the base model and connect to the server
base_model = 'Qwen/Qwen3-4B'
service_client = ServiceClient(
    base_url='http://localhost:8000',
    api_key='EMPTY-TOKEN'
)

# Step 3: Create a sampling client by loading weights from a saved checkpoint.
# The model_path is a twinkle:// URI pointing to a previously saved LoRA checkpoint.
# The server will load the base model and apply the LoRA adapter weights.
sampling_client = service_client.create_sampling_client(
    model_path='twinkle://xxx-Qwen_Qwen3-30B-A3B-Instruct-2507-xxx/weights/twinkle-lora-1',
    base_model=base_model
)

# Step 4: Load the tokenizer locally to encode the prompt and decode the results
print(f'Using model {base_model}')

template = Template(model_id=f'ms://{base_model}')

trajectory = Trajectory(
    messages=[
        Message(role='system', content='You are a helpful assistant'),
        Message(role='user', content='你是谁？'),
    ]
)

input_feature = template.encode(trajectory, add_generation_prompt=True)

input_ids = input_feature['input_ids'].tolist()

# Step 5: Prepare the prompt and sampling parameters
prompt = types.ModelInput.from_ints(input_ids)
params = types.SamplingParams(
    max_tokens=128,       # Maximum number of tokens to generate
    temperature=0.7,
    stop=['\n']          # Stop generation when a newline character is produced
)

# Step 6: Send the sampling request to the server.
# num_samples=8 generates 8 independent completions for the same prompt.
print('Sampling...')
future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
result = future.result()

# Step 7: Decode and print the generated responses
print('Responses:')
for i, seq in enumerate(result.sequences):
    print(f'{i}: {repr(template.decode(seq.tokens))}')
