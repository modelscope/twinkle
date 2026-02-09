# Twinkle Client - Sampler (Inference) Example
#
# This script demonstrates how to run text generation inference
# through the Twinkle client-server architecture.
# The server must be running first (see server.py and server_config.yaml).
#
# This is the client/server equivalent of cookbook/legacy/sampler/sampler_demo.py.
# Instead of running everything locally, the sampler runs on the server side
# while the client sends requests over HTTP.

# Step 1: Load environment variables from a .env file (e.g., API tokens)
import dotenv
dotenv.load_dotenv('.env')

import os
from transformers import AutoTokenizer

from twinkle import get_logger
from twinkle_client import init_twinkle_client
from twinkle_client.sampler import vLLMSampler

logger = get_logger()

MODEL_ID = 'Qwen/Qwen2.5-7B-Instruct'

# Optional: adapter URI for LoRA inference
# This can be a twinkle:// path from a training run checkpoint
# or None to use the base model
# ADAPTER_URI = None
# Example:
ADAPTER_URI = "twinkle://20260208_224851-fa3cdd11-default/weights/twinkle-epoch-2"


def sample():
    # Step 2: Initialize the Twinkle client to communicate with the remote server.
    client = init_twinkle_client(
        base_url='http://127.0.0.1:8000',
        api_key=os.environ.get('MODELSCOPE_SDK_TOKEN'),
    )

    # Step 3: Create the sampler client pointing to the model on the server
    sampler = vLLMSampler(model_id=MODEL_ID)

    # Step 4: Set the chat template so the sampler can encode Trajectory inputs
    sampler.set_template('Template', model_id=MODEL_ID)

    # Step 5: Prepare inputs as Trajectory dicts (messages format)
    # Each trajectory is a conversation with system and user messages
    trajectory = {
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'Who are you?'},
        ]
    }

    num_prompts = 4
    num_samples = 2  # Generate 2 completions per prompt

    # Step 6: Configure sampling parameters
    sampling_params = {
        'max_tokens': 128,
        'temperature': 1.0,
    }

    # Step 7: Call the sampler
    # - inputs: list of Trajectory dicts (will be encoded server-side using the template)
    # - sampling_params: controls generation behavior
    # - adapter_uri: optional LoRA adapter path for fine-tuned inference
    # - num_samples: number of completions per prompt
    response = sampler.sample(
        inputs=[trajectory] * num_prompts,
        sampling_params=sampling_params,
        adapter_uri=ADAPTER_URI,
        num_samples=num_samples,
    )

    # Step 8: Decode and print the results
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    logger.info(f"Generated {len(response['sequences'])} sequences "
                f"({num_prompts} prompts x {num_samples} samples)")

    for i, seq in enumerate(response['sequences']):
        text = tokenizer.decode(seq['tokens'], skip_special_tokens=True)
        logger.info(f"Sequence {i}:\n  {text}\n")


if __name__ == '__main__':
    sample()
