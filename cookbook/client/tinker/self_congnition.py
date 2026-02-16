# Tinker-Compatible Client - Self-Cognition Training & Evaluation Example
#
# This script demonstrates two workflows using the Tinker-compatible client:
#   1. train(): Fine-tune a model on a self-cognition dataset so it learns
#      a custom identity (name, author).
#   2. eval():  Load a trained checkpoint and sample from it to verify
#      that the model has learned the custom identity.
# The server must be running first (see server.py and server_config.yaml).
import numpy as np
import os
from tqdm import tqdm
from tinker import types
from twinkle_client import init_tinker_compat_client
from twinkle.data_format import Message, Trajectory
from twinkle.template import Template
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.preprocessor import SelfCognitionProcessor
from twinkle.server.tinker.common import input_feature_to_datum

# The base model to fine-tune / evaluate
base_model = 'Qwen/Qwen3-30B-A3B-Instruct-2507'


def train():
    # Step 1: Prepare the dataset

    # Load the self-cognition dataset from ModelScope (first 500 examples)
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(500)))

    # Apply the chat template matching the base model (max 256 tokens per sample)
    dataset.set_template('Template', model_id=f'ms://{base_model}', max_length=256)

    # Replace placeholder names with custom model/author identity
    dataset.map(SelfCognitionProcessor('twinkle模型', 'twinkle团队'), load_from_cache_file=False)

    # Tokenize and encode the dataset into model-ready input features
    dataset.encode(batched=True, load_from_cache_file=False)

    # Wrap the dataset into a DataLoader that yields batches of size 8
    dataloader = DataLoader(dataset=dataset, batch_size=8)

    # Step 2: Initialize the training client

    # Connect to the Twinkle server running locally
    service_client = init_tinker_compat_client(
        base_url='localhost:9000', api_key=os.environ.get('MODELSCOPE_TOKEN'))

    # Create a LoRA training client for the base model (rank=16 for the LoRA adapter)
    training_client = service_client.create_lora_training_client(base_model=base_model, rank=16)

    # Step 3: Run the training loop

    for epoch in range(3):
        print(f'Epoch {epoch}')
        for step, batch in tqdm(enumerate(dataloader)):
            # Convert each InputFeature into a Datum for the Tinker API
            input_datum = [input_feature_to_datum(input_feature) for input_feature in batch]

            # Send data to server: forward + backward pass (computes gradients)
            fwdbwd_future = training_client.forward_backward(input_datum, 'cross_entropy')

            # Optimizer step: update model weights with Adam
            optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))

            # Wait for both operations to complete
            fwdbwd_result = fwdbwd_future.result()
            optim_result = optim_future.result()

            # Compute weighted average log-loss per token for monitoring
            # logprobs = np.concatenate([output['logprobs'].tolist() for output in fwdbwd_result.loss_fn_outputs])
            # weights = np.concatenate([example.loss_fn_inputs['weights'].tolist() for example in input_datum])
            # print(f'Loss per token: {-np.dot(logprobs, weights) / weights.sum():.4f}')
            print(f'Training Metrics: {optim_result}')

        # Save a checkpoint after each epoch
        save_future = training_client.save_state(f'twinkle-lora-{epoch}')
        save_result = save_future.result()
        print(f'Saved checkpoint to {save_result.path}')


def eval():
    # Step 1: Load the trained LoRA checkpoint for inference

    # Path to a previously saved LoRA checkpoint (twinkle:// URI)
    weight_path = 'twinkle://20260212_174205-Qwen_Qwen2_5-7B-Instruct-51edc9ed/weights/twinkle-lora-2'

    # Connect to the server and create a sampling client with the trained weights
    service_client = init_tinker_compat_client(base_url='http://localhost:9000')
    sampling_client = service_client.create_sampling_client(model_path=weight_path, base_model=base_model)

    # Step 2: Prepare the chat prompt

    # Build a multi-turn conversation to test the model's self-cognition
    template = Template(model_id=f'ms://{base_model}')

    trajectory = Trajectory(
        messages=[
            Message(role='system', content='You are a helpful assistant'),
            Message(role='user', content='你是谁？'),
        ]
    )

    input_feature = template.encode(trajectory, add_generation_prompt=True)

    input_ids = input_feature['input_ids'].tolist()

    # Step 3: Generate responses

    prompt = types.ModelInput.from_ints(input_ids)
    params = types.SamplingParams(
        max_tokens=50,  # Maximum tokens to generate
        temperature=0.2,  # Low temperature for more focused responses
        stop=['\n']  # Stop at newline
    )

    # Sample 8 independent completions
    print('Sampling...')
    future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=8)
    result = future.result()

    # Decode and print each response
    print('Responses:')
    for i, seq in enumerate(result.sequences):
        print(f'{i}: {repr(template.decode(seq.tokens))}')


if __name__ == '__main__':
    train()   # Uncomment to run training
    # eval()      # Run evaluation / inference
