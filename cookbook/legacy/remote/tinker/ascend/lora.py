#%%
import os
import dotenv
dotenv.load_dotenv(".env")

from twinkle.utils.import_utils import requires
from twinkle_client import init_tinker_compat_client
timeout_s = float(os.environ.get("TWINKLE_HTTP_TIMEOUT", "180"))
service_client = init_tinker_compat_client(base_url=os.environ['EAS_URL'], timeout=timeout_s)

print("Available models:")
for item in service_client.get_server_capabilities().supported_models:
    print("- " + item.model_name)


#%%
rest_client = service_client.create_rest_client()

resume_path = ""
if os.environ.get("TWINKLE_SKIP_HISTORY", "0") != "1":
    future = rest_client.list_training_runs(limit=50)
    response = future.result()
    print(f"Found {len(response.training_runs)} training runs")
    for tr in response.training_runs:
        print(tr.model_dump_json(indent=2))
        
        chpts = rest_client.list_checkpoints(tr.training_run_id).result()
        for chpt in chpts.checkpoints:
            print("  " + chpt.model_dump_json(indent=2))
            resume_path = chpt.tinker_path  # Just get the last one for demo purposes
else:
    pass  # Skip training run history
    
#%%
if not resume_path:
    base_model = "Qwen/Qwen3-0.6B"
    training_client = service_client.create_lora_training_client(
        base_model=base_model
    )
else:
    training_client = service_client.create_training_client_from_state_with_optimizer(path=resume_path)

#%%
# Create some training examples
examples = [
    {"input": "banana split", "output": "anana-bay plit-say"},
    {"input": "quantum physics", "output": "uantum-qay ysics-phay"},
    {"input": "donut shop", "output": "onut-day op-shay"},
    {"input": "pickle jar", "output": "ickle-pay ar-jay"},
    {"input": "space exploration", "output": "ace-spay exploration-way"},
    {"input": "rubber duck", "output": "ubber-ray uck-day"},
    {"input": "coding wizard", "output": "oding-cay izard-way"},
]

max_examples = int(os.environ.get("TWINKLE_DEMO_EXAMPLES", "0"))
if max_examples > 0:
    examples = examples[:max_examples]
 
# Convert examples into the format expected by the training client
from tinker import types
# Get the tokenizer from the training client
# tokenizer = training_client.get_tokenizer() # NOTE: network call huggingface
def load_tokenizer(
    model_id: str,
    *,
    local_path: str | None = None,
    local_path_env: str = "TWINKLE_MODEL_PATH",
    trust_remote_code: bool = True,
):
    requires("modelscope")
    from modelscope import AutoTokenizer

    if local_path is None:
        local_path = os.environ.get(local_path_env)

    if local_path and os.path.isdir(local_path):
        return AutoTokenizer.from_pretrained(local_path, trust_remote_code=trust_remote_code)

    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)

tokenizer = load_tokenizer(base_model)
 
def process_example(example: dict, tokenizer) -> types.Datum:
    # Format the input with Input/Output template
    # For most real use cases, you'll want to use a renderer / chat template,
    # (see later docs) but here, we'll keep it simple.
    prompt = f"English: {example['input']}\nPig Latin:"
 
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)
    # Add a space before the output string, and finish with double newline
    completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)
    completion_weights = [1] * len(completion_tokens)
 
    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights
 
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:] # We're predicting the next token, so targets need to be shifted.
    weights = weights[1:]
 
    # A datum is a single training example for the loss function.
    # It has model_input, which is the input sequence that'll be passed into the LLM,
    # loss_fn_inputs, which is a dictionary of extra inputs used by the loss function.
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
    )
 
processed_examples = [process_example(ex, tokenizer) for ex in examples]
 
#%%
import numpy as np
skip_train = os.environ.get("TWINKLE_SKIP_TRAIN", "0") == "1"
epochs = int(os.environ.get("TWINKLE_DEMO_EPOCHS", 2))
batches = int(os.environ.get("TWINKLE_DEMO_BATCHES", 5))
if not skip_train:
    for epoch in range(epochs):
        for batch in range(batches):
            fwdbwd_future = training_client.forward_backward(processed_examples, "cross_entropy")
            optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))
        
            # Wait for the results
            fwdbwd_result = fwdbwd_future.result()
            optim_result = optim_future.result()
        
            # fwdbwd_result contains the logprobs of all the tokens we put in. Now we can compute the weighted
            # average log loss per token for progress visibility.
            logprobs = np.concatenate([output['logprobs'].tolist() for output in fwdbwd_result.loss_fn_outputs])
            weights = np.concatenate([example.loss_fn_inputs['weights'].tolist() for example in processed_examples])
            print(f"Epoch {epoch}, Batch {batch}: Loss per token: {-np.dot(logprobs, weights) / weights.sum():.4f}")

        save_future = training_client.save_state(f"pig-latin-lora-epoch-{epoch}")
        save_result = save_future.result()
        print(f"Saved checkpoint for epoch {epoch} to {save_result.path}")

#%%
# First, create a sampling client. We need to transfer weights
sampling_client = training_client.save_weights_and_get_sampling_client(name='pig-latin-model')
 
# Now, we can sample from the model.
prompt = types.ModelInput.from_ints(tokenizer.encode("English: coffee break\nPig Latin:"))
params = types.SamplingParams(max_tokens=20, temperature=0.0, stop=["\n"]) # Greedy sampling
future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=8)
result = future.result()
print("Responses:")
for i, seq in enumerate(result.sequences):
    print(f"{i}: {repr(tokenizer.decode(seq.tokens))}")
# %%
