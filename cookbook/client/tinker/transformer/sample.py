# Tinker-Compatible Client - Sampling / Inference Example
#
# This script demonstrates how to use a previously trained LoRA checkpoint
# for text generation (sampling) via the Tinker-compatible client API.
# The server must be running first (see server.py and server_config.yaml).

from tinker import types
from twinkle_client import init_tinker_compat_client
from modelscope import AutoTokenizer

# Step 1: Define the base model and connect to the server
base_model = "Qwen/Qwen2.5-0.5B-Instruct"
service_client = init_tinker_compat_client(base_url='http://localhost:8000', api_key="tml-EMPTY_TOKEN")

# Step 2: Create a sampling client by loading weights from a saved checkpoint.
# The model_path is a twinkle:// URI pointing to a previously saved LoRA checkpoint.
# The server will load the base model and apply the LoRA adapter weights.
sampling_client = service_client.create_sampling_client(
    model_path="twinkle://20260130_133245-Qwen_Qwen2_5-0_5B-Instruct-ffebd239/weights/pig-latin-lora-epoch-1",
    base_model=base_model)

# Step 3: Load the tokenizer locally to encode the prompt and decode the results
print(f"Using model {base_model}")
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

# Step 4: Prepare the prompt and sampling parameters
prompt = types.ModelInput.from_ints(tokenizer.encode("English: coffee break\nPig Latin:"))
params = types.SamplingParams(
    max_tokens=20,       # Maximum number of tokens to generate
    temperature=0.0,     # Greedy sampling (deterministic, always pick the top token)
    stop=["\n"]          # Stop generation when a newline character is produced
)

# Step 5: Send the sampling request to the server.
# num_samples=8 generates 8 independent completions for the same prompt.
print("Sampling...")
future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=8)
result = future.result()

# Step 6: Decode and print the generated responses
print("Responses:")
for i, seq in enumerate(result.sequences):
    print(f"{i}: {repr(tokenizer.decode(seq.tokens))}")
