#%%
from tinker import types
from twinkle_client import init_tinker_compat_client
from modelscope import AutoTokenizer

base_model = "Qwen/Qwen2.5-0.5B-Instruct"
service_client = init_tinker_compat_client(base_url='http://localhost:8000', api_key="tml-EMPTY_TOKEN")

sampling_client = service_client.create_sampling_client(
    model_path="twinkle://20260130_133245-Qwen_Qwen2_5-0_5B-Instruct-ffebd239/weights/pig-latin-lora-epoch-1",
    base_model=base_model)


print(f"Using model {base_model}")
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
# First, create a sampling client. We need to transfer weights

# Now, we can sample from the model.
prompt = types.ModelInput.from_ints(tokenizer.encode("English: coffee break\nPig Latin:"))
params = types.SamplingParams(max_tokens=20, temperature=0.0, stop=["\n"]) # Greedy sampling

print("Sampling...")
future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=8)
result = future.result()
print("Responses:")
for i, seq in enumerate(result.sequences):
    print(f"{i}: {repr(tokenizer.decode(seq.tokens))}")
