import numpy as np
from tqdm import tqdm
from tinker import types
from twinkle_client import init_tinker_compat_client
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.preprocessor import SelfCognitionProcessor
from twinkle.server.tinker.common import input_feature_to_datum
from modelscope import AutoTokenizer

base_model = "Qwen/Qwen2.5-7B-Instruct"

def train():
    # process data
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(500)))
    dataset.set_template('Template', model_id=f'ms://{base_model}', max_length=256)
    dataset.map(SelfCognitionProcessor('twinkle模型', 'twinkle团队'), load_from_cache_file=False)
    dataset.encode(batched=True, load_from_cache_file=False)
    dataloader = DataLoader(dataset=dataset, batch_size=8)

    # init service client
    service_client = init_tinker_compat_client(base_url='http://localhost:8000')
    training_client = service_client.create_lora_training_client(
        base_model=base_model,
        rank=16
    )

    for epoch in range(3):
        print(f"Epoch {epoch}")
        for step, batch in tqdm(enumerate(dataloader)):
            input_datum = [input_feature_to_datum(input_feature) for input_feature in batch]
            fwdbwd_future = training_client.forward_backward(input_datum, "cross_entropy")
            optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))

            # Wait for the results
            fwdbwd_result = fwdbwd_future.result()
            optim_result = optim_future.result()

            # fwdbwd_result contains the logprobs of all the tokens we put in. Now we can compute the weighted
            logprobs = np.concatenate([output['logprobs'].tolist() for output in fwdbwd_result.loss_fn_outputs])
            weights = np.concatenate([example.loss_fn_inputs['weights'].tolist() for example in input_datum])
            print(f"Loss per token: {-np.dot(logprobs, weights) / weights.sum():.4f}")

        save_future = training_client.save_state(f"twinkle-lora-{epoch}")
        save_result = save_future.result()
        print(f"Saved checkpoint to {save_result.path}")

def eval():
    weight_path = "twinkle://20260203_194633-Qwen_Qwen2_5-0_5B-Instruct-03aa3f06/weights/twinkle-lora"

    service_client = init_tinker_compat_client(base_url='http://localhost:8000')
    sampling_client = service_client.create_sampling_client(
        model_path=weight_path,
        base_model=base_model)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    inputs = [
        {
            'role': 'user',
            'content': 'what is your name?'
        }
    ]
    input_ids = tokenizer.apply_chat_template(
        inputs,
        tokenize=True,
        add_generation_prompt=True  # usually needed for chat models
    )
    # Now, we can sample from the model.
    prompt = types.ModelInput.from_ints(input_ids)
    params = types.SamplingParams(max_tokens=50, temperature=0.2, stop=["\n"])

    print("Sampling...")
    future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=8)
    result = future.result()
    print("Responses:")
    for i, seq in enumerate(result.sequences):
        print(f"{i}: {repr(tokenizer.decode(seq.tokens))}")

if __name__ == "__main__":
    train()
    # eval()
