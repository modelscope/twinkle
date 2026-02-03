import twinkle
from twinkle import DeviceGroup, DeviceMesh
from transformers import AutoTokenizer
from twinkle.sampler import VLLMSampler
from twinkle.sampler.types import SamplingParams
from twinkle.template import Template
from twinkle.data_format import Trajectory

MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
VLLM_TP = 2  # Tensor parallelism for vLLM (GPUs per worker)
VLLM_DP = 2  # Data parallelism (number of workers)
NUM_GPUS = VLLM_TP * VLLM_DP  # Total GPUs = 8

if __name__ == '__main__':
    twinkle.initialize(
        mode='ray',
        nproc_per_node=NUM_GPUS,
        groups=[
            # gpus_per_worker=VLLM_TP creates 4 workers (8 GPUs / 2 per worker)
            DeviceGroup(name='sampler', ranks=list(range(NUM_GPUS)), device_type='GPU', gpus_per_worker=VLLM_TP),
        ],
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    engine_kwargs = {
        'gpu_memory_utilization': 0.4,
        'max_model_len': 1024,
        # 'enforce_eager': True,
        # 'enable_sleep_mode': True,
    }
    sampler = VLLMSampler(
        model_id=MODEL_ID, 
        engine_args=engine_kwargs,
        device_mesh=DeviceMesh.from_sizes(world_size=VLLM_DP, dp_size=VLLM_DP),
        remote_group='sampler',
    )
    sampler.set_template(Template, model_id=MODEL_ID)
    trajectory = Trajectory(messages=[{'role': 'user', 'content': 'Hello! Tell a joke.'}])
    
    num_prompts = 4
    num_samples = 2  # Generate 2 completions per prompt
    sampling_params = SamplingParams(max_tokens=128, temperature=1.0)
    
    # Pass num_samples to sample() method (aligned with tinker's API)
    response = sampler.sample([trajectory] * num_prompts, sampling_params, num_samples=num_samples)
    if callable(response):
        response = response()

    for i, seq in enumerate(response.sequences):
        text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
        print(f"{i}:\n {text}")
