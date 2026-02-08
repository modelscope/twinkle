import numpy as np
from peft import LoraConfig
import os
import twinkle
from twinkle import DeviceMesh, get_device_placement
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.infra import DeviceGroup, remote_function, remote_class
from twinkle.model import TransformersModel
from twinkle.reward import MathReward
from twinkle.sampler import VLLMSampler, TorchSampler
from twinkle.data_format.types import SamplingParams, SampleResponse
from twinkle.weight_loader import NativeLoader
from twinkle.advantage import compute_advantages

# Environment variable setup
os.environ.setdefault('TRUST_REMOTE_CODE', '1')
os.environ.setdefault('TWINKLE_SEED', '42')
os.environ.setdefault('TWINKLE_FULL_DETERMINISM', '1')
os.environ.setdefault('RAY_TMPDIR', os.path.expanduser('~/tmp/ray'))

# Training configuration
use_ref_model = os.environ.get('TWINKLE_USE_REF_MODEL', '1') != '0'
use_torch_sampler = os.environ.get('TWINKLE_USE_TORCH_SAMPLER', '0') != '0'
num_generations = 2
kl_beta = 0.0
max_length = int(os.environ.get('TWINKLE_MAX_LENGTH', '4096'))
model_path = os.environ.get('TWINKLE_MODEL_ID', 'Qwen/Qwen3-0.6B')
debug_mode = os.environ.get('TWINKLE_DEBUG', '0') != '0'


def build_template_kwargs(include_model_id: bool = False):
    kwargs = {}
    if include_model_id:
        kwargs['model_id'] = model_path
    if max_length > 0:
        kwargs['max_length'] = max_length
        kwargs['truncation_strategy'] = 'right'
    return kwargs


def parse_device_config():
    visible_devices_env = os.environ.get('ASCEND_RT_VISIBLE_DEVICES')
    if visible_devices_env:
        visible_devices = [d for d in visible_devices_env.split(',') if d.strip()]
        nproc_per_node = len(visible_devices)
    else:
        nproc_per_node = 8
    
    def _parse_ranks_env(name: str):
        raw = os.environ.get(name)
        if not raw:
            return None
        ranks = [int(v.strip()) for v in raw.split(',') if v.strip()]
        return ranks or None
    
    actor_ranks = _parse_ranks_env('TWINKLE_ACTOR_RANKS')
    ref_ranks = _parse_ranks_env('TWINKLE_REF_RANKS')
    
    if actor_ranks is None:
        actor_size = int(os.environ.get('TWINKLE_ACTOR_SIZE', '6'))
        actor_ranks = list(range(actor_size))
    
    if ref_ranks is None and use_ref_model:
        ref_size = int(os.environ.get('TWINKLE_REF_SIZE', '2'))
        ref_start = (max(actor_ranks) + 1) if actor_ranks else 0
        ref_ranks = list(range(ref_start, ref_start + ref_size))
    
    return nproc_per_node, actor_ranks, ref_ranks


def create_device_groups(actor_ranks, ref_ranks):
    device_groups = [
        DeviceGroup(
            name='actor',
            ranks=actor_ranks,
            device_type='npu',
        ),
    ]
    
    if use_ref_model and ref_ranks:
        device_groups.append(
            DeviceGroup(
                name='ref',
                ranks=ref_ranks,
                device_type='npu',
            )
        )
    
    return device_groups


def create_device_meshes(actor_ranks, ref_ranks):
    actor_device_mesh = DeviceMesh(
        device_type='npu',
        mesh=np.array([len(actor_ranks)]),
        mesh_dim_names=('dp',),
    )
    
    ref_device_mesh = DeviceMesh(
        device_type='npu',
        mesh=np.array([len(ref_ranks) if ref_ranks is not None else 0]),
        mesh_dim_names=('dp',),
    )
    
    return actor_device_mesh, ref_device_mesh


def get_eos_token_ids():
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        eos_ids = tokenizer.eos_token_id
        if eos_ids is None:
            return []
        elif isinstance(eos_ids, (list, tuple)):
            return list(eos_ids)
        else:
            return [eos_ids]
    except Exception:
        return []


def get_sampling_params(eos_token_ids) -> SamplingParams:
    """Create SamplingParams for generation."""
    return SamplingParams(
        max_tokens=128,
        temperature=1,
        top_p=0.95,
    )


def build_trajectories_from_sample_response(sample_response: SampleResponse, batch_list, tokenizer):
    """Convert sampler output into GRPO trajectories."""
    if not sample_response or not getattr(sample_response, 'sequences', None):
        return []
    if not batch_list:
        return []

    trajectories = []
    for i, seq in enumerate(sample_response.sequences):
        src_batch = batch_list[i % len(batch_list)]
        src_messages = [dict(msg) for msg in src_batch.get('messages', [])]
        if src_messages and src_messages[-1].get('role') == 'assistant':
            # Remove reference answer and append sampled assistant reply.
            src_messages = src_messages[:-1]

        response_text = tokenizer.decode(seq.tokens, skip_special_tokens=True) if tokenizer is not None else ''
        trajectories.append({
            'messages': src_messages + [{'role': 'assistant', 'content': response_text}],
            'user_data': list(src_batch.get('user_data', [])),
        })
    return trajectories


def debug_print_rollout(step, trajectories, ground_truths, rewards=None):
    """Debug helper that prints rollout intermediates (sampling, rewards, etc.).

    Set the TWINKLE_DEBUG environment variable to '1' to enable this output.
    The output covers:
    1. Prompt, model prediction, and ground truth for the first sample
    2. Reward statistics (min, mean, max)

    Args:
        step: Current training step
        trajectories: List of sampled trajectories
        ground_truths: List of ground truth records
        rewards: Optional list of reward values (prints stats when provided)

    Environment Variables:
        TWINKLE_DEBUG: Set to '1' to enable debug logging

    Example:
        # Enable debug mode
        TWINKLE_DEBUG=1 python lora_npu.py
    """
    if not debug_mode:
        return

    # Extract prediction content (last message of the first sample)
    try:
        pred_msg = trajectories[0]['messages'][-1]['content'] if trajectories else None
    except (KeyError, IndexError, TypeError):
        pred_msg = None

    # Extract prompt and ground truth (first sample)
    try:
        prompt_msg = ground_truths[0]['messages'][0]['content'] if ground_truths else None
        gt_msg = ground_truths[0]['messages'][-1]['content'] if ground_truths else None
    except (KeyError, IndexError, TypeError):
        prompt_msg, gt_msg = None, None

    # Print prompt, prediction, and ground truth
    print(
        f'[DEBUG][step {step}] prompt={prompt_msg} | pred={pred_msg} | gt={gt_msg}',
        flush=True,
    )

    # Print reward statistics
    if rewards is not None and isinstance(rewards, (list, tuple)) and rewards:
        rewards_np = np.array(rewards, dtype=np.float32)
        print(
            f'[DEBUG][step {step}] rewards: n={len(rewards)}, '
            f'min={rewards_np.min():.4f}, mean={rewards_np.mean():.4f}, max={rewards_np.max():.4f}',
            flush=True,
        )


def _collect_sample_responses(results):
    """Custom collect function to merge multiple SampleResponse objects."""
    if not results:
        return SampleResponse(sequences=[])
    if len(results) == 1:
        return results[0]
    all_sequences = []
    for resp in results:
        if resp is not None and hasattr(resp, 'sequences'):
            all_sequences.extend(resp.sequences)
    return SampleResponse(sequences=all_sequences)


@remote_class()
class ActorGroup:
    
    def __init__(self, engine_args=None, lora_config=None, adapter_name=None, **kwargs):
        if use_torch_sampler:
            self.sampler = TorchSampler(
                model_path,
                device_mesh=actor_device_mesh,
            )
        else:
            if engine_args is None:
                raise ValueError("engine_args is required for VLLMSampler.")
            self.sampler = VLLMSampler(
                model_path,
                engine_args=engine_args,
                device_mesh=actor_device_mesh,
            )
        self.sampler.add_adapter_to_sampler(adapter_name, lora_config)
        # Fix: use 'Template' instead of 'Qwen3Template' - Qwen3Template was never exported in twinkle.template
        self.sampler.set_template('Template', adapter_name=adapter_name, **build_template_kwargs(include_model_id=True))
        
        self.model = TransformersModel(
            model_id=model_path, 
            remote_group='actor', 
            device_mesh=actor_device_mesh
        )
        self.model.add_adapter_to_model(adapter_name, lora_config)
        
        self.model.set_loss(
            'GRPOLoss',
            epsilon=0.2,
            beta=kl_beta,
            num_generations=num_generations,
        )
        
        self.model.set_optimizer('AdamW', lr=1e-6)
        self.model.set_lr_scheduler('LinearLR')
        self.model.set_template('Template', **build_template_kwargs(include_model_id=False))
        self.model.set_processor('GRPOLossProcessor')
        
        self.weight_loader = NativeLoader()
        self.adapter_name = adapter_name
        self.lora_config = lora_config
    
    @remote_function(collect=_collect_sample_responses)
    def sample(self, batch, sampling_params: SamplingParams = None):
        return self.sampler.sample(batch, sampling_params=sampling_params, adapter_name=self.adapter_name)
    
    @remote_function()
    def forward(self, inputs, **kwargs):
        outputs = self.model.forward(inputs=inputs, **kwargs)
        return outputs['logits']
    
    @remote_function()
    def forward_only(self, inputs, **kwargs):
        outputs = self.model.forward_only(inputs=inputs, **kwargs)
        return outputs['logits']
    
    @remote_function()
    def forward_backward(self, inputs, trajectories, ref_logits=None, old_logits=None, **kwargs):
        if old_logits is None:
            old_logits = self.model.forward_only(inputs=inputs, **kwargs)['logits']
        return self.model.forward_backward(
            inputs=inputs,
            trajectories=trajectories,
            ref_logits=ref_logits,
            old_logits=old_logits,
            **kwargs,
        )
    
    @remote_function()
    def step(self):
        return self.model.step()
    
    @remote_function()
    def zero_grad(self):
        return self.model.zero_grad()
    
    @remote_function()
    def lr_step(self):
        return self.model.lr_step()
    
    @remote_function()
    def sync_weights(self):
        self.weight_loader(self.model, self.sampler, self.adapter_name)


def create_dataset():
    dataset = Dataset(DatasetMeta('ms://modelscope/competition_math'))
    dataset.set_template('Template', **build_template_kwargs(include_model_id=True))
    dataset.map('CompetitionMathGRPOProcessor')
    dataset.check(batched=True)
    return dataset


def train():
    raise NotImplementedError("Not implemented")
    nproc_per_node, actor_ranks, ref_ranks = parse_device_config()
    
    device_groups = create_device_groups(actor_ranks, ref_ranks)
    global actor_device_mesh, ref_device_mesh
    actor_device_mesh, ref_device_mesh = create_device_meshes(actor_ranks, ref_ranks)
    
    twinkle.initialize(mode='ray', groups=device_groups, nproc_per_node=nproc_per_node)
    
    dataloader = DataLoader(
        create_dataset, 
        remote_group='actor', 
        device_mesh=actor_device_mesh
    )
    
    eos_token_ids = get_eos_token_ids()
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        tokenizer = None
    
    engine_args = {
        'model': model_path,
        'enable_lora': True,
        'max_loras': 1,
        'max_lora_rank': 64,
        'max_model_len': max_length,
        'gpu_memory_utilization': float(os.environ.get('TWINKLE_VLLM_GPU_MEMORY_UTILIZATION', '0.9')),
    }
    
    lora_config = LoraConfig(
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
    )
    
    actor_group = ActorGroup(
        None if use_torch_sampler else engine_args,
        remote_group='actor',
        lora_config=lora_config,
        adapter_name='default',
    )
    
    ref_model = None
    if use_ref_model:
        ref_model = TransformersModel(
            model_id=model_path, 
            remote_group='ref', 
            device_mesh=ref_device_mesh
        )
        ref_model.set_processor('InputProcessor')
        ref_model.set_template('Template', **build_template_kwargs())
    
    reward = MathReward()
    
    print('Device placement:', get_device_placement())
    
    step = 0
    max_steps = int(os.environ.get('TWINKLE_MAX_STEPS', '20'))
    
    for batch in dataloader:
        step += 1
        print(f'[step {step}] batch ready', flush=True)
        
        if isinstance(batch, dict):
            batch_list = [batch]
        else:
            batch_list = list(batch)
        sampling_params = get_sampling_params(eos_token_ids)
        
        sample_response = actor_group.sample(batch_list, sampling_params)
        if callable(sample_response):
            sample_response = sample_response()
        trajectories = build_trajectories_from_sample_response(sample_response, batch_list, tokenizer)
        if not trajectories:
            print(f'[step {step}] empty sampled trajectories, skip.', flush=True)
            continue

        # Expand ground truths to align with sampled trajectory count.
        ground_truths = [batch_list[i % len(batch_list)] for i in range(len(trajectories))]

        ref_logits = None
        if use_ref_model:
            ref_outputs = ref_model.forward_only(inputs=trajectories)
            if callable(ref_outputs) and getattr(ref_outputs, '_is_lazy_collect', False):
                ref_outputs = ref_outputs()
            if isinstance(ref_outputs, list):
                ref_logits = [o['logits'] if isinstance(o, dict) else o.logits for o in ref_outputs]
            else:
                ref_logits = ref_outputs['logits'] if isinstance(ref_outputs, dict) else ref_outputs.logits
        
        rewards = reward(trajectories, ground_truths)
        if callable(rewards):
            rewards = rewards()

        effective_num_generations = num_generations if len(rewards) % num_generations == 0 else 1
        scale = 'group' if effective_num_generations > 1 else 'batch'
        advantages = compute_advantages(
            rewards,
            num_generations=effective_num_generations,
            scale=scale,
        )
        for trajectory, advantage in zip(trajectories, advantages.tolist()):
            trajectory['advantages'] = float(advantage)

        # Debug: print reward statistics (enable via TWINKLE_DEBUG=1)
        debug_print_rollout(step, trajectories, ground_truths, rewards=rewards)

        loss = actor_group.forward_backward(trajectories, trajectories, ref_logits)
        if callable(loss):
            loss = loss()
        
        print(f'[step {step}] loss: {loss}', flush=True)
        
        actor_group.step()
        actor_group.zero_grad()
        actor_group.lr_step()
        
        if max_steps and step >= max_steps:
            break



if __name__ == '__main__':
    train()
