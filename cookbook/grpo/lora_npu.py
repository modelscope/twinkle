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
from twinkle.sampler import VLLMSampler
from twinkle.weight_loader import NativeLoader

import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()
os.environ.setdefault('TRUST_REMOTE_CODE', '1')
os.environ.setdefault('TWINKLE_SEED', '42')
os.environ.setdefault('TWINKLE_FULL_DETERMINISM', '1')
use_ref_model = True
num_generations = 2
kl_beta = 0.0
max_length = 4096
truncation_strategy = 'right'

def build_template_kwargs(include_model_id: bool = False):
    kwargs = {}
    if include_model_id:
        kwargs['model_id'] = '/home/zyh/model/Qwen2.5-7B-Instruct'
    if max_length > 0:
        kwargs['max_length'] = max_length
        kwargs['truncation_strategy'] = truncation_strategy
    return kwargs

visible_devices_env = os.environ.get('ASCEND_RT_VISIBLE_DEVICES') or os.environ.get('CUDA_VISIBLE_DEVICES')
if visible_devices_env:
    visible_devices = [d for d in visible_devices_env.split(',') if d.strip()]
    nproc_per_node = len(visible_devices)
else:
    nproc_per_node = 8


device_groups = [
    DeviceGroup(
        name='actor',
        ranks=list(range(0, 6)),
        device_type='npu',
    ),
    DeviceGroup(
        name='ref',
        ranks=[6, 7],
        device_type='npu',
    ),
]
actor_device_mesh = DeviceMesh(
    device_type='npu',
    mesh=np.array([6]),
    mesh_dim_names=('dp',),
)
ref_device_mesh = DeviceMesh(
    device_type='npu',
    mesh=np.array([2]),
    mesh_dim_names=('dp',),
)

twinkle.initialize(mode='ray', groups=device_groups, nproc_per_node=nproc_per_node)


@remote_class()
class ActorGroup:

    def __init__(self, engine_args, lora_config=None, adapter_name=None, **kwargs):
        self.sampler = VLLMSampler(
            '/home/zyh/model/Qwen2.5-7B-Instruct',
            engine_args,
            device_mesh=actor_device_mesh,
        )
        self.sampler.add_adapter_to_sampler(adapter_name, lora_config)
        self.sampler.set_template('Qwen3Template', adapter_name=adapter_name, **build_template_kwargs())

        self.model = TransformersModel(
            model_id='/home/zyh/model/Qwen2.5-7B-Instruct', 
            remote_group='actor', 
            device_mesh=actor_device_mesh
        )
        self.model.add_adapter_to_model(adapter_name, lora_config)

        self.model.set_loss(
            'GRPOLoss',
            loss_type='grpo',
            epsilon=0.2,
            beta=kl_beta,
            num_generations=num_generations,
            scale_rewards='group',
        )
        self.model.set_optimizer('AdamW', lr=1e-6)
        self.model.set_lr_scheduler('LinearLR')
        self.model.set_template('Qwen3Template', **build_template_kwargs())
        self.model.set_processor('GRPOLossProcessor')
        self.weight_loader = NativeLoader()
        self.adapter_name = adapter_name
        self.lora_config = lora_config

    @remote_function(collect='flatten')
    def sample(self, batch):
        return self.sampler.sample(batch)

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
    dataset.set_template('Qwen3Template', **build_template_kwargs(include_model_id=True))
    dataset.map('CompetitionMathProcessor')
    dataset.check(batched=True)
    return dataset


def train():
    dataloader = DataLoader(
        create_dataset, 
        remote_group='actor', 
        device_mesh=actor_device_mesh
    )
    
    engine_args = {

    }
    lora_config = LoraConfig(
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
    )

    actor_group = ActorGroup(
        engine_args,
        remote_group='actor',
        lora_config=lora_config,
        adapter_name='default',
    )
    
    ref_model = None
    if use_ref_model:
        ref_model = TransformersModel(
            model_id='/home/zyh/model/Qwen2.5-7B-Instruct', 
            remote_group='ref', 
            device_mesh=ref_device_mesh
        )
        ref_model.set_processor('InputProcessor')
        ref_model.set_template('Qwen3Template', **build_template_kwargs())
    reward = MathReward()
    
    print("Device placement:", get_device_placement())
    
    step = 0
    max_steps = int(os.environ.get('TWINKLE_MAX_STEPS', '0'))
    for batch in dataloader:
        step += 1
        print(f"[step {step}] batch ready", flush=True)
        trajectories = actor_group.sample(batch)
        if callable(trajectories):
            trajectories = trajectories()
        print(f"[step {step}] sampled trajectories", flush=True)
        ref_logits = None
        if use_ref_model:
            ref_outputs = ref_model.forward_only(inputs=trajectories)
            if callable(ref_outputs) and getattr(ref_outputs, '_is_lazy_collect', False):
                ref_outputs = ref_outputs()
            if isinstance(ref_outputs, list):
                ref_logits = [o['logits'] if isinstance(o, dict) else o.logits for o in ref_outputs]
            else:
                ref_logits = ref_outputs['logits'] if isinstance(ref_outputs, dict) else ref_outputs.logits

        rewards = reward.calculate(trajectories, batch)
        if callable(rewards):
            rewards = rewards()
        for trajectory, reward_value in zip(trajectories, rewards):
            trajectory['rewards'] = reward_value
        print(f"[step {step}] rewards computed", flush=True)

        loss = actor_group.forward_backward(trajectories, trajectories, ref_logits)
        if callable(loss):
            loss = loss()
        print(f"[step {step}] loss: {loss}", flush=True)
        actor_group.step()
        actor_group.zero_grad()
        actor_group.lr_step()
        if max_steps and step >= max_steps:
            break

if __name__ == '__main__':
    train()
