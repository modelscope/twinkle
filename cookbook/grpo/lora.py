import numpy as np
from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, get_device_placement
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.infra import DeviceGroup, remote_function, remote_class
from twinkle.model import TransformersModel
from twinkle.processor import GRPOLossProcessor
from twinkle.reward import MathReward
from twinkle.sampler import VLLMSampler
from twinkle.weight_loader import NativeLoader

device_groups = [
    DeviceGroup(
        name='actor',
        ranks=list(range(0, 6)),
        device_type='cuda',
    ),
    DeviceGroup(
        name='ref',
        ranks=list(range(6, 8)),
        device_type='cuda',
    ),
]


actor_device_mesh = DeviceMesh(
    device_type='cuda',
    mesh=np.array([6]),
    mesh_dim_names=('data',)
)


ref_device_mesh = DeviceMesh(
    device_type='cuda',
    mesh=np.array([2]),
    mesh_dim_names=('data',)
)

twinkle.initialize(mode='ray', groups=device_groups)


@remote_class()
class ActorGroup:

    def __init__(self, engine_args, lora_config=None, adapter_name=None, **kwargs):
        self.sampler = VLLMSampler(
            'Qwen/Qwen2.5-7B-Instruct', 
            engine_args, 
            device_mesh=actor_device_mesh
        )
        self.sampler.set_template('Qwen3Template', adapter_name=adapter_name)

        self.model = TransformersModel(
            model_id='Qwen/Qwen2.5-7B-Instruct', 
            remote_group='actor', 
            device_mesh=actor_device_mesh
        )
        
        self.model.set_loss(
            'GRPOLoss', 
            adapter_name=adapter_name,
            loss_type='grpo',
            epsilon=0.2,
            beta=0.04,
            num_generations=8,
            scale_rewards='group',
        )
        self.model.set_optimizer('AdamW', lr=1e-6, adapter_name=adapter_name)
        self.model.set_lr_scheduler('LinearLR', adapter_name=adapter_name)
        self.model.set_template('Qwen3Template', adapter_name=adapter_name)
        self.model.set_processor('InputProcessor', adapter_name=adapter_name)
        self.model.add_adapter_to_model(adapter_name, lora_config)
        self.sampler.add_adapter_to_sampler(adapter_name, lora_config)
        self.weight_loader = NativeLoader()
        self.adapter_name = adapter_name
        self.lora_config = lora_config
        
        # Loss processor for preparing GRPO-specific fields
        self.loss_processor = GRPOLossProcessor()

    @remote_function()
    def sample(self, batch):
        return self.sampler.sample(batch)

    @remote_function()
    def forward(self, inputs, **kwargs):
        return self.model.forward(inputs, **kwargs)

    @remote_function()
    def forward_backward(self, inputs, trajectories, ref_logits, **kwargs):
        # Process inputs to add GRPO-specific fields (completion_mask, logits_to_keep, num_items_in_batch)
        inputs = self.loss_processor(inputs)
        return self.model.forward_backward(inputs, trajectories, ref_logits, **kwargs)

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
    dataset.set_template('Qwen3Template')
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
    
    ref_model = TransformersModel(
        model_id='Qwen/Qwen2.5-7B-Instruct', 
        remote_group='ref', 
        device_mesh=ref_device_mesh
    )
    ref_model.set_processor('InputProcessor')
    ref_model.set_template('Qwen3Template')
    reward = MathReward()
    
    print("Device placement:", get_device_placement())
    
    for batch in dataloader:
        trajectories = actor_group.sample(batch)
        old_logits = actor_group.forward(trajectories)
        ref_logits = ref_model.forward(trajectories)
        trajectories = reward.calculate(trajectories, batch)
        actor_group.forward_backward(batch, trajectories, ref_logits, adapter_name='default')
        actor_group.step()
        actor_group.zero_grad()
        actor_group.lr_step()
