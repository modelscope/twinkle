import numpy as np
from peft import LoraConfig

import client
from client.dataloader import DataLoader
from client.dataset import Dataset, DatasetMeta
from client.model import TransformersModel
from client.processor import GRPOLossProcessor
from client.reward import MathReward
from client.sampler import VLLMSampler
from client.weight_syncronizer.vanilla_synchronizer import VanillaSynchronizer


client.initialize(mode='remote')


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
