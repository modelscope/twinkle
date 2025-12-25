import twinkle
from twinkle.infra import DeviceGroup
from twinkle.model import TransformersModel
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.sampler import VLLMSampler
from twinkle.dataloader import DataLoader
from twinkle.template import Qwen3Template
from twinkle.reward import MathReward

device_groups = [
    DeviceGroup(
        name='actor',
        ranks=list(range(0, 4)),
        device_type='GPU',
    ),
    DeviceGroup(
        name='rollout',
        ranks=list(range(4, 6)),
        device_type='GPU',
    ),
    DeviceGroup(
        name='ref',
        ranks=list(range(6, 8)),
        device_type='GPU',
    ),
]


twinkle.initialize(mode='local', groups=device_groups)


def create_dataset():
    dataset = Dataset(DatasetMeta('ms://modelscope/competition_math'))
    dataset.map('CompetitionMathProcessor')
    return dataset


def train():
    dataloader = DataLoader(create_dataset, remote_group='rollout')
    engine_args = {

    }
    sampler = VLLMSampler(engine_args, remote_group='rollout')
    sampler.set_input_processor('GRPOInputProcessor')
    sampler.set_template('Qwen3Template')
    model = TransformersModel(pretrained_model_name_or_path='Qwen/Qwen2.5-7B-Instruct', remote_group='actor')
    ref_model = TransformersModel(pretrained_model_name_or_path='Qwen/Qwen2.5-7B-Instruct', remote_group='ref')
    model.set_loss('GRPOLoss')
    model.set_optimizer('AdamW')
    model.set_lr_scheduler('LinearDecay')
    model.set_input_processor('GRPOInputProcessor')
    model.set_template('Qwen3Template')
    ref_model.set_input_processor('GRPOInputProcessor')
    ref_model.set_template('Qwen3Template')
    reward = MathReward()
    for batch in dataloader:
        trajectories = sampler.sample(batch)
        logits = ref_model.forward(trajectories)
        trajectories = reward.calculate(trajectories)
        model.forward(trajectories)
        model.calculate_loss(ref_logits=logits, trajectories=trajectories)
        model.backward()
        model.step()
        model.zero_grad()
        model.lr_step()
