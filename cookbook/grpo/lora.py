import twinkle
from twinkle.infra import DeviceGroup
from twinkle.model import TransformersModel
from twinkle.dataset import Dataset
from twinkle.sampler import VLLMSampler
from twinkle.dataloader import DataLoader
from twinkle.template import Qwen3Template
from twinkle.loss import GRPOLoss
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


def preprocess(row):
    return row


def create_dataset():
    dataset = Dataset('ms://swift/self-cognition')
    dataset.map(preprocess)
    return dataset


def train():
    dataloader = DataLoader(create_dataset, remote_group='rollout')
    engine_args = {

    }
    sampler = VLLMSampler(engine_args, template=Qwen3Template, remote_group='rollout')
    model = TransformersModel(pretrained_model_name_or_path='Qwen/Qwen2.5-7B-Instruct', remote_group='actor')
    ref_model = TransformersModel(pretrained_model_name_or_path='Qwen/Qwen2.5-7B-Instruct', remote_group='ref')
    model.set_loss(GRPOLoss)
    model.set_optimizer('AdamW')
    model.set_lr_scheduler('LinearDecay')
    template = Qwen3Template('qwen2.5')
    reward = MathReward()
    for batch in dataloader:
        trajectories = sampler.sample(batch)
        inputs = template.encode(trajectories)
        logits = ref_model.forward(inputs)
        rewards = reward.calculate(trajectories)
        model.forward(inputs)
        model.calculate_loss(ref_logits=logits, reward=rewards)
        model.backward()
        model.step()
        model.zero_grad()
        model.lr_step()
