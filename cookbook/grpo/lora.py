import twinkle
from twinkle.infra import DeviceGroup
from twinkle.model import TransformersModel
from twinkle.dataset import Dataset
from twinkle.sampler import VLLMSampler
from twinkle.dataloader import DataLoader
from twinkle.template import Template
from twinkle.loss import GRPOLoss

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


def train():
    dataset = Dataset('ms://swift/self-cognition', remote_group='rollout', reward=)
    dataset.map(preprocess)
    dataloader = DataLoader(dataset, remote_group='rollout')
    from vllm import EngineArgs
    engine_args = EngineArgs()
    sampler = VLLMSampler(engine_args, template_type='qwen2.5', remote_group='rollout')
    model = TransformersModel(pretrained_model_name_or_path='Qwen/Qwen2.5-7B-Instruct', remote_group='actor')
    ref_model = TransformersModel(pretrained_model_name_or_path='Qwen/Qwen2.5-7B-Instruct', remote_group='ref')
    model.set_loss(GRPOLoss)
    model.set_optimizer('AdamW')
    model.set_lr_scheduler('LinearDecay')
    template = Template('qwen2.5')
    for batch in dataloader:
        trajectories = sampler.sample(batch)
        inputs = template.encode(trajectories)
        logits = ref_model.forward(inputs)
        trajectories = reward.calculate(trajectories)
        #model.forward(inputs)
        #model.loss(ref_logits=logits)
        #model.backward()
        model.forward_backward(inputs, ref_logits=logits, reward=reward)
        model.step()
        model.zero_grad()
        model.lr_step()
