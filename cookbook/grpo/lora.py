import twinkle
from twinkle.infra import DeviceGroup
from twinkle.model import Transformers
from twinkle.dataset import Dataset

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
    dataset = Dataset('ms://swift/self-cognition', remote_group='rollout')
    dataset.map(preprocess)
    model = Transformers(pretrained_model_name_or_path='Qwen/Qwen2.5-7B-Instruct')
    model.load_state_dict()