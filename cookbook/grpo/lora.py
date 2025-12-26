import twinkle
from twinkle.infra import DeviceGroup, remote_function, remote_class
from twinkle.model import TransformersModel
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.sampler import VLLMSampler
from twinkle.dataloader import DataLoader
from twinkle.reward import MathReward
from twinkle.weight_syncronizer.vanilla_synchronizer import VanillaSynchronizer

device_groups = [
    DeviceGroup(
        name='actor',
        ranks=list(range(0, 6)),
        device_type='GPU',
    ),
    DeviceGroup(
        name='ref',
        ranks=list(range(6, 8)),
        device_type='GPU',
    ),
]


twinkle.initialize(mode='local', groups=device_groups)


@remote_class()
class ActorGroup:

    def __init__(self, engine_args, lora_config=None, adapter_name=None):
        self.sampler = VLLMSampler(engine_args)
        self.sampler.set_processor('GRPOInputProcessor')
        self.sampler.set_template('Qwen3Template')

        self.model = TransformersModel(pretrained_model_name_or_path='Qwen/Qwen2.5-7B-Instruct', remote_group='actor')
        self.model.set_loss('GRPOLoss')
        self.model.set_optimizer('AdamW')
        self.model.set_lr_scheduler('LinearDecay')
        self.model.set_processor('GRPOInputProcessor')
        self.model.set_template('Qwen3Template', 'Qwen/Qwen2.5-7B-Instruct')
        self.model.add_adapter_to_model(adapter_name, lora_config)
        self.sampler.add_adapter_to_sampler(adapter_name, lora_config)
        self.weight_sync = VanillaSynchronizer()
        self.adapter_name = adapter_name
        self.lora_config = lora_config

    @remote_function()
    def sample(self, batch):
        return self.sampler.sample(batch)

    @remote_function()
    def forward_backward(self, inputs, trajectories, ref_logits):
        return self.model.forward_backward(inputs, trajectories, ref_logits)

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
        self.weight_sync(self.model, self.sampler, self.adapter_name)

def create_dataset():
    dataset = Dataset(DatasetMeta('ms://modelscope/competition_math'))
    dataset.map('CompetitionMathProcessor')
    return dataset


def train():
    dataloader = DataLoader(create_dataset, remote_group='rollout')
    dataloader.set_processor('GRPOInputProcessor')
    engine_args = {

    }
    actor_group = ActorGroup(engine_args, remote_group='rollout')
    ref_model = TransformersModel(pretrained_model_name_or_path='Qwen/Qwen2.5-7B-Instruct', remote_group='ref')
    ref_model.set_processor('GRPOInputProcessor')
    ref_model.set_template('Qwen3Template', 'Qwen/Qwen2.5-7B-Instruct')
    reward = MathReward()
    for batch in dataloader:
        trajectories = actor_group.sample(batch)
        logits = ref_model.forward(trajectories)
        trajectories = reward.calculate(trajectories)
        actor_group.forward_backward(batch, trajectories, logits)
        actor_group.step()
        actor_group.zero_grad()
        actor_group.lr_step()
