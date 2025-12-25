from typing import Callable, Union

from torch.distributed import DeviceMesh
from torch.utils.data import DataLoader as TorchDataLoader

from twinkle import remote_class, Dataset


@remote_class()
class DataLoader(TorchDataLoader):

    def __init__(self, dataset: Union[Dataset, Callable], device_mesh: DeviceMesh, **dataloader_params):
        if isinstance(dataset, Callable):
            self.dataset = dataset()
        else:
            self.dataset = dataset
        self.dataloader = TorchDataLoader(self.dataset, **dataloader_params)
        self.device_mesh = device_mesh

    def set_template(self, template: Union[Type[Template], str]):
        if isinstance(template, str):
            if hasattr(twinkle.template, template):
                template = getattr(twinkle.template, template)
            else:
                template = Plugin.load_plugin(template, Template)
        self.template = template(self.model_id)
