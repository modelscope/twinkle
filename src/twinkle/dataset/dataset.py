import os
from typing import Callable
from ..hub import MSHub, HFHub
from ..infra import remote_class, remote_function


@remote_class()
class Dataset:

    def __init__(self, dataset_id_or_path: str, subset_name, split, **kwargs):
        if dataset_id_or_path.startswith('hf://'):
            self.dataset = HFHub.load_dataset(dataset_id_or_path[len('hf://'):], subset_name, split, **kwargs)
        elif dataset_id_or_path.startswith('ms://'):
            self.dataset = MSHub.load_dataset(dataset_id_or_path, subset_name, split, **kwargs)
        elif os.path.exists(dataset_id_or_path):
            self.dataset = MSHub.load_dataset(dataset_id_or_path, subset_name, split, **kwargs)
        else:
            raise FileNotFoundError(f'Dataset {dataset_id_or_path} not found.')

    @remote_function(dispatch='all')
    def map(self, processing_func: Callable) -> None:
        self.dataset = self.dataset.map(processing_func)
