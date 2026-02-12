# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Callable, Any

from torch.utils.data import Dataset
from torch.utils.data._utils.fetch import _BaseDatasetFetcher
from twinkle import DeviceMesh


class DeviceMeshIterableFetcher(_BaseDatasetFetcher):
    """A data sampler which fetch data by DeviceMesh.

    Args:
        dataset: The input dataset.
        auto_collation: The collect method when fetching batches. When input is a dataset, keep this param `true`.
        collate_fn: The collate fn.
        drop_last: Whether to drop the last not full batch.
        batch_size: The batch size.
        device_mesh: DeviceMesh instance.
        max_retries: The maximum number of retries when fetching failed.
    """

    def __init__(self, dataset: Dataset, auto_collation: bool, collate_fn: Callable[[Any], Any],
                 drop_last: bool, batch_size: int, device_mesh: DeviceMesh, min_batch_size: int = None, max_retries:int=20):
        super().__init__(dataset, auto_collation, collate_fn, drop_last)
        self.dataset_iter = iter(dataset)
        self.ended = False
        self.batch_size = batch_size
        self.device_mesh = device_mesh
        self.max_retries = max_retries
        self.min_batch_size = min_batch_size
        if self.min_batch_size is None and self.device_mesh is not None:
            self.min_batch_size = self.device_mesh.data_world_size

    def fetch(self, _):
        """Fetch data of global batch size and returns the slices belong to the current RANK.

        This function will retry until a valid data returns.
        Returns:
            The input data slice.
        """
        if self.ended:
            raise StopIteration

        if self.auto_collation:
            data = []
            for _ in range(self.batch_size):
                try:
                    _data = None
                    for _ in range(self.max_retries):
                        try:
                            _data = next(self.dataset_iter)
                            if _data is None:
                                continue
                        except StopIteration as e:
                            raise e
                        except Exception: # noqa
                            continue
                        else:
                            break
                    data.append(_data)
                except StopIteration:
                    self.ended = True
                    break
            if len(data) == 0 or (
                self.drop_last and len(data) < self.batch_size
            ):
                raise StopIteration
        else:
            data = next(self.dataset_iter)

        if self.device_mesh:
            if len(data) < self.min_batch_size:
                raise StopIteration
            else:
                data = data[self.device_mesh.get_slice(len(data))]
        return self.collate_fn(data)