# Copyright (c) ModelScope Contributors. All rights reserved.
from torch.utils.data._utils.fetch import _BaseDatasetFetcher
from twinkle import DeviceMesh


class DeviceMeshIterableFetcher(_BaseDatasetFetcher):

    def __init__(self, dataset, auto_collation, collate_fn, drop_last, batch_size: int, device_mesh: DeviceMesh, max_retries=20):
        super().__init__(dataset, auto_collation, collate_fn, drop_last)
        self.dataset_iter = iter(dataset)
        self.ended = False
        self.batch_size = batch_size
        self.device_mesh = device_mesh
        self.max_retries = max_retries

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
            if len(data) < self.device_mesh.data_world_size:
                raise StopIteration
            else:
                data = data[self.device_mesh.get_slice(len(data))]
        return self.collate_fn(data)