from torch.utils.data._utils.fetch import _BaseDatasetFetcher
from twinkle import DeviceMesh


class _IterableDatasetFetcher(_BaseDatasetFetcher):

    def __init__(self, base_dataset_fetcher: _BaseDatasetFetcher, batch_size: int, device_mesh: DeviceMesh):
        self._base_dataset_fetcher = base_dataset_fetcher
        self.batch_size = batch_size
        self.device_mesh = device_mesh

    def fetch(self, possibly_batched_index):
        batch = self._base_dataset_fetcher.fetch(list(range(self.batch_size)))
        if not self.device_mesh:
            yield batch
        else:
            data = batch[self.device_mesh.get_slice(len(batch))]
            yield data