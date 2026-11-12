# Copyright (c) ModelScope Contributors. All rights reserved.
from twinkle.dataset import DatasetMeta
from twinkle_client.http import ClientTransport
from .base import Dataset


class LazyDataset(Dataset):
    """Client wrapper for LazyDataset that calls server HTTP endpoints."""

    def __init__(
        self,
        dataset_meta: DatasetMeta = None,
        *,
        transport: ClientTransport | None = None,
        **kwargs,
    ):
        self._bind_remote('dataset', 'LazyDataset', dataset_meta=dataset_meta, transport=transport, **kwargs)
