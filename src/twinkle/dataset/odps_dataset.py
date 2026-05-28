# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from typing import Any, Callable, Dict, List, Optional, Union

from datasets import IterableDataset as HfIterableDataset

from twinkle.infra import remote_class, remote_function
from .base import DatasetMeta
from .iterable_dataset import IterableDataset


def _odps_record_to_dict(record, columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """Convert an ODPS Record to a plain dict."""
    if columns:
        return {col: record[col] for col in columns}
    names = [col.name for col in record.columns]
    return {name: record[name] for name in names}


def _make_odps_generator(
    odps,
    table_name: str,
    partition: Optional[str] = None,
    columns: Optional[List[str]] = None,
    row_converter: Optional[Callable] = None,
):
    """Return a generator function that streams records from ODPS table."""

    def _gen():
        table = odps.get_table(table_name)
        reader_kwargs = {'streaming': True}
        if partition:
            reader_kwargs['partition'] = partition
        if columns:
            reader_kwargs['columns'] = columns
        with table.open_reader(**reader_kwargs) as reader:
            for record in reader:
                row = _odps_record_to_dict(record, columns)
                if row_converter is not None:
                    row = row_converter(row)
                    if row is None:
                        continue
                yield row

    return _gen


def _make_multi_partition_generator(
    odps,
    table_name: str,
    partitions: List[str],
    columns: Optional[List[str]] = None,
    row_converter: Optional[Callable] = None,
):
    """Generator that streams records from multiple partitions sequentially."""

    def _gen():
        table = odps.get_table(table_name)
        for part in partitions:
            reader_kwargs = {'streaming': True, 'partition': part}
            if columns:
                reader_kwargs['columns'] = columns
            with table.open_reader(**reader_kwargs) as reader:
                for record in reader:
                    row = _odps_record_to_dict(record, columns)
                    if row_converter is not None:
                        row = row_converter(row)
                        if row is None:
                            continue
                    yield row

    return _gen


@remote_class(execute='first')
class OdpsIterableDataset(IterableDataset):
    """Streaming dataset backed by PyODPS table reader.

    Wraps ODPS table as an HF IterableDataset so all existing operations
    (map, filter, encode, mix_dataset) work unchanged.

    Usage:
        # Standalone
        ds = OdpsIterableDataset(
            access_id='...', access_key='...', project='proj', endpoint='http://...',
            table_name='my_table', partition='ds=20260522',
        )
        ds.set_template(MyTemplate)
        ds.encode()

        # Mix with local dataset
        ds.add_dataset(DatasetMeta(dataset_id='/path/to/local.jsonl'))
        ds.mix_dataset(interleave=True)
    """

    def __init__(
        self,
        table_name: str = '',
        partition: Union[str, List[str], None] = None,
        columns: Optional[List[str]] = None,
        row_converter: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
        # ODPS connection params (ignored if `odps` is provided)
        access_id: Optional[str] = None,
        access_key: Optional[str] = None,
        project: Optional[str] = None,
        endpoint: Optional[str] = None,
        odps=None,
        **kwargs,
    ):
        # bypass parent __init__ that would call _load_dataset
        self.template = None
        self._mixed = False
        self.datasets = {}
        self.dataset = None

        if not table_name:
            return

        odps_instance = self._get_odps_instance(
            odps, access_id, access_key, project, endpoint)

        if isinstance(partition, list) and len(partition) > 1:
            gen_fn = _make_multi_partition_generator(
                odps_instance, table_name, partition, columns, row_converter)
        else:
            single_part = partition[0] if isinstance(partition, list) else partition
            gen_fn = _make_odps_generator(
                odps_instance, table_name, single_part, columns, row_converter)

        hf_dataset = HfIterableDataset.from_generator(gen_fn)
        dataset_key = f'odps://{odps_instance.project}/{table_name}'
        if partition:
            part_str = partition if isinstance(partition, str) else ','.join(partition)
            dataset_key += f'/{part_str}'
        self.datasets[dataset_key] = hf_dataset
        self.dataset = hf_dataset

    @staticmethod
    def _get_odps_instance(odps, access_id, access_key, project, endpoint):
        if odps is not None:
            return odps
        from odps import ODPS
        _id = access_id or os.environ.get('ODPS_ACCESS_ID', '')
        _key = access_key or os.environ.get('ODPS_ACCESS_KEY', '')
        _project = project or os.environ.get('ODPS_PROJECT', '')
        _endpoint = endpoint or os.environ.get('ODPS_ENDPOINT', '')
        if not all([_id, _key, _project, _endpoint]):
            raise ValueError(
                'Must provide access_id/access_key/project/endpoint '
                'or set ODPS_ACCESS_ID/ODPS_ACCESS_KEY/ODPS_PROJECT/ODPS_ENDPOINT env vars.')
        return ODPS(_id, _key, _project, _endpoint)

    @remote_function()
    def add_dataset(self, dataset_meta: DatasetMeta, **kwargs):
        """Add a local/hub dataset for interleaved training."""
        kwargs['streaming'] = True
        from .base import Dataset
        dataset = Dataset._load_dataset(dataset_meta, **kwargs)
        self.datasets[dataset_meta.get_id()] = dataset
        if len(self.datasets) == 1:
            self.dataset = dataset

    @remote_function()
    def __len__(self):
        raise NotImplementedError('OdpsIterableDataset is streaming-only, no __len__.')

    @remote_function()
    def __getitem__(self, idx):
        raise NotImplementedError('OdpsIterableDataset is streaming-only, no __getitem__.')

    @remote_function()
    def __iter__(self):
        for row in self.dataset:
            self._write_through(row)
            yield row
