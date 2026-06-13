# Copyright (c) ModelScope Contributors. All rights reserved.
import json as _json
import os.path
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datasets import DatasetDict, IterableDataset, concatenate_datasets, interleave_datasets, load_dataset
from queue import Queue
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import IterableDataset as TorchIterableDataset
from typing import Any, Callable, Dict, List, Optional, Type, Union

import twinkle
from twinkle import preprocessor
from twinkle.hub import HubOperation
from twinkle.infra import remote_class, remote_function
from twinkle.preprocessor import DataFilter, Preprocessor
from twinkle.template import Template
from twinkle.utils import construct_class, processing_lock
from twinkle.utils.parallel import PosixFileLock

try:
    import multiprocess
    multiprocess.set_start_method('spawn', force=True)
except RuntimeError:
    pass


@dataclass
class DatasetMeta:
    """
    The dataset meta-information, used to describe a dataset.
    """
    # The dataset id or local path
    dataset_id: str = ''
    # The subset name
    subset_name: str = 'default'
    # The split
    split: str = 'train'
    # Pick a data slice
    data_slice: Iterable = None
    # In-memory / in-process data source. Supports:
    #   - List[Dict]      (row-oriented, eager)
    #   - Dict[str, List] (column-oriented, eager)
    #   - Callable        (generator function; routed to HF from_generator,
    #                      streaming vs eager picked from `streaming` kwarg.
    #                      Bind args via functools.partial.)
    #   - HFDataset / HFIterableDataset (already-constructed, passed through)
    data: Any = None

    def get_id(self):
        if self.data is not None:
            return f'__memory_{self._uid}__:' + self.subset_name + ':' + self.split
        return self.dataset_id.replace(os.sep, '_').replace('.', '_') + ':' + self.subset_name + ':' + self.split

    def __post_init__(self):
        import uuid
        self._uid = uuid.uuid4().hex[:8]
        if self.data_slice is not None and not isinstance(self.data_slice, Iterable):
            raise ValueError('data_slice must be an iterable')
        if not self.dataset_id and self.data is None:
            raise ValueError('Either dataset_id or data must be provided')


@remote_class(execute='first')
class Dataset(TorchDataset):
    """A dataset wrapper to load and map the dataset.

    Args:
        dataset_meta: A dataset meta information for loading the original dataset.
        kwargs:
            streaming: Whether is streaming mode.
            num_proc: Number of processes to use.
            revision: The revision of the dataset, only available when dataset is id in the hf/ms hub.
            Any other kwargs supported by `datasets.load_dataset`.
    """

    def __init__(self, dataset_meta: DatasetMeta = None, **kwargs):
        self.template = None
        self._mixed = False
        if dataset_meta is None:
            self.datasets = {}
            self.dataset = None
            return
        trust_remote_code = bool(os.environ.get('TWINKLE_TRUST_REMOTE_CODE', '1'))
        if not trust_remote_code:
            kwargs['trust_remote_code'] = False
        dataset = self._load_dataset(dataset_meta, **kwargs)
        self.datasets = {dataset_meta.get_id(): dataset}
        self.dataset = dataset

    @remote_function()
    def set_template(self, template_func: Union[Template, Type[Template], str], **kwargs):
        """Set the template to encode/check the dataset.

        Args:
            template_func: The template class/instance, or the template plugin, or the template class name to load.
            **kwargs: The template init params.
        """
        self.template = construct_class(template_func, Template, twinkle.template, **kwargs)

    @staticmethod
    def _normalize_cache_kwargs(target, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        kw = dict(kwargs)
        # Streaming datasets (HF IterableDataset / torch IterableDataset wrappers) reject load_from_cache_file.
        if isinstance(target, (IterableDataset, TorchIterableDataset)):
            kw.pop('load_from_cache_file', None)
        else:
            kw.setdefault('load_from_cache_file', False)
        return kw

    @remote_function()
    def encode(self, add_generation_prompt: bool = False, **kwargs):
        """An inplace operation to encode the dataset.

        Args:
            add_generation_prompt: If True, append generation prompt suffix
                (e.g. ``<|im_start|>assistant\\n``) to each encoded sample.
                Useful when the encoded dataset will be used for sampling/inference.
            **kwargs: The mapping and filter kwargs of the `datasets.map`.
        """
        kwargs['batched'] = True  # Only supported batched, because a single row may explode to several rows
        kwargs = self._normalize_cache_kwargs(self.dataset, kwargs)
        from functools import partial
        encode_fn = partial(self.template.batch_encode, add_generation_prompt=add_generation_prompt)
        # Dataset.filter() does not accept map-only kwargs (e.g. remove_columns); split them off.
        filter_kwargs = {k: v for k, v in kwargs.items() if k != 'remove_columns'}
        with processing_lock('dataset'):
            # use a default lock because encode is to all datasets
            self.dataset = self.dataset.map(encode_fn, **kwargs).filter(
                lambda batch: [True] * len(next(iter(batch.values())))
                if 'input_ids' not in batch else [len(x) > 0 for x in batch['input_ids']], **filter_kwargs)

    @remote_function()
    def check(self, **kwargs):
        """An inplace operation to check the dataset.

        Args:
            **kwargs: The mapping and filter kwargs of the `datasets.map`.
        """
        kwargs['batched'] = True  # Only supported batched, because a single row may explode to several rows
        kwargs = self._normalize_cache_kwargs(self.dataset, kwargs)
        with processing_lock('dataset'):
            # use a default lock because check is to all datasets
            def _check_batch(batch):
                # HF datasets.map expects dict/None; filter expects bool mask, so adapt batch_check output.
                rows = self.template.map_col_to_row(batch) if isinstance(batch, Mapping) else batch
                checked = self.template.batch_check(rows)
                return [item is not None for item in checked]

            self.dataset = self.dataset.filter(_check_batch, **kwargs)

    @staticmethod
    def _load_dataset(dataset_meta: DatasetMeta, **kwargs):
        # In-memory / in-process data path
        if dataset_meta.data is not None:
            from datasets import Dataset as HFDataset
            from datasets import IterableDataset as HFIterableDataset
            d = dataset_meta.data
            if isinstance(d, (HFDataset, HFIterableDataset)):
                return d
            if isinstance(d, list):
                return HFDataset.from_list(d)
            if isinstance(d, dict):
                return HFDataset.from_dict(d)
            if callable(d):
                cls = HFIterableDataset if kwargs.get('streaming') else HFDataset
                return cls.from_generator(d)
            raise ValueError(f'DatasetMeta.data must be list, dict, callable, or HF Dataset/IterableDataset, '
                             f'got {type(d).__name__}')

        dataset_id = dataset_meta.dataset_id
        subset_name = dataset_meta.subset_name
        split = dataset_meta.split
        with processing_lock(dataset_meta.get_id()):
            if os.path.exists(dataset_id):
                streaming = kwargs.get('streaming', False)
                num_proc = kwargs.get('num_proc', 1)
                kwargs['split'] = 'train'
                if streaming:
                    kwargs['streaming'] = True
                else:
                    kwargs['num_proc'] = num_proc
                load_kwargs = {}
                if os.path.isdir(dataset_id):
                    files = os.listdir(dataset_id)
                    if not files:
                        raise ValueError(f'Cannot load dataset from empty directory: {dataset_id}')
                    filename_for_ext = files[0]
                    load_kwargs['data_dir'] = dataset_id
                else:
                    filename_for_ext = dataset_id
                    load_kwargs['data_files'] = dataset_id
                ext = os.path.splitext(filename_for_ext)[1].lstrip('.')
                file_type = {'jsonl': 'json', 'txt': 'text'}.get(ext) or ext
                if file_type == 'csv':
                    kwargs['na_filter'] = False
                dataset = load_dataset(file_type, **load_kwargs, **kwargs)
            else:
                dataset = HubOperation.load_dataset(dataset_id, subset_name, split, **kwargs)

        # fix: Some dataset sources return DatasetDict instead of Dataset, which breaks downstream select/map calls.
        # fix: Normalize split resolution here (target split first, then train) and fail early with a clear error.
        if isinstance(dataset, DatasetDict):
            if split in dataset:
                dataset = dataset[split]
            elif 'train' in dataset:
                dataset = dataset['train']
            else:
                available_splits = list(dataset.keys())
                raise KeyError(f"Split '{split}' not found for dataset '{dataset_id}'. "
                               f'Available splits: {available_splits}')

        if hasattr(dataset, 'to_hf_dataset'):
            dataset = dataset.to_hf_dataset()

        if isinstance(dataset_meta.data_slice, Iterable) and hasattr(dataset, '__len__'):

            iter_list = []
            _data_len = len(dataset)
            for idx in dataset_meta.data_slice:
                if idx >= _data_len:
                    # Prevent out of range, repeat sampling
                    idx = idx % _data_len
                iter_list.append(idx)

            dataset = dataset.select(iter_list)
        return dataset

    @remote_function()
    def cast_column(self, column: str, decode: bool = True) -> None:
        """Cast an image/audio column's decode mode.

        Useful for setting ``decode=False`` before ``.map()`` to keep media
        as raw bytes and avoid expensive PIL encode/decode round-trips.
        """
        from datasets import Image as ImageFeature
        for key in list(self.datasets.keys()):
            self.datasets[key] = self.datasets[key].cast_column(column, ImageFeature(decode=decode))
        if len(self.datasets) == 1:
            self.dataset = self.datasets[next(iter(self.datasets.keys()))]

    @remote_function()
    def map(self,
            preprocess_func: Union[Preprocessor, Callable, str, Type[Preprocessor]],
            dataset_meta: DatasetMeta = None,
            init_args: Dict[str, Any] = None,
            **kwargs) -> None:
        """An inplace method to operate or transform the dataset.

        Args:
            preprocess_func: A preprocess function, or a `Preprocessor` class/instance, or a preprocessor plugin name.
            dataset_meta: The dataset_meta information of the loaded dataset.
            init_args: The init args to construct the preprocessor.
            **kwargs: The kwargs of the `datasets.map`.
        """
        init_args = init_args or {}
        preprocess_func = construct_class(preprocess_func, Preprocessor, twinkle.preprocessor, **init_args)
        kwargs['batched'] = True

        if self._mixed:
            self.dataset = self.dataset.map(preprocess_func, **self._normalize_cache_kwargs(self.dataset, kwargs))
        else:
            if dataset_meta is None:
                assert len(self.datasets) == 1
                key = next(iter(self.datasets.keys()))
            else:
                key = dataset_meta.get_id()
            with processing_lock(key):
                kw = self._normalize_cache_kwargs(self.datasets[key], kwargs)
                self.datasets[key] = self.datasets[key].map(preprocess_func, **kw)
            if len(self.datasets) == 1:
                self.dataset = self.datasets[key]

    @remote_function()
    def filter(self,
               filter_func: Union[Callable, str, Type[DataFilter], DataFilter],
               dataset_meta: DatasetMeta = None,
               init_args: Dict[str, Any] = None,
               **kwargs) -> None:
        """An inplace method to operate or transform the dataset.

        Args:
            filter_func: A filter function, or a `DataFilter` class name, or a filter plugin name.
            dataset_meta: The dataset_meta information of the loaded dataset.
            init_args: The init args to construct the filter.
            **kwargs: The kwargs of the `datasets.map`.
        """
        init_args = init_args or {}
        filter_func = construct_class(filter_func, DataFilter, twinkle.preprocessor, **init_args)
        if self._mixed:
            kwargs['batched'] = False
            self.dataset = self.dataset.filter(filter_func, **kwargs)
        else:
            if dataset_meta is None:
                assert len(self.datasets) == 1
                key = next(iter(self.datasets.keys()))
            else:
                key = dataset_meta.get_id()
            kwargs['batched'] = False
            with processing_lock(key):
                self.datasets[key] = self.datasets[key].filter(filter_func, **kwargs)
            if len(self.datasets) == 1:
                self.dataset = self.datasets[key]

    @remote_function()
    def add_dataset(self, dataset_meta: DatasetMeta, **kwargs):
        """Add a new dataset.

        Args:
            dataset_meta: The dataset_meta information of the loaded dataset.
        """
        trust_remote_code = bool(os.environ.get('TWINKLE_TRUST_REMOTE_CODE', '1'))
        if not trust_remote_code:
            kwargs['trust_remote_code'] = False
        dataset = self._load_dataset(dataset_meta, **kwargs)
        self.datasets[dataset_meta.get_id()] = dataset
        if len(self.datasets) == 1:
            self.dataset = dataset

    @remote_function()
    def mix_dataset(self, interleave=True):
        """Mix the datasets if `add_dataset` was called.

        Args:
            interleave: Whether to interleave the dataset, or concatenate the dataset.
        """
        if len(self.datasets) > 1:
            dataset_types = [isinstance(ds, IterableDataset) for ds in self.datasets]
            assert all(
                dataset_types) or not any(dataset_types), 'All datasets must be all streaming=True or streaming=False'
            if not any(dataset_types):
                dsets = list(self.datasets.values())
                # Align features
                ref_features = dsets[0].features
                aligned = []
                for ds in dsets:
                    if ds.features != ref_features:
                        ds = ds.cast(ref_features)
                    aligned.append(ds)
            else:
                aligned = list(self.datasets.values())
            if interleave:
                self.dataset = interleave_datasets(aligned)
            else:
                self.dataset = concatenate_datasets(aligned)
            self._mixed = True

    @remote_function()
    def save_as(self,
                output_path: str,
                format: Optional[str] = None,
                batch_size: int = 1000,
                mode: str = 'immediate',
                **kwargs) -> None:
        """Save the merged dataset to a local file.

        Args:
            output_path: Target file path. Extension determines format if `format` is None.
            format: One of 'jsonl', 'json', 'csv', 'parquet'. Auto-detected from extension if None.
            batch_size: Batch size for buffered writing.
            mode: 'immediate' to save all data now; 'training' to write-through as data is
                consumed by __iter__/__getitem__ — call flush_save() when training ends.
            **kwargs: Extra args passed to the underlying HF export method (immediate bulk only).
        """
        if self.dataset is None:
            raise ValueError('No dataset to save.')
        if len(self.datasets) > 1 and not self._mixed:
            raise ValueError('Call mix_dataset() before save_as() when multiple datasets are loaded.')

        fmt = format or self._infer_format(output_path)
        if fmt not in ('jsonl', 'json', 'csv', 'parquet'):
            raise ValueError(f"Unsupported format: '{fmt}'. Use jsonl/json/csv/parquet.")

        dir_path = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(dir_path, exist_ok=True)

        if mode == 'training':
            self._save_state = _SaveState(output_path, fmt, batch_size)
            return

        if self._should_materialize():
            self._save_incremental(output_path, fmt, batch_size)
        else:
            self._save_bulk(output_path, fmt, **kwargs)

    @remote_function()
    def flush_save(self) -> None:
        """Finalize and close the training-mode writer opened by save_as(mode='training')."""
        state = getattr(self, '_save_state', None)
        if state is not None:
            state.close()
            self._save_state = None

    def _write_through(self, row):
        """If training-mode save is active, persist the row."""
        state = getattr(self, '_save_state', None)
        if state is not None:
            state.write(row)
        return row

    @staticmethod
    def _infer_format(path: str) -> str:
        ext = os.path.splitext(path)[1].lstrip('.').lower()
        return {
            'jsonl': 'jsonl',
            'json': 'jsonl',
            'csv': 'csv',
            'parquet': 'parquet',
            'pq': 'parquet'
        }.get(ext, 'jsonl')

    def _should_materialize(self) -> bool:
        if isinstance(self.dataset, IterableDataset):
            return True
        if hasattr(self, 'do_encode') and self.do_encode:
            return True
        if getattr(self, '_lazy_map_ops', None) or getattr(self, '_global_map_ops', None):
            return True
        return False

    def _save_bulk(self, path: str, fmt: str, **kwargs) -> None:
        if fmt in ('jsonl', 'json'):
            self.dataset.to_json(path, **kwargs)
        elif fmt == 'csv':
            self.dataset.to_csv(path, **kwargs)
        elif fmt == 'parquet':
            self.dataset.to_parquet(path, **kwargs)

    def _save_incremental(self, path: str, fmt: str, batch_size: int) -> None:
        iterator = self._row_iterator()
        if fmt in ('jsonl', 'json'):
            self._write_jsonl(path, iterator)
        elif fmt == 'csv':
            self._write_csv(path, iterator, batch_size)
        elif fmt == 'parquet':
            self._write_parquet(path, iterator, batch_size)

    def _row_iterator(self):
        if isinstance(self.dataset, IterableDataset):
            yield from self.dataset
        else:
            for i in range(len(self)):
                yield self[i]

    @staticmethod
    def _write_jsonl(path: str, iterator) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            for row in iterator:
                f.write(_json.dumps(row, ensure_ascii=False, default=_default_serializer) + '\n')

    @staticmethod
    def _write_csv(path: str, iterator, batch_size: int) -> None:
        import pandas as pd
        first = True
        batch: List[Dict] = []
        for row in iterator:
            batch.append(row)
            if len(batch) >= batch_size:
                pd.DataFrame(batch).to_csv(path, mode='a', header=first, index=False)
                first = False
                batch = []
        if batch:
            pd.DataFrame(batch).to_csv(path, mode='a', header=first, index=False)

    @staticmethod
    def _write_parquet(path: str, iterator, batch_size: int) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        writer = None
        batch: List[Dict] = []
        for row in iterator:
            batch.append(row)
            if len(batch) >= batch_size:
                table = pa.Table.from_pylist(batch)
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema)
                writer.write_table(table)
                batch = []
        if batch:
            table = pa.Table.from_pylist(batch)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
        if writer:
            writer.close()

    @remote_function()
    def __getitem__(self, idx):
        item = self.dataset[idx]
        self._write_through(item)
        return item

    @remote_function()
    def __len__(self):
        return len(self.dataset)


def _default_serializer(obj):
    """Handle numpy types in JSON serialization."""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


_SENTINEL = object()


class _SaveState:
    """Async persistent writer for training-mode save_as.

    Writes happen on a background daemon thread so the training loop is never blocked.
    Uses fcntl file-lock for cross-process safety when multiple ranks write one file.
    """

    def __init__(self, path: str, fmt: str, batch_size: int):

        self._path = path
        self._fmt = fmt
        self._batch_size = batch_size
        self._queue: Queue = Queue(maxsize=batch_size * 4)
        self._lock = PosixFileLock(path + '.lock')
        self._error = None

        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def write(self, row: Dict) -> None:
        self._queue.put(row)

    def close(self) -> None:
        self._queue.put(_SENTINEL)
        self._thread.join()
        self._lock.close()
        if self._error:
            raise self._error

    def _writer_loop(self) -> None:
        try:
            if self._fmt in ('jsonl', 'json'):
                self._loop_jsonl()
            elif self._fmt == 'csv':
                self._loop_csv()
            elif self._fmt == 'parquet':
                self._loop_parquet()
        except Exception as e:
            self._error = e

    def _acquire_lock(self):
        self._lock.acquire()

    def _release_lock(self):
        self._lock.release()

    def _loop_jsonl(self) -> None:
        buffer: List[str] = []

        def _flush(f):
            if not buffer:
                return
            payload = ''.join(buffer)
            self._acquire_lock()
            try:
                f.write(payload)
                f.flush()
            finally:
                self._release_lock()
            buffer.clear()

        with open(self._path, 'a', encoding='utf-8') as f:
            while True:
                item = self._queue.get()
                if item is _SENTINEL:
                    _flush(f)
                    return
                buffer.append(_json.dumps(item, ensure_ascii=False, default=_default_serializer) + '\n')
                if len(buffer) >= self._batch_size:
                    _flush(f)

    def _loop_csv(self) -> None:
        import pandas as pd
        header_written = False
        buffer: List[Dict] = []
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                if buffer:
                    self._acquire_lock()
                    try:
                        pd.DataFrame(buffer).to_csv(self._path, mode='a', header=not header_written, index=False)
                    finally:
                        self._release_lock()
                return
            buffer.append(item)
            if len(buffer) >= self._batch_size:
                self._acquire_lock()
                try:
                    pd.DataFrame(buffer).to_csv(self._path, mode='a', header=not header_written, index=False)
                    header_written = True
                finally:
                    self._release_lock()
                buffer = []

    def _loop_parquet(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        writer = None
        buffer: List[Dict] = []
        try:
            while True:
                item = self._queue.get()
                if item is _SENTINEL:
                    if buffer:
                        table = pa.Table.from_pylist(buffer)
                        if writer is None:
                            writer = pq.ParquetWriter(self._path, table.schema)
                        self._acquire_lock()
                        try:
                            writer.write_table(table)
                        finally:
                            self._release_lock()
                    return
                buffer.append(item)
                if len(buffer) >= self._batch_size:
                    table = pa.Table.from_pylist(buffer)
                    if writer is None:
                        writer = pq.ParquetWriter(self._path, table.schema)
                    self._acquire_lock()
                    try:
                        writer.write_table(table)
                    finally:
                        self._release_lock()
                    buffer = []
        finally:
            if writer:
                writer.close()
