# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import twinkle
from twinkle import remote_class, remote_function
from twinkle.preprocessor import DataFilter, Preprocessor
from twinkle.utils import construct_class, processing_lock
from .base import Dataset, DatasetMeta


@remote_class(execute='first')
class LazyDataset(Dataset):
    """A lazy dataset wrapper that defers map, mix, and encode operations.

    This class prevents OOM for multimodal datasets by deferring expensive
    operations (preprocessing, encoding) to __getitem__ time.

    Features:
        - Lazy map: Records per-dataset operations, applies in __getitem__
        - Lazy mix: Records strategy, resolves indices lazily
        - Lazy encode: Applies encoding in __getitem__
        - Eager filter: Must execute immediately (index mapping required)
    """

    def __init__(self, dataset_meta: DatasetMeta = None, **kwargs):
        super().__init__(dataset_meta, **kwargs)
        self._init_lazy_state()

    def _init_lazy_state(self):
        """Initialize or reset lazy operation state."""
        self.do_encode = False
        self.do_check = False
        self.encode_kwargs = {}
        self.add_generation_prompt = False

        # Per-dataset lazy map ops: {dataset_id: [(func, kwargs), ...]}
        self._lazy_map_ops: Dict[str, List[Tuple[Callable, Dict]]] = {}
        # Global lazy map ops (applied after mix): [(func, kwargs), ...]
        self._global_map_ops: List[Tuple[Callable, Dict]] = []

        # Mix state. `_mixed` is the base class's flag and is reused verbatim -- a second field for
        # the same concept would have to be kept in sync with it by hand, and inherited methods that
        # read `_mixed` would silently see the wrong value.
        self._mixed = False
        self._mix_interleave = True
        self._length_cache: Optional[int] = None

    def _invalidate_length_cache(self):
        """Invalidate cached length."""
        self._length_cache = None

    def _get_dataset_info(self) -> List[Tuple[str, int]]:
        """Get list of (dataset_key, length) tuples."""
        return [(key, len(ds)) for key, ds in self.datasets.items()]

    def _target_keys(self, dataset_meta: DatasetMeta = None) -> List[str]:
        """Sub-datasets an op applies to: the named one, or every loaded one.

        Lazy ops are always recorded against ``self.datasets`` keys (never against the merged view),
        because mixing is deferred and ``__getitem__`` resolves indices back to the sub-datasets.
        """
        if dataset_meta is not None:
            return [dataset_meta.get_id()]
        return list(self.datasets.keys())

    @remote_function()
    def map(self,
            preprocess_func: Union[Preprocessor, Callable, str, Type[Preprocessor]],
            dataset_meta: DatasetMeta = None,
            init_args: Dict[str, Any] = None,
            **kwargs) -> None:
        """Record map operation for lazy execution.

        Args:
            preprocess_func: A preprocess function or Preprocessor class/instance.
            dataset_meta: Target dataset. If None and multiple datasets exist,
                applies to all datasets (before mix) or globally (after mix).
            init_args: Init args for constructing the preprocessor.
            **kwargs: Additional kwargs stored for reference.
        """
        init_args = init_args or {}
        func = construct_class(preprocess_func, Preprocessor, twinkle.preprocessor, **init_args)

        if self._mixed:
            self._global_map_ops.append((func, kwargs))
            return
        for key in self._target_keys(dataset_meta):
            self._lazy_map_ops.setdefault(key, []).append((func, kwargs))

    @remote_function()
    def filter(self,
               filter_func: Union[Callable, str, Type[DataFilter], DataFilter],
               dataset_meta: DatasetMeta = None,
               init_args: Dict[str, Any] = None,
               **kwargs) -> None:
        """Execute filter eagerly (index mapping requires knowing valid indices).

        Not delegated to ``Dataset.filter``: that one filters the merged ``self.dataset`` once
        mixing has happened, but lazy mixing never materializes a merged view, so it would filter
        only the first sub-dataset while ``__getitem__`` keeps reading all of them. Filtering each
        sub-dataset here matches how ``map`` targets them and keeps ``_resolve_index`` correct.
        """
        init_args = init_args or {}
        filter_func = construct_class(filter_func, DataFilter, twinkle.preprocessor, **init_args)
        kwargs['batched'] = False
        for key in self._target_keys(dataset_meta):
            with processing_lock(key):
                self.datasets[key] = self.datasets[key].filter(filter_func, **kwargs)
        self._invalidate_length_cache()

    @remote_function()
    def add_dataset(self, dataset_meta: DatasetMeta, **kwargs):
        """Add a new dataset."""
        super().add_dataset(dataset_meta, **kwargs)
        self._invalidate_length_cache()

    @remote_function()
    def mix_dataset(self, interleave=True):
        """Record mix strategy for lazy execution.

        Only the flag is set: unlike the base class this does NOT build a merged dataset, so
        ``_merged`` stays None and ``_resolve_index`` maps global indices onto the sub-datasets.

        Args:
            interleave: True for round-robin interleaving, False for concatenation.
        """
        if len(self.datasets) > 1:
            self._mixed = True
            self._mix_interleave = interleave
            self._invalidate_length_cache()

    @remote_function()
    def encode(self, add_generation_prompt: bool = False, **kwargs):
        """Record encode operation for lazy execution."""
        assert self.template is not None
        assert self.template.truncation_strategy != 'split', (
            'Lazy tokenize does not support truncation_strategy==`split`')
        self.do_encode = True
        self.add_generation_prompt = add_generation_prompt
        self.encode_kwargs = kwargs

    @remote_function()
    def check(self, **kwargs):
        """Record check operation for lazy execution."""
        assert self.template is not None
        self.do_check = True

    def _resolve_index(self, idx: int) -> Tuple[str, int]:
        """Resolve global index to (dataset_key, local_index)."""
        dataset_info = self._get_dataset_info()

        if not self._mixed or len(dataset_info) == 1:
            return dataset_info[0][0], idx

        if self._mix_interleave:
            # Round-robin: idx 0 -> ds0[0], idx 1 -> ds1[0], idx 2 -> ds2[0], idx 3 -> ds0[1], ...
            num_datasets = len(dataset_info)
            assert num_datasets >= 1
            dataset_idx = idx % num_datasets
            local_idx = idx // num_datasets
            key, length = dataset_info[dataset_idx]
            # Wrap for uneven lengths (matches interleave_datasets behavior)
            if local_idx >= length:
                local_idx = local_idx % length
            return key, local_idx
        else:
            # Concatenate: sequential access
            offset = 0
            for key, length in dataset_info:
                if idx < offset + length:
                    return key, idx - offset
                offset += length
            raise IndexError(f'Index {idx} out of range (total length: {offset})')

    def _apply_map_op(self, item: Dict, func: Callable) -> Dict:
        """Apply a map operation to a single item."""
        if hasattr(func, 'preprocess'):
            return func.preprocess(item)
        # Callable expects batched columnar format
        columnar = {k: [v] for k, v in item.items()}
        result = func(columnar)
        return {k: v[0] for k, v in result.items()}

    def _should_materialize(self) -> bool:
        """Lazy mixing has no merged dataset to bulk-export, so rows must be produced one by one.

        Without this, ``save_as`` on a mixed LazyDataset would take the bulk path and hand
        ``self.dataset`` -- the FIRST sub-dataset -- to ``to_json``, silently writing a fraction of
        the data.
        """
        if self._mixed:
            return True
        return super()._should_materialize()

    @remote_function()
    def __getitem__(self, idx):
        dataset_key, local_idx = self._resolve_index(idx)
        item = self.datasets[dataset_key][local_idx]

        # Apply per-dataset lazy map operations
        for func, _ in self._lazy_map_ops.get(dataset_key, []):
            item = self._apply_map_op(item, func)

        # Apply global lazy map operations (post-mix)
        for func, _ in self._global_map_ops:
            item = self._apply_map_op(item, func)

        # Lazy encode
        if self.do_encode:
            encoded = self.template.encode(item, add_generation_prompt=self.add_generation_prompt, **self.encode_kwargs)
            # Preserve extra fields not produced by encoding
            for key in item:
                if key not in encoded:
                    encoded[key] = item[key]
            item = encoded
        elif self.do_check:
            item = self.template.check(item)

        self._write_through(item)
        return item

    @remote_function()
    def __len__(self):
        if self._length_cache is not None:
            return self._length_cache

        dataset_info = self._get_dataset_info()

        if not self._mixed:
            # Not mixed: only first dataset is accessible (matches base Dataset behavior)
            self._length_cache = dataset_info[0][1]
        elif self._mix_interleave:
            # Interleave uses max_length * num_datasets (with wrap-around)
            max_len = max(length for _, length in dataset_info)
            self._length_cache = len(dataset_info) * max_len
        else:
            # Concatenate: sum of all lengths
            self._length_cache = sum(length for _, length in dataset_info)

        return self._length_cache
