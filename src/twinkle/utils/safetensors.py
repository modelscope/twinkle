import json
import numpy as np
import os
import torch
from functools import partial
from pathlib import Path
from typing import Any, Literal

from .device_mesh import is_last_rank, is_master


class LazyTensor:

    def __init__(self, tensor=None, loader=None):
        """You need to provide a tensor or loader"""
        self.tensor = tensor
        self.loader = loader

    def load(self):
        if self.tensor is None:
            return self.loader()
        return self.tensor


_STATE_TYPE = '__twinkle_state_type__'


def save_state_dict(state: Any, path: str) -> None:
    """Safely persist a nested checkpoint state without pickle serialization."""
    from safetensors.torch import save_file

    tensors = {}
    manifest = _encode_state(state, tensors)
    save_file(tensors, path)
    with open(f'{path}.json', 'w') as f:
        json.dump(manifest, f, separators=(',', ':'))


def load_state_dict(path: str) -> Any:
    """Load a checkpoint written by :func:`save_state_dict`."""
    from safetensors.torch import load_file

    with open(f'{path}.json') as f:
        manifest = json.load(f)
    return _decode_state(manifest, load_file(path, device='cpu'))


def _encode_state(value: Any, tensors: dict[str, torch.Tensor]) -> Any:
    if isinstance(value, torch.Tensor):
        name = f'tensor_{len(tensors)}'
        tensors[name] = value.detach().cpu().contiguous().clone()
        return {_STATE_TYPE: 'tensor', 'name': name}
    if isinstance(value, np.ndarray):
        name = f'tensor_{len(tensors)}'
        tensors[name] = torch.from_numpy(value).contiguous().clone()
        return {_STATE_TYPE: 'ndarray', 'name': name, 'dtype': value.dtype.str}
    if isinstance(value, np.generic):
        return {_STATE_TYPE: 'numpy_scalar', 'dtype': value.dtype.str, 'value': value.item()}
    if isinstance(value, dict):
        return {
            _STATE_TYPE: 'dict',
            'items': [[_encode_state(k, tensors), _encode_state(v, tensors)] for k, v in value.items()]
        }
    if isinstance(value, tuple):
        return {_STATE_TYPE: 'tuple', 'items': [_encode_state(v, tensors) for v in value]}
    if isinstance(value, list):
        return {_STATE_TYPE: 'list', 'items': [_encode_state(v, tensors) for v in value]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f'Unsupported checkpoint value type: {type(value).__name__}')


def _decode_state(value: Any, tensors: dict[str, torch.Tensor]) -> Any:
    if not isinstance(value, dict) or _STATE_TYPE not in value:
        return value

    value_type = value[_STATE_TYPE]
    if value_type == 'tensor':
        return tensors[value['name']]
    if value_type == 'ndarray':
        return tensors[value['name']].numpy().astype(np.dtype(value['dtype']), copy=False)
    if value_type == 'numpy_scalar':
        return np.dtype(value['dtype']).type(value['value'])
    if value_type == 'dict':
        return {_decode_state(k, tensors): _decode_state(v, tensors) for k, v in value['items']}
    if value_type == 'tuple':
        return tuple(_decode_state(v, tensors) for v in value['items'])
    if value_type == 'list':
        return [_decode_state(v, tensors) for v in value['items']]
    raise ValueError(f'Unknown checkpoint value type: {value_type}')


class SafetensorLazyLoader:

    def __init__(self, hf_model_dir: str, is_peft_format: bool = False):
        self.hf_model_dir = hf_model_dir
        self.is_peft_format = is_peft_format
        self._weight_map = {}
        self._file_handles = {}
        self._load_index()

    def _open_file(self, filename: str):
        """Open a safetensors file if not already open."""
        from safetensors.torch import safe_open, save_file
        if filename not in self._file_handles:
            file_path = os.path.join(self.hf_model_dir, filename)
            self._file_handles[filename] = safe_open(file_path, framework='pt')
        return self._file_handles[filename]

    def _load_index(self):
        """Load the model index file to get weight map."""
        from safetensors.torch import safe_open, save_file
        index_path = os.path.join(self.hf_model_dir, 'model.safetensors.index.json')

        if os.path.exists(index_path):
            with open(index_path) as f:
                self._index_file = json.load(f)
                self._weight_map = self._index_file.get('weight_map', {})
        else:
            if self.is_peft_format:
                safetensors_fname = 'adapter_model.safetensors'
            else:
                safetensors_fname = 'model.safetensors'
            # Single file model
            safetensors_file = os.path.join(self.hf_model_dir, safetensors_fname)
            if os.path.exists(safetensors_file):
                with safe_open(safetensors_file, framework='pt') as f:
                    for key in f.keys():
                        self._weight_map[key] = safetensors_fname

    def get_state_dict(self):
        res = {}
        for k in self._weight_map.keys():
            res[k] = LazyTensor(loader=partial(self._load_tensor, key=k))
        return res

    def keys(self):
        """The checkpoint's tensor keys -- the keyspace a streaming transform iterates over."""
        return list(self._weight_map.keys())

    def partition_keyspace(self, world_size: int):
        """Split the keyspace into ``world_size`` disjoint key lists for data-parallel streaming.

        Keys are grouped by module path (the key minus its trailing ``.weight``/``.bias``/... segment)
        and whole groups are dealt round-robin, so every tensor of one module lands on the same worker:
        a per-module transform (a LoRA merge folds a module's weight and bias together) then never needs
        a tensor held by another worker. The parts are pairwise disjoint and their union is the full
        keyspace; ``world_size <= 1`` returns the whole keyspace as a single part.
        """
        if world_size <= 1:
            return [self.keys()]
        groups = {}
        for key in self._weight_map:
            module_path = key.rsplit('.', 1)[0] if '.' in key else key
            groups.setdefault(module_path, []).append(key)
        partitions = [[] for _ in range(world_size)]
        for index, module_keys in enumerate(groups.values()):
            partitions[index % world_size].extend(module_keys)
        return partitions

    def _load_tensor(self, key):
        filename = self._weight_map[key]
        file_handle = self._open_file(filename)
        return file_handle.get_tensor(key)

    def close(self):
        self._file_handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class StreamingSafetensorSaver:

    def __init__(
        self,
        save_dir,
        max_shard_size: str = '5GB',
        save_rank: Literal['master', 'last'] = 'last',
        is_peft_format: bool = False,
        *,
        rank: int = None,
        world_size: int = 1,
    ) -> None:
        self.save_dir = save_dir
        if isinstance(max_shard_size, str):
            import re
            match = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*(B|KB|MB|GB)\s*', max_shard_size.upper())
            if match is None:
                raise ValueError(f'Invalid max_shard_size: {max_shard_size}')
            multipliers = {'B': 1, 'KB': 1000, 'MB': 1000**2, 'GB': 1000**3}
            max_shard_size = int(float(match.group(1)) * multipliers[match.group(2)])
        elif isinstance(max_shard_size, int):
            # Preserve the historical API: integer values are expressed in GB.
            max_shard_size *= 1000**3
        else:
            raise TypeError('max_shard_size must be a byte count or a size string such as "5GB".')
        self.max_shard_size = max_shard_size
        self.current_shard = {}
        self.current_shard_size = 0
        self.total_size = 0
        self.shard_index = 1
        self.weight_map = {}
        # Multi-writer mode: an explicit (rank, world_size) makes every rank write its OWN shards under
        # rank-tagged names, and finalize() returns a manifest for the driver to assemble into one global
        # index (see assemble_sharded_index). It is decoupled from the env-based is_master/is_last_rank,
        # which every Ray worker would read as True (no torchrun env), so it cannot pick a single writer.
        self.rank = rank
        self.world_size = world_size
        self.multi_writer = rank is not None and world_size > 1 and not is_peft_format
        if self.multi_writer:
            self.is_save_rank = True
        else:
            self.is_save_rank = is_last_rank() if save_rank == 'last' else is_master()
        self.is_peft_format = is_peft_format
        if self.is_save_rank:
            os.makedirs(save_dir, exist_ok=True)
        self._previous_model_files = set()
        # Stale-file cleanup is a whole-directory concern; in multi-writer mode the driver does it once
        # in assemble_sharded_index (a per-rank glob would also catch the other ranks' in-flight shards).
        if self.is_save_rank and not self.is_peft_format and not self.multi_writer:
            self._previous_model_files = set(Path(save_dir).glob('model*.safetensors'))
            index_path = Path(save_dir) / 'model.safetensors.index.json'
            if index_path.exists():
                self._previous_model_files.add(index_path)

    def add_tensor(self, name, tensor):
        if not self.is_save_rank:
            return
        tensor_size = tensor.numel() * tensor.element_size()
        if (self.current_shard_size + tensor_size > self.max_shard_size and self.current_shard
                and not self.is_peft_format):
            self._save_current_shard()

        # Break shared storage (for example tied input/output embeddings) so a
        # standard Transformers state dict remains valid for safetensors.
        self.current_shard[name] = tensor.detach().cpu().contiguous().clone()
        self.current_shard_size += tensor_size

    def _save_current_shard(self, shard_filename: str = None):
        from safetensors.torch import safe_open, save_file
        if not self.current_shard:
            return
        if shard_filename is None:
            if self.is_peft_format:
                shard_filename = 'adapter_model.safetensors'
            elif self.multi_writer:
                # Rank-tagged so concurrent Ray workers writing one save_dir never collide; `@@@@@` is
                # this rank's local shard total, unknown until finalize().
                shard_filename = f'model-{self.rank:05d}-{self.shard_index:05d}-of-@@@@@.safetensors'
            else:
                shard_filename = f'model-{self.shard_index:05d}-of-?????.safetensors'
        shard_path = os.path.join(self.save_dir, shard_filename)
        save_file(self.current_shard, str(shard_path))
        for key in self.current_shard.keys():
            self.weight_map[key] = shard_filename

        self.total_size += self.current_shard_size
        self.current_shard = {}
        self.current_shard_size = 0
        self.shard_index += 1

    def finalize(self):
        if not self.is_save_rank:
            return
        if self.current_shard:
            self._save_current_shard()
        if self.is_peft_format:
            return
        if self.multi_writer:
            return self._finalize_multi_writer()
        total_shards = self.shard_index - 1
        # rename `?????`
        for i in range(1, total_shards + 1):
            old_path = os.path.join(self.save_dir, f'model-{i:05d}-of-?????.safetensors')
            if total_shards == 1:
                new_name = 'model.safetensors'
            else:
                new_name = f'model-{i:05d}-of-{total_shards:05d}.safetensors'
            new_path = os.path.join(self.save_dir, new_name)
            if os.path.exists(old_path):
                os.rename(old_path, new_path)

        if total_shards > 1:
            updated_weight_map = {}
            for key, filename in self.weight_map.items():
                new_filename = filename.replace('?????', f'{total_shards:05d}')
                updated_weight_map[key] = new_filename

            self._write_index(self.save_dir, updated_weight_map, self.total_size)

        if total_shards == 1:
            current_files = {Path(self.save_dir) / 'model.safetensors'}
        else:
            current_files = {
                Path(self.save_dir) / f'model-{index:05d}-of-{total_shards:05d}.safetensors'
                for index in range(1, total_shards + 1)
            }
            current_files.add(Path(self.save_dir) / 'model.safetensors.index.json')
        for stale_path in self._previous_model_files - current_files:
            stale_path.unlink(missing_ok=True)

    def _finalize_multi_writer(self):
        """Rename this rank's shards to their local totals and return a manifest for the driver.

        Each rank only sees its own shards, so it cannot write the index (the global shard count and the
        key->file map span every rank). The driver collects all manifests and calls
        :meth:`assemble_sharded_index` to renumber them into one globally-contiguous layout. The returned
        ``files`` are ordered by local shard index, and ``weight_map`` maps each key to its file in
        ``files`` (both use the rank-tagged local name, the input the driver renumbers from).
        """
        local_total = self.shard_index - 1
        files = []
        for i in range(1, local_total + 1):
            temp_name = f'model-{self.rank:05d}-{i:05d}-of-@@@@@.safetensors'
            final_name = f'model-{self.rank:05d}-{i:05d}-of-{local_total:05d}.safetensors'
            old_path = os.path.join(self.save_dir, temp_name)
            new_path = os.path.join(self.save_dir, final_name)
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
            files.append(final_name)
        weight_map = {key: name.replace('@@@@@', f'{local_total:05d}') for key, name in self.weight_map.items()}
        return {'rank': self.rank, 'files': files, 'weight_map': weight_map, 'total_size': self.total_size}

    @staticmethod
    def assemble_sharded_index(save_dir, manifests):
        """Renumber every rank's shards into one global HF layout and write the single index.

        ``manifests`` is the list returned by each rank's :meth:`_finalize_multi_writer`. Files are
        concatenated in rank order, so the ``g``-th file across all ranks becomes global shard ``g``,
        renamed to ``model-{g:05d}-of-{global_total:05d}.safetensors``; the merged weight_map is rewritten
        to those new names. Stale ``model*.safetensors`` from a previous run are removed here -- the
        whole-directory cleanup each rank deliberately skipped (a per-rank glob would also catch the other
        ranks' in-flight shards).
        """
        manifests = sorted(manifests, key=lambda manifest: manifest['rank'])
        ordered = [name for manifest in manifests for name in manifest['files']]
        global_total = len(ordered)
        total_size = sum(manifest['total_size'] for manifest in manifests)

        if global_total == 1:
            rename_map = {ordered[0]: 'model.safetensors'}
        else:
            rename_map = {name: f'model-{g:05d}-of-{global_total:05d}.safetensors' for g, name in enumerate(ordered, 1)}
        for old_name, new_name in rename_map.items():
            old_path = os.path.join(save_dir, old_name)
            new_path = os.path.join(save_dir, new_name)
            if os.path.exists(old_path):
                os.rename(old_path, new_path)

        if global_total > 1:
            weight_map = {}
            for manifest in manifests:
                for key, old_name in manifest['weight_map'].items():
                    weight_map[key] = rename_map[old_name]
            StreamingSafetensorSaver._write_index(save_dir, weight_map, total_size)
            current_files = set(rename_map.values()) | {'model.safetensors.index.json'}
        else:
            current_files = set(rename_map.values())
            # A previous multi-shard run may leave an index that no longer matches this single file.
            (Path(save_dir) / 'model.safetensors.index.json').unlink(missing_ok=True)
        for stale_path in Path(save_dir).glob('model*.safetensors'):
            if stale_path.name not in current_files:
                stale_path.unlink(missing_ok=True)

    @staticmethod
    def _write_index(save_dir, weight_map, total_size):
        index = {'metadata': {'total_size': total_size}, 'weight_map': weight_map}
        index_path = os.path.join(save_dir, 'model.safetensors.index.json')
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
