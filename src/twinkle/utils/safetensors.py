import json
import os
from pathlib import Path
from functools import partial
from typing import Literal

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
        self.is_save_rank = is_last_rank() if save_rank == 'last' else is_master()
        self.is_peft_format = is_peft_format
        if self.is_save_rank:
            os.makedirs(save_dir, exist_ok=True)
        self._previous_model_files = set()
        if self.is_save_rank and not self.is_peft_format:
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

            self._save_index(updated_weight_map)

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

    def _save_index(self, weight_map):
        index = {'metadata': {'total_size': self.total_size}, 'weight_map': weight_map}

        index_path = os.path.join(self.save_dir, 'model.safetensors.index.json')
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
