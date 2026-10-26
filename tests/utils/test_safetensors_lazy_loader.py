# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for SafetensorLazyLoader (index parsing, lazy reads, keyspace partitioning)."""
import json

import pytest
import torch
from safetensors.torch import save_file

from twinkle.utils.safetensors import SafetensorLazyLoader


def _write_single_file(directory, filename='model.safetensors'):
    save_file({'layer.weight': torch.arange(4.0), 'layer.bias': torch.ones(2) * 7},
              str(directory / filename))


class TestSingleFileModel:

    def test_keys_and_lazy_tensors(self, tmp_path):
        _write_single_file(tmp_path)
        loader = SafetensorLazyLoader(str(tmp_path))
        assert set(loader.keys()) == {'layer.weight', 'layer.bias'}
        state = loader.get_state_dict()
        assert torch.equal(state['layer.weight'].load(), torch.arange(4.0))
        assert torch.equal(state['layer.bias'].load(), torch.ones(2) * 7)

    def test_peft_format_reads_adapter_file(self, tmp_path):
        _write_single_file(tmp_path, filename='adapter_model.safetensors')
        loader = SafetensorLazyLoader(str(tmp_path), is_peft_format=True)
        assert set(loader.keys()) == {'layer.weight', 'layer.bias'}


class TestShardedModel:

    def test_weight_map_splits_across_shards(self, tmp_path):
        save_file({'a.weight': torch.zeros(2)}, str(tmp_path / 'shard0.safetensors'))
        save_file({'b.weight': torch.zeros(3)}, str(tmp_path / 'shard1.safetensors'))
        index = {'weight_map': {'a.weight': 'shard0.safetensors', 'b.weight': 'shard1.safetensors'}}
        (tmp_path / 'model.safetensors.index.json').write_text(json.dumps(index))
        loader = SafetensorLazyLoader(str(tmp_path))
        assert set(loader.keys()) == {'a.weight', 'b.weight'}
        state = loader.get_state_dict()
        assert state['a.weight'].load().shape == (2, )
        assert state['b.weight'].load().shape == (3, )


class TestPartitionKeyspace:

    def _loader_with_keys(self, tmp_path, keys):
        save_file({key: torch.zeros(1) for key in keys}, str(tmp_path / 'model.safetensors'))
        return SafetensorLazyLoader(str(tmp_path))

    def test_world_size_one_returns_everything(self, tmp_path):
        loader = self._loader_with_keys(tmp_path, ['a.weight', 'a.bias', 'b.weight'])
        assert loader.partition_keyspace(1) == [['a.bias', 'a.weight', 'b.weight']]

    def test_modules_stay_together_across_partitions(self, tmp_path):
        loader = self._loader_with_keys(tmp_path, ['a.weight', 'a.bias', 'b.weight'])
        parts = loader.partition_keyspace(2)
        assert len(parts) == 2
        flat = [key for part in parts for key in part]
        assert sorted(flat) == ['a.bias', 'a.weight', 'b.weight']
        # Every part holds whole modules only.
        for part in parts:
            modules = {key.rsplit('.', 1)[0] for key in part}
            all_keys_of_module = {key for key in flat if key.rsplit('.', 1)[0] in modules}
            assert all_keys_of_module == set(part)

    def test_bare_keys_form_their_own_group(self, tmp_path):
        loader = self._loader_with_keys(tmp_path, ['w', 'x', 'y'])
        parts = loader.partition_keyspace(2)
        flat = [key for part in parts for key in part]
        assert sorted(flat) == ['w', 'x', 'y']
        assert all(len(part) >= 1 for part in parts)
