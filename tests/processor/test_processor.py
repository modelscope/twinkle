# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for InputProcessor: normal, padding_free, micro_batch, multimodal, GRPO."""
from types import SimpleNamespace

import torch

import twinkle
from twinkle.processor import InputProcessor

twinkle.initialize(mode='local')


def _make_text_batch(n: int, seq_len: int = 8):
    """Synthetic text batch: input_ids, attention_mask, position_ids, labels (tensors)."""
    return [{
        'input_ids': torch.randint(1, 1000, (seq_len, )),
        'attention_mask': torch.ones(seq_len),
        'position_ids': torch.arange(seq_len),
        'labels': torch.full((seq_len, ), -100),
    } for _ in range(n)]


class TestNormalMode:
    """Normal mode: padding + collate."""

    def test_normal_padding(self):
        proc = InputProcessor(padding_free=False, padding_side='right')
        batch = _make_text_batch(3, seq_len=6)
        out = proc.collate_fn(batch)
        assert len(out) == 1
        b = out[0]
        assert b['input_ids'].shape == (3, 6)
        assert b['attention_mask'].shape == (3, 6)

    def test_padding_side_left(self):
        proc = InputProcessor(padding_free=False, padding_side='left')
        batch = _make_text_batch(2, seq_len=5)
        out = proc.collate_fn(batch)
        assert out[0]['input_ids'].shape == (2, 5)

    def test_channel_and_loss_scale_survive_collation(self):
        proc = InputProcessor(padding_free=False, padding_side='right')
        batch = _make_text_batch(2, seq_len=3)
        batch[0].update(channel='math', loss_scale=torch.tensor([1.0, 0.5, 0.0]))
        batch[1].update(channel='code', loss_scale=torch.tensor([0.0, 1.0, 1.0]))

        out = proc.to_transformers_dict(proc.collate_fn(batch))[0]

        assert out['channel'] == ['math', 'code']
        assert out['loss_scale'].tolist() == [[1.0, 0.5, 0.0], [0.0, 1.0, 1.0]]

    def test_completion_mask_padding(self):
        proc = InputProcessor(padding_free=False, padding_side='right')
        batch = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'position_ids': torch.arange(3),
                'labels': torch.tensor([-100, 10, 11]),
                'completion_mask': torch.tensor([0, 1, 1]),
            },
            {
                'input_ids': torch.tensor([4, 5]),
                'position_ids': torch.arange(2),
                'labels': torch.tensor([-100, 12]),
                'completion_mask': torch.tensor([0, 1]),
            },
        ]
        out = proc.collate_fn(batch)
        assert len(out) == 1
        b = out[0]
        assert b['completion_mask'].shape == b['input_ids'].shape == (2, 3)
        assert b['completion_mask'].tolist() == [[0, 1, 1], [0, 1, 0]]


class TestPaddingFreeMode:
    """padding_free: concatenate multiple samples into single row."""

    def test_padding_free_concatenate(self):
        proc = InputProcessor(padding_free=True)
        batch = _make_text_batch(3, seq_len=4)
        out = proc.collate_fn(batch)
        assert len(out) == 1
        b = out[0]
        assert b['input_ids'].shape == (1, 12)
        assert b['labels'].shape == (1, 12)

    def test_padding_free_unpacks_loss_scale_and_keeps_channels(self):
        proc = InputProcessor(padding_free=True)
        batch = _make_text_batch(2, seq_len=3)
        batch[0].update(channel='math', loss_scale=torch.tensor([1.0, 0.5, 0.0]))
        batch[1].update(channel='code', loss_scale=torch.tensor([0.0, 1.0, 1.0]))
        packed = proc.collate_fn(batch)[0]

        inputs, _ = proc.unpack_packed_sequences(packed)

        assert inputs['channel'] == ['math', 'code']
        assert inputs['loss_scale'].tolist() == [[1.0, 0.5, 0.0], [0.0, 1.0, 1.0]]


class TestContextParallelMode:
    """Context parallelism keeps token-level loss metadata aligned."""

    def test_loss_scale_is_padded_and_split_with_labels(self):
        mesh = SimpleNamespace(cp_world_size=2, cp_rank=0, tp_world_size=1, sequence_parallel=False)
        proc = InputProcessor(device_mesh=mesh, framework='megatron')
        proc.device_mesh = mesh
        inputs = [{
            'input_ids': torch.arange(1, 7).unsqueeze(0),
            'attention_mask': torch.ones(1, 6),
            'position_ids': torch.arange(6).unsqueeze(0),
            'labels': torch.arange(1, 7).unsqueeze(0),
            'loss_scale': torch.arange(1, 7, dtype=torch.float32).unsqueeze(0),
        }]

        padded = proc.pad_cp(inputs)
        split = proc.split_cp(padded)[0]

        assert split['labels'].tolist() == [[1, 2, -100, -100]]
        assert split['loss_scale'].tolist() == [[1.0, 2.0, 0.0, 0.0]]


class TestMicroBatchMode:
    """micro_batch split."""

    def test_micro_batch_fixed_length(self):
        proc = InputProcessor(padding_free=False)
        batch = _make_text_batch(4, seq_len=6)
        out = proc.collate_fn(batch, micro_batch_size=2, variable_seq_lengths=False)
        assert len(out) == 2
        for b in out:
            assert b['input_ids'].shape == (2, 6)

    def test_micro_batch_variable_length(self):
        proc = InputProcessor(padding_free=False)
        batch = _make_text_batch(4, seq_len=5)
        out = proc.collate_fn(batch, micro_batch_size=2, variable_seq_lengths=True)
        assert len(out) == 2
        for b in out:
            assert b['input_ids'].shape[0] == 2


class TestMultimodalMode:
    """Multimodal: pixel_values, image_grid_thw."""

    def test_multimodal_collate(self):
        proc = InputProcessor()
        batch = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'position_ids': torch.arange(3),
                'pixel_values': torch.randn(1, 3, 32, 32),
                'image_grid_thw': torch.tensor([[1, 4, 4]]),
            },
            {
                'input_ids': torch.tensor([4, 5]),
                'position_ids': torch.arange(2),
                'pixel_values': torch.randn(1, 3, 32, 32),
                'image_grid_thw': torch.tensor([[1, 4, 4]]),
            },
        ]
        out = proc.collate_fn(batch)
        assert len(out) == 1
        b = out[0]
        assert 'input_ids' in b
        assert 'pixel_values' in b
        # 2 images x 3 channels after squeeze, cat along dim=0 -> shape[0]=6
        assert b['pixel_values'].shape[0] == 6
        assert b['image_grid_thw'].shape == (2, 3)


class TestGRPOMode:
    """GRPO: input_ids + labels."""

    def test_grpo_collate(self):
        proc = InputProcessor()
        batch = [
            {
                'input_ids': torch.tensor([1, 2, 3, 4, 5]),
                'position_ids': torch.arange(5),
                'labels': torch.tensor([-100, -100, 10, 11, 12])
            },
            {
                'input_ids': torch.tensor([6, 7, 8]),
                'position_ids': torch.arange(3),
                'labels': torch.tensor([-100, 20, 21])
            },
        ]
        out = proc.collate_fn(batch)
        assert len(out) == 1
        b = out[0]
        assert b['input_ids'].shape[0] == 2
        assert b['labels'].shape[0] == 2

    def test_grpo_padding_free(self):
        proc = InputProcessor(padding_free=True)
        batch = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'labels': torch.tensor([-100, -100, -100])
            },
            {
                'input_ids': torch.tensor([4, 5, 6]),
                'labels': torch.tensor([10, 11, 12])
            },
        ]
        out = proc.collate_fn(batch)
        assert out[0]['input_ids'].shape == (1, 6)
        assert out[0]['labels'].shape == (1, 6)
