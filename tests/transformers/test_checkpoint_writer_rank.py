"""All ranks gather checkpoint state, but only the master writes model shards."""
import inspect
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from peft import LoraConfig
from safetensors.torch import load_file, save_file

from twinkle.model.multi_lora import MultiLora, MultiLoraTaskHeadSlots
from twinkle.model.transformers.multi_lora_transformers import MultiLoraTransformersModel
from twinkle.model.transformers.transformers import TransformersModel


@pytest.mark.parametrize('async_save', [False, True])
def test_multi_lora_checkpoint_writes_the_captured_head_once(tmp_path, monkeypatch, async_save):
    wrapper = object.__new__(MultiLoraTransformersModel)
    manager = MultiLora(max_loras=1, max_r=2)
    config = LoraConfig(r=2, target_modules=['0'])
    model = manager.patch(torch.nn.Sequential(torch.nn.Linear(2, 2)), lora_config=config)
    manager.acquire_lora('tenant', config)
    host = model.get_base_model()
    slots = MultiLoraTaskHeadSlots(1)
    slots.build(host, lambda: torch.nn.Linear(2, 1))
    slots.bind('tenant', host)
    wrapper.__dict__['model'] = model
    wrapper.multi_adapter = manager
    wrapper.task_head_slots = slots
    wrapper.optimizer_group = {'tenant': SimpleNamespace(cur_step=1, template=None)}
    wrapper._default_tokenizer = Mock()
    wrapper.strategy = SimpleNamespace(unwrap_model=lambda model: model)
    wrapper._task_head_host = lambda: host
    wrapper._lazy_wrap_model = lambda: None
    wrapper._adapter_context = lambda _: nullcontext()
    wrapper._get_adapter_state_dict_for_save = lambda _: {'lora.weight': torch.ones(1)}
    pending = []
    wrapper.submit_async_write = pending.append
    monkeypatch.setattr('twinkle.Platform.is_master', lambda: True)
    writes = Mock(wraps=save_file)
    monkeypatch.setattr('safetensors.torch.save_file', writes)
    before = host.task_heads['tenant'].weight.detach().clone()

    checkpoint_dir = inspect.unwrap(MultiLoraTransformersModel.save)(
        wrapper, 'checkpoint', output_dir=str(tmp_path), adapter_name='tenant', async_save=async_save)
    if async_save:
        assert not writes.called
        slots.release('tenant', host, lambda tensor, value: tensor.copy_(value))
        wrapper.optimizer_group.pop('tenant')
        with torch.no_grad():
            host.task_head_slot_0.weight.fill_(42)
        pending.pop()()

    writes.assert_called_once()
    torch.testing.assert_close(load_file(f'{checkpoint_dir}/task_head.safetensors')['task_head.weight'], before)
    assert (tmp_path / 'checkpoint' / 'adapter_config.json').is_file()


class CheckpointWriterRankTest(unittest.TestCase):
    def test_full_checkpoint_has_one_writer_after_collectives(self):
        module = 'twinkle.model.transformers.transformers'
        for async_save in (False, True):
            for rank in range(2):
                with self.subTest(async_save=async_save, rank=rank):
                    wrapper = object.__new__(TransformersModel)
                    model = Mock()
                    wrapper.__dict__['model'] = model
                    state = {'weight': object()}
                    wrapper.strategy = Mock()
                    wrapper.strategy.unwrap_model.return_value = model
                    wrapper.strategy.get_full_state_dict.return_value = state
                    wrapper.optimizer_group = {'': SimpleNamespace(
                        cur_step=3, gradient_accumulation_steps=1, optimizer=None, template=None,
                        require_finalized=Mock())}
                    wrapper._save_training_state = Mock()
                    wrapper._default_tokenizer = Mock()
                    wrapper._task_head_state = Mock(return_value={})
                    wrapper.submit_async_write = Mock()
                    with patch(f'{module}.Platform.is_master', return_value=rank == 0), \
                            patch(f'{module}.copy_checkpoint_args'), patch(f'{module}.rotate_checkpoints'):
                        result = inspect.unwrap(TransformersModel.save)(
                            wrapper, name='checkpoint-final', output_dir='/unused',
                            save_optimizer=True, async_save=async_save)
                        self.assertEqual(result, '/unused/checkpoint-final')
                        wrapper.strategy.get_full_state_dict.assert_called_once_with(model)
                        wrapper._save_training_state.assert_called_once()
                        self.assertEqual(wrapper._save_training_state.call_args.kwargs['defer_trainer_state'],
                                         async_save)
                        wrapper._task_head_state.assert_called_once_with('')
                        if async_save and rank == 0:
                            model.save_pretrained.assert_not_called()
                            wrapper.submit_async_write.assert_called_once()
                            wrapper.submit_async_write.call_args.args[0]()
                        else:
                            wrapper.submit_async_write.assert_not_called()
                        if rank == 0:
                            model.save_pretrained.assert_called_once_with(
                                '/unused/checkpoint-final', state_dict=state, is_main_process=True,
                                safe_serialization=True, max_shard_size='5GB')
                        else:
                            model.save_pretrained.assert_not_called()

    def test_partial_accumulation_is_rejected_before_weight_collection(self):
        wrapper = object.__new__(TransformersModel)
        wrapper.strategy = Mock()
        wrapper.optimizer_group = {'': SimpleNamespace(
            cur_step=3, gradient_accumulation_steps=2, optimizer=None, require_finalized=Mock())}

        with self.assertRaisesRegex(ValueError, 'complete gradient accumulation window'):
            inspect.unwrap(TransformersModel.save)(
                wrapper, name='partial', output_dir='/unused', save_optimizer=True)

        wrapper.strategy.get_full_state_dict.assert_not_called()
        wrapper.strategy.unwrap_model.assert_not_called()


if __name__ == '__main__':
    unittest.main()
