"""All ranks gather checkpoint state, but only the master writes model shards."""
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from peft import LoraConfig
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from twinkle import Platform, initialize
from twinkle.model.transformers.multi_lora_transformers import MultiLoraTransformersModel
from twinkle.model.transformers.transformers import TransformersModel


@pytest.mark.parametrize('async_save', [False, True])
def test_multi_lora_checkpoint_writes_one_snapshot_even_after_slot_reuse(tmp_path, monkeypatch, async_save):
    monkeypatch.setenv('ACCELERATE_USE_CPU', 'true')
    monkeypatch.setattr(Platform, 'get_local_device', lambda *args, **kwargs: 'cpu')
    initialize(mode='local')
    base = tmp_path / 'base'
    config = Qwen2Config(vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2, pad_token_id=0)
    Qwen2ForCausalLM(config).save_pretrained(base)
    tokenizer = Tokenizer(WordPiece({'[PAD]': 0, '[UNK]': 1}, unk_token='[UNK]'))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='[PAD]', unk_token='[UNK]').save_pretrained(base)
    targets = ['q_proj']
    model = MultiLoraTransformersModel(
        model_id=str(base), mixed_precision='no', gradient_checkpointing=False, max_loras=1, max_r=2,
        target_modules=targets, task='seq_cls', num_labels=3)
    adapter = LoraConfig(r=2, target_modules=targets)
    model.add_adapter_to_model('tenant', adapter)
    pending = []
    monkeypatch.setattr(model, 'submit_async_write', pending.append)
    writes = Mock(wraps=save_file)
    monkeypatch.setattr('safetensors.torch.save_file', writes)
    with torch.no_grad():
        for name, param in model.model.named_parameters():
            if '.lora_' in name:
                param.fill_(3.)
    before_head = model._task_head_state('tenant')

    checkpoint_dir = Path(model.save('checkpoint', output_dir=str(tmp_path), adapter_name='tenant',
                                     async_save=async_save))
    if async_save:
        assert not checkpoint_dir.exists()
        model.remove_adapter('tenant')
        model.add_adapter_to_model('next', adapter)
        with torch.no_grad():
            for name, param in model.model.named_parameters():
                if '.lora_' in name or 'task_head_slot_' in name:
                    param.fill_(7.)
        pending.pop()()

    writes.assert_called_once()
    torch.testing.assert_close(load_file(checkpoint_dir / 'task_head.safetensors'), before_head)
    weights = load_file(checkpoint_dir / 'adapter_model.safetensors')
    assert weights and all(torch.all(tensor == 3.) for tensor in weights.values())
    assert (checkpoint_dir / 'adapter_config.json').is_file()


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
