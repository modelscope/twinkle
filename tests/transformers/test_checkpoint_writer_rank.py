"""All ranks gather checkpoint state, but only the master writes model shards."""
import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from twinkle.model.transformers.transformers import TransformersModel


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
                        cur_step=3, gradient_accumulation_steps=1, optimizer=None, require_finalized=Mock())}
                    wrapper._save_training_state = Mock()
                    wrapper._save_tokenizer = Mock()
                    wrapper._save_task_head_checkpoint = Mock()
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
                        if async_save:
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
