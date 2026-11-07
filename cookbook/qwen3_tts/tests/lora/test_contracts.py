"""CPU regressions for projection and acceptance assertions; Twinkle may be doubled."""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / 'tests/batch1'))
import test_contracts as previous
import torch
from torch import nn
from tts.qwen3.runtime import data_config
from model_assertions import frozen_digest, trainable_snapshot, assert_updated, assert_gradients


class AcceptanceContracts(unittest.TestCase):
    def test_new_optimizer_groups_receive_local_tokenizer_policy(self):
        module = types.ModuleType('twinkle.model.transformers.transformers')
        module.OptimizerGroup = types.SimpleNamespace
        wrapper = types.SimpleNamespace(tokenizer_id='/local/model', device_mesh=None)
        with patch.dict(sys.modules, {module.__name__: module}), \
             patch('tts.qwen3.tokenization.local_template') as template:
            for _ in range(2):  # initial group and newly added adapter group
                group = previous.Qwen3TTSModel._construct_default_optimizer_group(wrapper)
                self.assertIs(group.template, template.return_value)
                self.assertIsInstance(group.processor, previous.Qwen3TTSInputProcessor)
                self.assertIsInstance(group.loss_instance, previous.Qwen3TTSLoss)
            self.assertEqual(template.call_count, 2)
            template.assert_called_with('/local/model')

    def cpu_batch(self, model, cfg, rows, mels):
        # These are CPU regressions even on an NPU host. Fail if the processor
        # consults the platform default instead of the supplied model device.
        with patch('tts.qwen3.input_processor.Platform.get_local_device',
                   side_effect=AssertionError('CPU fixture must use model.device')):
            batch = previous.Qwen3TTSInputProcessor()(
                previous.collate_qwen3_batch(rows, cfg, mels), model=model)
        self.assertTrue(all(value.device.type == 'cpu' for value in batch.values()))
        return batch

    def test_text_projection_handles_different_width_and_receives_gradient(self):
        cfg, rows, mels = previous.samples(1)
        model = previous.Model()
        batch = self.cpu_batch(model, cfg, rows, mels)
        model.talker.model.text_embedding = nn.Embedding(64, 12)
        model.talker.text_projection = nn.Linear(12, 8)
        seen = []
        handle = model.talker.text_projection.register_forward_hook(lambda m, a, o: seen.append(o.shape))
        outputs = previous.qwen3_tts_forward_step(model, batch)
        previous.Qwen3TTSLoss()({}, outputs)['loss'].backward()
        handle.remove()
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][-1], 8)
        self.assertGreater(model.talker.text_projection.weight.grad.abs().sum().item(), 0)

    def test_speaker_encoder_has_no_gradient(self):
        cfg, rows, mels = previous.samples(1)
        model = previous.Model()
        batch = self.cpu_batch(model, cfg, rows, mels)
        # Preserve the speaker encoder's [B,F,128] -> [B,H] contract.
        class Speaker(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(128, 8)
            def forward(self, mel):
                return self.linear(mel.mean(1))
        model.speaker_encoder = Speaker()
        outputs = previous.qwen3_tts_forward_step(model, batch)
        previous.Qwen3TTSLoss()({}, outputs)['loss'].backward()
        self.assertIsNone(model.speaker_encoder.linear.weight.grad)
        self.assertFalse(outputs['target_speaker_embedding'].requires_grad)

    def test_config_uses_all_real_token_ids_not_defaults(self):
        names = ['codec_pad_id', 'codec_bos_id', 'codec_eos_token_id',
                 'codec_nothink_id', 'codec_think_bos_id', 'codec_think_eos_id', 'num_code_groups']
        tc = types.SimpleNamespace(**dict(zip(names, range(10, 17))))
        config = types.SimpleNamespace(talker_config=tc, tts_pad_token_id=1, tts_bos_token_id=2, tts_eos_token_id=3)
        actual = data_config(config)
        self.assertEqual((actual.tts_pad_token_id, actual.tts_bos_token_id, actual.tts_eos_token_id), (1, 2, 3))
        for name in names:
            self.assertEqual(getattr(actual, name), getattr(tc, name))

    def tiny_model(self):
        model = nn.Module()
        model.base = nn.Module()
        model.base.talker = nn.Module()
        for branch in ('model', 'code_predictor'):
            layer = nn.Module()
            layer.lora_B = nn.ParameterDict({'smoke': nn.Parameter(torch.zeros(2, 2))})
            layer.weight = nn.Parameter(torch.ones(2, 2), requires_grad=False)
            setattr(model.base.talker, branch, layer)
        return model

    def test_assertions_reject_no_update_and_frozen_or_missing_branch(self):
        model = self.tiny_model()
        before = trainable_snapshot(model)
        with self.assertRaisesRegex(AssertionError, 'No parameter update'):
            assert_updated(before, trainable_snapshot(model))
        model.base.talker.model.weight.requires_grad_(True)
        with self.assertRaisesRegex(AssertionError, 'Only LoRA'):
            trainable_snapshot(model)
        model.base.talker.model.weight.requires_grad_(False)
        model.base.talker.code_predictor.lora_B['smoke'].requires_grad_(False)
        with self.assertRaisesRegex(AssertionError, 'Missing trainable branch'):
            trainable_snapshot(model)

    def test_gradients_updates_and_frozen_digest_detect_corruption(self):
        model = self.tiny_model()
        before = trainable_snapshot(model)
        digest = frozen_digest(model)
        with self.assertRaisesRegex(AssertionError, 'Missing/nonfinite'):
            assert_gradients(model)
        sum(p.sum() for p in model.parameters() if p.requires_grad).backward()
        self.assertEqual(len(assert_gradients(model)), 2)
        torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1).step()
        self.assertEqual(len(assert_updated(before, trainable_snapshot(model))), 2)
        self.assertEqual(digest, frozen_digest(model))
        with torch.no_grad():
            model.base.talker.model.weight[0, 0] += 1
        self.assertNotEqual(digest, frozen_digest(model))
        model.base.talker.model.lora_B['smoke'].grad.fill_(float('nan'))
        with self.assertRaisesRegex(AssertionError, 'Missing/nonfinite'):
            assert_gradients(model)


if __name__ == '__main__':
    unittest.main(verbosity=2)
