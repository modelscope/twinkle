"""Five paired-speech micro-batches: continuous versus fresh-model resume."""
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / 'tests/lora'))
import test_real_model as baseline  # local-only imports and shared fixture utilities
import numpy as np
import torch
from peft import LoraConfig
from tts.qwen3.runtime import ADAPTER, DEFAULT_MODEL_PATH, load_twinkle_model, data_config
from model_assertions import trainable_snapshot, frozen_digest, assert_updated
from tts.qwen3.data import build_assistant_text, build_qwen3_sample, collate_qwen3_batch
from tts.qwen3.tokenization import load_text_tokenizer
from speech_fixture import load_speech_fixture
from tts.qwen3.input_processor import Qwen3TTSInputProcessor
from tts.qwen3.loss import Qwen3TTSLoss
from tts.qwen3.training import Progress, windows, train_window, save_boundary, resume_boundary


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def assert_tree_equal(test, a, b, rtol=1e-3, atol=1e-5):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for key in a:
            assert_tree_equal(test, a[key], b[key], rtol, atol)
    elif isinstance(a, (tuple, list)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            assert_tree_equal(test, x, y, rtol, atol)
    else:
        test.assertEqual(a, b)


def configure(model):
    model.add_adapter_to_model(ADAPTER, LoraConfig(r=8, lora_alpha=16,
        target_modules=['q_proj', 'v_proj'], lora_dropout=0.1, bias='none'),
        gradient_accumulation_steps=1)
    model.set_processor(Qwen3TTSInputProcessor, adapter_name=ADAPTER)
    model.set_loss(Qwen3TTSLoss(), adapter_name=ADAPTER)
    model.set_optimizer('AdamW', lr=1e-3, weight_decay=0.0, adapter_name=ADAPTER)
    model.set_lr_scheduler('StepLR', step_size=1, gamma=0.8, adapter_name=ADAPTER)
    model._lazy_wrap_model()


def rng_probe():
    return [random.random(), float(np.random.rand()), torch.rand(1).item(),
            torch.rand(1, device='npu:0').cpu().item()]


class ResumeAcceptance(unittest.TestCase):
    def test_five_micro_batches_tail_and_fresh_optimizer_resume(self):
        torch.npu.set_device(0)
        mesh = baseline.DeviceMesh.from_sizes(fsdp_size=1, dp_size=1)
        baseline.twinkle.initialize(mode='local', global_device_mesh=mesh, seed=20261023)
        path = Path(os.environ.get('MODEL_PATH', DEFAULT_MODEL_PATH)).resolve(strict=True)
        model = load_twinkle_model(path, mesh)
        wave, text, meta = load_speech_fixture(ROOT / 'tests/fixtures/librispeech')
        codec = model.model.speech_tokenizer
        codec.model.float().eval()
        with torch.no_grad():
            codes = baseline.prepare.encode_audio_codes(wave, codec)
        mel = baseline.prepare.extract_mel(wave)
        model.model.speech_tokenizer = None
        del codec
        tokenizer = load_text_tokenizer(path)
        ids = tokenizer.encode(build_assistant_text(text))
        suffix = tokenizer.encode('<|im_end|>\n<|im_start|>assistant\n', add_special_tokens=False)
        self.assertEqual(ids[-len(suffix):], suffix)
        cfg = data_config(model.hf_config)
        raw = collate_qwen3_batch([build_qwen3_sample(np.asarray(ids[:-len(suffix)]), codes, cfg)], cfg, [mel])
        fingerprint = hashlib.sha256(json.dumps({'sample': meta['sha256'], 'text_ids': ids,
            'model_config': hashlib.sha256((path / 'config.json').read_bytes()).hexdigest(),
            'schedule': 'same public sample repeated five times; accumulation=2'}, sort_keys=True).encode()).hexdigest()
        configure(model)
        base = model.strategy.unwrap_model(model.model)
        self.assertEqual(base.device.type, 'npu')
        self.assertEqual(base.dtype, torch.bfloat16)
        initial = trainable_snapshot(base)
        frozen = frozen_digest(base)
        group = model.optimizer_group[ADAPTER]
        batch = group.processor(raw, model=base)
        progress = Progress(accumulation_steps=2)
        train_window(model, [batch, batch], progress, ADAPTER)
        self.assertEqual((progress.micro_steps, progress.optimizer_steps, progress.consumed_samples), (2, 1, 2))
        at_boundary = trainable_snapshot(base)
        assert_updated(initial, at_boundary)
        optimizer_boundary = cpu_tree(group.optimizer.state_dict())
        scheduler_boundary = copy.deepcopy(group.lr_scheduler.state_dict())
        output_root = ROOT / 'outputs'
        output_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='resume-', dir=output_root) as temp:
            checkpoint = save_boundary(model, progress, temp, 'after-update-1', ADAPTER, fingerprint)
            baseline.report('断点已保存 / Boundary saved', micro_steps=2, optimizer_steps=1,
                            accumulation_steps=2, dropout=0.1, visible_devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'))
            trace = []
            losses = []
            sizes = []
            for window in windows([batch] * 3, 2):
                sizes.append(len(window))
                losses.extend(train_window(model, window, progress, ADAPTER,
                                           before_micro=lambda step: trace.append((step, rng_probe()))))
            expected_weights = trainable_snapshot(base)
            expected_optimizer = cpu_tree(group.optimizer.state_dict())
            expected_scheduler = copy.deepcopy(group.lr_scheduler.state_dict())
            expected_progress = copy.deepcopy(progress)
            self.assertEqual(sizes, [2, 1])
            self.assertEqual((progress.micro_steps, progress.optimizer_steps, progress.consumed_samples), (5, 3, 5))
            self.assertEqual(group.cur_step, 5)
            self.assertEqual(group.lr_scheduler.last_epoch, 3)
            self.assertEqual(group.train_status.num_tokens, 0)
            self.assertTrue(all(p.grad is None for p in base.parameters()))
            self.assertEqual(frozen, frozen_digest(base))
            assert_updated(at_boundary, expected_weights)
            self.assertEqual(len(expected_optimizer['state']), len(initial))
            self.assertTrue(all(int(state['step'].item()) == 3 for state in expected_optimizer['state'].values()))
            baseline.report('连续训练完成 / Continuous run completed', micro_steps=5, optimizer_steps=3,
                            window_sizes=[2, 2, 1], learning_rates=group.lr_scheduler.get_last_lr())
            del model, base, group, batch
            gc.collect()
            torch.npu.empty_cache()

            resumed = load_twinkle_model(path, mesh)
            resumed.model.speech_tokenizer = None
            configure(resumed)
            fresh = resumed.strategy.unwrap_model(resumed.model)
            new_group = resumed.optimizer_group[ADAPTER]
            new_batch = new_group.processor(raw, model=fresh)
            restored = resume_boundary(resumed, checkpoint, ADAPTER, fingerprint, 2)
            restored_weights = trainable_snapshot(fresh)
            for name in at_boundary:
                torch.testing.assert_close(restored_weights[name], at_boundary[name], rtol=0, atol=0)
            assert_tree_equal(self, optimizer_boundary, cpu_tree(new_group.optimizer.state_dict()), rtol=0, atol=0)
            self.assertEqual(scheduler_boundary, new_group.lr_scheduler.state_dict())
            actual_trace = []
            actual_losses = []
            for window in windows([new_batch] * (5 - restored.consumed_samples), 2):
                actual_losses.extend(train_window(resumed, window, restored, ADAPTER,
                    before_micro=lambda step: actual_trace.append((step, rng_probe()))))
            actual_weights = trainable_snapshot(fresh)
            assert_tree_equal(self, expected_weights, actual_weights)
            assert_tree_equal(self, expected_optimizer, cpu_tree(new_group.optimizer.state_dict()))
            self.assertEqual(expected_scheduler, new_group.lr_scheduler.state_dict())
            self.assertEqual(expected_progress, restored)
            self.assertEqual(trace, actual_trace, 'Python/NumPy/CPU/NPU random streams differ')
            np.testing.assert_allclose(losses, actual_losses, rtol=1e-3, atol=1e-5)
            self.assertEqual(frozen, frozen_digest(fresh))
            max_difference = max((actual_weights[n].float() - expected_weights[n].float()).abs().max().item()
                                 for n in expected_weights)
            baseline.report('断点续训对照通过 / Resume comparison passed', micro_steps=restored.micro_steps,
                optimizer_steps=restored.optimizer_steps, consumed_samples=restored.consumed_samples,
                max_weight_difference=max_difference, random_streams_equal=True,
                losses_continuous=losses, losses_resumed=actual_losses, rtol=1e-3, atol=1e-5,
                scope='five repeated public-speech micro-batches; boundary-only resume, single NPU')


if __name__ == '__main__':
    unittest.main(verbosity=2)
