"""Real pretrained Qwen3-TTS + Twinkle + one NPU. No doubles or skip path.

Uses the existing local 1.7B Base directory without downloads. The synthetic sine
wave/text pair only exercises mechanics; it is not speech/quality evidence.
"""
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch_npu  # noqa: F401

# Also enforce local-only model loading when this test is invoked directly.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from peft import LoraConfig
from safetensors.torch import load_file
import twinkle
from twinkle import DeviceMesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))
from tts.qwen3.data import build_assistant_text, build_qwen3_sample, collate_qwen3_batch
from tts.qwen3.forward import qwen3_tts_forward_step
from tts.qwen3.input_processor import Qwen3TTSInputProcessor
from tts.qwen3.loss import Qwen3TTSLoss
from tts.qwen3.tokenization import load_text_tokenizer, assert_tokenizers_match
from tts.qwen3.runtime import DEFAULT_MODEL_PATH, ADAPTER, data_config, load_twinkle_model
from model_assertions import trainable_snapshot, frozen_digest, assert_gradients, assert_updated

spec = importlib.util.spec_from_file_location('prepare', ROOT / 'scripts/audio_features.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def report(stage, **values):
    print(json.dumps({'stage': stage, **values}, ensure_ascii=False), flush=True)


def evaluation(model, batch):
    model.eval()
    with torch.no_grad():
        result = qwen3_tts_forward_step(model, batch)
    return {key: result[key].detach().float().cpu() for key in
            ('talker_loss', 'sub_talker_logits', 'target_speaker_embedding')}


class RealModelAcceptance(unittest.TestCase):
    scope = 'one synthetic sample; not speech quality or optimizer resume'

    def sample(self):
        t = np.arange(24000, dtype=np.float32) / 24000
        wave = (0.08 * np.sin(2 * np.pi * 220 * t) * np.sin(np.pi * t)**2).astype(np.float32)
        return wave, '这是一条训练链路测试。', {'kind': 'synthetic sine wave'}

    def test_pretrained_single_sample_lora_update_and_fresh_reload(self):
        torch.manual_seed(20261023)
        np.random.seed(20261023)
        torch.npu.set_device(0)
        mesh = DeviceMesh.from_sizes(fsdp_size=1, dp_size=1)
        twinkle.initialize(mode='local', global_device_mesh=mesh, seed=20261023)
        path = Path(os.environ.get('MODEL_PATH', DEFAULT_MODEL_PATH)).expanduser().resolve(strict=True)
        report('加载本地模型 / Load local model', path=str(path),
               source_revision='unknown: user-provided local model',
               config_sha256=hashlib.sha256((path / 'config.json').read_bytes()).hexdigest(),
               visible_devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES', 'unset'))
        model = load_twinkle_model(path, mesh)
        cfg = data_config(model.hf_config)
        self.assertEqual(model.hf_config.tts_model_type, 'base')
        self.assertEqual(cfg.num_code_groups, 16)
        self.assertEqual(model.model.dtype, torch.bfloat16)

        # Actual codec on CPU fp32; training and speaker encoder on NPU bf16.
        wave, text, sample_metadata = self.sample()
        report('样本来源 / Sample provenance', **sample_metadata)
        codec = model.model.speech_tokenizer
        codec.model.float().eval()
        with torch.no_grad():
            codes = prepare.encode_audio_codes(wave, codec)
        mel = prepare.extract_mel(wave)
        model.model.speech_tokenizer = None
        del codec
        tokenizer = load_text_tokenizer(path)
        probes = [text, '你好，世界！Hello 2026.\nA  B', build_assistant_text(text)]
        assert_tokenizers_match(tokenizer,
            [model.optimizer_group[model._get_default_group()].template.tokenizer], probes)
        text_ids = tokenizer.encode(build_assistant_text(text))
        # Verify the fixed suffix on the real tokenizer rather than blindly trim.
        suffix = tokenizer.encode('<|im_end|>\n<|im_start|>assistant\n', add_special_tokens=False)
        self.assertEqual(len(suffix), 5)
        self.assertEqual(text_ids[-len(suffix):], suffix)
        row = build_qwen3_sample(np.asarray(text_ids[:-len(suffix)]), codes, cfg)
        raw_batch = collate_qwen3_batch([row], cfg, [mel])
        self.assertGreater(len(codes), 0)
        self.assertGreaterEqual(int(codes.min()), 0)
        self.assertLess(int(codes.max()), model.hf_config.talker_config.code_predictor_config.vocab_size)
        report('样本已编码 / Sample encoded', audio_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),
               codec_shape=list(codes.shape), mel_shape=list(mel.shape), tokens=len(text_ids))

        model.add_adapter_to_model(ADAPTER, LoraConfig(
            r=8, lora_alpha=16, target_modules=['q_proj', 'v_proj'],
            lora_dropout=0.0, bias='none'), gradient_accumulation_steps=1)
        assert_tokenizers_match(tokenizer, [model.optimizer_group[ADAPTER].template.tokenizer], probes)
        model.set_processor(Qwen3TTSInputProcessor, adapter_name=ADAPTER)
        model.set_loss(Qwen3TTSLoss(), adapter_name=ADAPTER)
        model.set_optimizer('AdamW', lr=1e-3, weight_decay=0.0, adapter_name=ADAPTER)
        model._lazy_wrap_model()
        base = model.strategy.unwrap_model(model.model)
        self.assertEqual(base.device.type, 'npu')
        self.assertEqual(base.dtype, torch.bfloat16)
        before = trainable_snapshot(base)
        frozen_before = frozen_digest(base)
        batch = model.optimizer_group[ADAPTER].processor(raw_batch, model=base)
        report('执行一次更新 / Run one update', trainable_parameters=sum(v.numel() for v in before.values()),
               trainable_tensors=len(before), adapter_dtype=str(next(iter(before.values())).dtype))
        outputs = model.forward_backward(inputs=batch, adapter_name=ADAPTER)
        loss = torch.as_tensor(outputs['loss'])
        self.assertTrue(torch.isfinite(loss).all().item())
        nonzero = assert_gradients(base)
        model.clip_grad_and_step(adapter_name=ADAPTER)
        torch.npu.synchronize()
        after = trainable_snapshot(base)
        changed = assert_updated(before, after)
        self.assertEqual(frozen_before, frozen_digest(base), 'Frozen base weights changed')
        self.assertEqual(model.optimizer_group[ADAPTER].cur_step, 1)
        reference = evaluation(base, batch)
        for value in reference.values():
            self.assertTrue(torch.isfinite(value).all().item())
        report('更新断言通过 / Update assertions passed', loss=loss.detach().float().cpu().tolist(),
               nonzero_gradient_tensors=len(nonzero), changed_tensors=len(changed), frozen_sha256=frozen_before)

        output_root = ROOT / 'outputs'
        output_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='real-lora-', dir=output_root) as output:
            checkpoint = model.save('adapter', output_dir=output, adapter_name=ADAPTER, save_optimizer=False)
            assert_tokenizers_match(tokenizer, [load_text_tokenizer(checkpoint)], probes)
            saved = load_file(str(Path(checkpoint) / 'adapter_model.safetensors'))
            expected = {name.replace('.' + ADAPTER + '.', '.'): value for name, value in after.items()}
            self.assertEqual(saved.keys(), expected.keys(), 'Checkpoint must contain only this adapter')
            for name in saved:
                torch.testing.assert_close(saved[name], expected[name], rtol=0, atol=0)
            # Remove the original model and optimizer before loading another base.
            del before, after, outputs, loss, base, model, batch
            gc.collect()
            torch.npu.empty_cache()
            report('从新底模重载 / Reload into fresh pretrained base')
            reloaded = load_twinkle_model(path, mesh)
            reloaded.model.speech_tokenizer = None
            reloaded.add_adapter_to_model(ADAPTER, str(checkpoint), is_trainable=True)
            reloaded.set_processor(Qwen3TTSInputProcessor, adapter_name=ADAPTER)
            assert_tokenizers_match(tokenizer,
                [reloaded.optimizer_group[ADAPTER].template.tokenizer], probes)
            report('文本编码一致 / Tokenizer IDs consistent', probes=len(probes),
                   fix_mistral_regex=True, paths=['text loader', 'Twinkle templates', 'saved tokenizer', 'fresh reload'])
            reloaded._lazy_wrap_model()
            fresh = reloaded.strategy.unwrap_model(reloaded.model)
            actual_weights = trainable_snapshot(fresh)
            normalized = {n.replace('.' + ADAPTER + '.', '.'): v for n, v in actual_weights.items()}
            self.assertEqual(normalized.keys(), saved.keys())
            for name in saved:
                torch.testing.assert_close(normalized[name], saved[name], rtol=0, atol=0)
            self.assertEqual(frozen_before, frozen_digest(fresh))
            reloaded_batch = reloaded.optimizer_group[ADAPTER].processor(raw_batch, model=fresh)
            actual = evaluation(fresh, reloaded_batch)
            differences = {}
            for key in reference:
                torch.testing.assert_close(actual[key], reference[key], rtol=1e-3, atol=1e-3)
                differences[key] = (actual[key] - reference[key]).abs().max().item()
            report('保存重载通过 / Save and fresh reload passed', max_absolute_differences=differences,
                   scope=self.scope)


if __name__ == '__main__':
    unittest.main(verbosity=2)
