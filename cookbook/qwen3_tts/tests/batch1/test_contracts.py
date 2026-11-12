"""Run: python tests/batch1/test_contracts.py [--isolated-twinkle].

Default requires installed Twinkle + Qwen3-TTS + transformers. Isolated mode
substitutes only Twinkle infrastructure, and skips official mel parity.
All tensors, autograd and ForCausalLMLoss use real torch/transformers.
"""
import ast
import importlib.util
import inspect
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from transformers.loss.loss_utils import ForCausalLMLoss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ISOLATED = '--isolated-twinkle' in sys.argv
if ISOLATED:
    sys.argv.remove('--isolated-twinkle')
    from isolated_support import install
    install(ROOT)

from tts.qwen3.data import Qwen3TTSDataConfig, build_qwen3_sample, collate_qwen3_batch
from tts.qwen3.input_processor import Qwen3TTSInputProcessor
from tts.qwen3.forward import qwen3_tts_forward_step
from tts.qwen3.loss import Qwen3TTSLoss
from tts.qwen3.model import Qwen3TTSModel

spec = importlib.util.spec_from_file_location('prepare', ROOT / 'scripts/audio_features.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def samples(n=2):
    cfg = Qwen3TTSDataConfig(
        tts_pad_token_id=0, tts_bos_token_id=1, tts_eos_token_id=2,
        codec_pad_id=3, codec_bos_id=4, codec_eos_token_id=5,
        codec_nothink_id=6, codec_think_bos_id=7, codec_think_eos_id=8)
    rows = [build_qwen3_sample(np.arange(10, 15-i),
            np.arange((3-i)*16).reshape(3-i, 16) % 20 + 9, cfg) for i in range(n)]
    mels = [np.full((10, 128), i+1, dtype=np.float32) for i in range(n)]
    return cfg, rows, mels


class Talker(nn.Module):
    """Small differentiable double of Qwen's two-stage API, with real HF loss."""
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(num_code_groups=16)
        self.model = nn.Module()
        self.model.text_embedding = nn.Embedding(64, 8)
        self.text_projection = nn.Identity()
        self.model.codec_embedding = nn.Embedding(64, 8)
        self.code_predictor = nn.Module()
        self.code_predictor.embeddings = nn.ModuleList([nn.Embedding(64, 8) for _ in range(15)])
        self.code_predictor.get_input_embeddings = lambda: self.code_predictor.embeddings
        self.head = nn.Linear(8, 64)
        self.heads = nn.ModuleList([nn.Linear(8, 64) for _ in range(15)])

    def forward(self, inputs_embeds, attention_mask, labels, **kwargs):
        self.labels = labels
        self.hidden = inputs_embeds
        self.logits = self.head(inputs_embeds)
        self.logits.retain_grad()
        return types.SimpleNamespace(
            loss=ForCausalLMLoss(self.logits, labels, 64),
            hidden_states=((inputs_embeds,), None))

    def forward_sub_talker_finetune(self, codes, hidden):
        self.selected_hidden = hidden
        self.selected_codes = codes
        logits = torch.stack([head(hidden) for head in self.heads], dim=1)
        logits.retain_grad()
        self.sub_logits = logits
        return logits, ForCausalLMLoss(logits, codes[:, 1:], 64)


class Model(nn.Module):
    dtype = torch.float32
    device = torch.device('cpu')
    def __init__(self):
        super().__init__()
        self.talker = Talker()
        self.speaker_encoder = lambda mels: mels.mean(1)[:, :8]


class Contracts(unittest.TestCase):
    def test_current_frame_residual_codes_do_not_leak_into_condition(self):
        cfg, rows, mels = samples(1)
        raw = collate_qwen3_batch(rows, cfg, mels)
        model = Model()
        processor = Qwen3TTSInputProcessor()
        batch = processor(raw, model=model)
        qwen3_tts_forward_step(model, batch)
        before = model.talker.selected_hidden.detach().clone()
        changed = {k: v.clone() for k,v in batch.items()}
        first = int(torch.where(changed['codec_mask'][0])[0][0])
        changed['codec_ids'][0, first, 1:] = (changed['codec_ids'][0, first, 1:] + 1) % 64
        qwen3_tts_forward_step(model, changed)
        torch.testing.assert_close(model.talker.selected_hidden[0], before[0], rtol=0, atol=0)
        self.assertFalse(torch.equal(model.talker.selected_hidden[1], before[1]))


    def test_initialize_accepts_local_mode(self):
        tree = ast.parse((ROOT / 'scripts/train_qwen3_tts_lora.py').read_text(encoding='utf-8'))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == 'initialize']
        self.assertEqual(len(calls), 1)
        seen = []
        def initialize(mode, global_device_mesh):
            self.assertIn(mode, ('local', 'ray'))  # pinned Twinkle's accepted contract
            self.assertEqual(mode, 'local')
            seen.append(global_device_mesh)
        eval(compile(ast.Expression(calls[0]), '<initialize>', 'eval'),
             {'twinkle': types.SimpleNamespace(initialize=initialize), 'device_mesh': 'single'})
        self.assertEqual(seen, ['single'])

    def test_codec_single_list_tensor_and_batch_tensor(self):
        expected = np.arange(48).reshape(3, 16)
        for value in ([torch.tensor(expected)], [expected], torch.tensor(expected),
                      torch.tensor(expected)[None], expected):
            tokenizer = types.SimpleNamespace(encode=lambda *a, **kw: types.SimpleNamespace(audio_codes=value))
            np.testing.assert_array_equal(prepare.encode_audio_codes(np.ones(4096), tokenizer), expected)

    def test_codec_rejects_empty_multiple_and_wrong_shapes(self):
        for value in ([], [np.zeros((2,16))]*2, np.zeros((2,3,16)), np.zeros((2,15)), np.zeros((0,16))):
            with self.subTest(shape=str(type(value))), self.assertRaises(ValueError):
                tokenizer = types.SimpleNamespace(encode=lambda *a, **kw: types.SimpleNamespace(audio_codes=value))
                prepare.encode_audio_codes(np.ones(4096), tokenizer)

    def test_numpy_mel_batch_identity_and_singleton(self):
        for n in (1,2):
            cfg, rows, mels = samples(n)
            for refs in (mels, [m[None] for m in mels]):
                batch = collate_qwen3_batch(rows, cfg, refs)
                self.assertEqual(batch['ref_mels'].shape, (n,10,128))
                for i in range(n):
                    np.testing.assert_array_equal(batch['ref_mels'][i], mels[i])

    def test_mel_length_and_count_fail_explicitly(self):
        cfg, rows, mels = samples()
        for refs in ([mels[0], mels[1][:7]], mels[:1], [], [mels[0], np.zeros((10,80))]):
            with self.assertRaises(ValueError):
                collate_qwen3_batch(rows, cfg, refs)
            feats = [dict(row, ref_mels=mel) for row, mel in zip(rows, refs)]
            if len(feats) == 2:
                with self.assertRaises(ValueError):
                    Qwen3TTSInputProcessor().collate_fn(feats)

    def test_processor_entire_call_samples_and_collated(self):
        for n in (1,2):
            cfg, rows, mels = samples(n)
            expected = collate_qwen3_batch(rows, cfg, mels)
            inputs = [[dict(row, ref_mels=mel) for row,mel in zip(rows,mels)], expected,
                      {k:torch.as_tensor(v) for k,v in expected.items()}]
            for value in inputs:
                actual = Qwen3TTSInputProcessor()(value, model=Model())
                self.assertEqual(set(actual), set(expected))
                for key in expected:
                    np.testing.assert_array_equal(actual[key].numpy(), expected[key])
                self.assertEqual(actual['codec_mask'].dtype, torch.bool)

    def test_processor_rejects_missing_mel_bad_rank_and_packing(self):
        cfg, rows, mels = samples()
        batch = collate_qwen3_batch(rows, cfg, mels)
        missing = dict(batch); del missing['ref_mels']
        for value in (missing, dict(batch, ref_mels=np.zeros((20,128)))):
            with self.assertRaises(ValueError):
                Qwen3TTSInputProcessor()(value, model=Model())
        with self.assertRaises(ValueError):
            Qwen3TTSInputProcessor(padding_free=True)(batch, model=Model())

    def test_wrapper_forward_alignment_and_every_head_gradient(self):
        for n in (1,2):
            cfg, rows, mels = samples(n)
            batch = collate_qwen3_batch(rows, cfg, mels)
            model = Model()
            status = types.SimpleNamespace()
            config = types.SimpleNamespace(processor=Qwen3TTSInputProcessor(), train_status=status)
            wrapper = types.SimpleNamespace(
                model=model, optimizer_group={'default':config},
                _get_default_group=lambda:'default', _lazy_wrap_model=lambda:None,
                _not_encoded=lambda x:False, sp_strategy=None, hf_config=None)
            outputs = inspect.unwrap(Qwen3TTSModel.forward)(wrapper, inputs=batch)
            t = model.talker
            labels = torch.as_tensor(batch['codec_0_labels'])
            self.assertTrue(torch.equal(t.labels, labels))
            expected = nn.functional.cross_entropy(t.logits[:,:-1].reshape(-1,64),
                                                   labels[:,1:].reshape(-1))
            torch.testing.assert_close(outputs['talker_loss'], expected)
            mask = torch.as_tensor(batch['codec_mask'])
            torch.testing.assert_close(t.selected_hidden, t.hidden[:, :-1][mask[:, 1:]])
            codes = torch.as_tensor(batch['codec_ids'])[mask]
            torch.testing.assert_close(outputs['sub_talker_labels'], codes[:,1:])
            loss = Qwen3TTSLoss()({}, outputs)['loss']
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for b,row in enumerate(rows):
                targets = np.where(row['codec_0_labels'] != -100)[0]
                self.assertGreater(t.logits.grad[b,targets[0]-1].abs().sum().item(), 0)
                self.assertGreater(t.logits.grad[b,targets[-1]-1].abs().sum().item(), 0)  # EOS
            shifted_valid = nn.functional.pad(labels[:,1:] != -100, (0,1))
            self.assertEqual(t.logits.grad[~shifted_valid].abs().sum().item(), 0)
            # Every head receives its own label, including cb1 and cb15.
            want = t.sub_logits.detach().softmax(-1)
            want.scatter_add_(-1, codes[:,1:,None], -torch.ones_like(codes[:,1:,None], dtype=want.dtype))
            want *= 0.3 / (codes.shape[0]*15)
            torch.testing.assert_close(t.sub_logits.grad, want)
            for head in t.heads:
                self.assertTrue(torch.isfinite(head.weight.grad).all())
                self.assertGreater(head.weight.grad.abs().sum().item(), 0)
            self.assertIs(status.outputs['talker_loss'], outputs['talker_loss'])
            self.assertEqual(status.loss_value, 0)

    def test_talker_time_shift_independent_of_processor(self):
        cfg, rows, mels = samples()
        batch = {key:torch.as_tensor(value) for key,value in collate_qwen3_batch(rows,cfg).items()}
        batch['ref_mels'] = torch.as_tensor(np.stack(mels))
        model = Model()
        outputs = qwen3_tts_forward_step(model,batch)
        # Independent oracle: position t predicts original label[t+1], including
        # the first frame and EOS. This fails the old external+internal shift.
        expected = nn.functional.cross_entropy(model.talker.logits[:,:-1].reshape(-1,64),
                                               batch['codec_0_labels'][:,1:].reshape(-1))
        torch.testing.assert_close(outputs['talker_loss'], expected)
        self.assertEqual(model.talker.logits.shape[:2],batch['codec_0_labels'].shape)

    def test_official_mel_call_contract(self):
        fake = types.ModuleType('qwen_tts.core.models.modeling_qwen3_tts')
        def mel(wave, **kw):
            self.assertEqual(wave.shape, (1,4096))
            self.assertEqual(kw, dict(n_fft=1024,num_mels=128,sampling_rate=24000,
                             hop_size=256,win_size=1024,fmin=0,fmax=12000.0))
            return torch.ones(1,128,16)
        fake.mel_spectrogram = mel
        with patch.dict(sys.modules, {'qwen_tts.core.models.modeling_qwen3_tts':fake}):
            self.assertEqual(prepare.extract_mel(np.ones(4096)).shape, (16,128))

    @unittest.skipIf(ISOLATED, 'Official qwen_tts not installed in isolated CPU run')
    def test_official_sub_talker_methods_and_last_head_gradient(self):
        from qwen_tts.core.models.modeling_qwen3_tts import (
            Qwen3TTSTalkerForConditionalGeneration,
            Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
        )
        # Run the installed official assembly and head-selection methods. Only
        # the expensive transformer decoder is replaced by a tiny causal sum.
        embeddings = nn.ModuleList([nn.Embedding(64,8) for _ in range(15)])
        heads = nn.ModuleList([nn.Linear(8,64) for _ in range(15)])
        cp = types.SimpleNamespace(
            config=types.SimpleNamespace(num_code_groups=16, vocab_size=64,
                output_attentions=False, output_hidden_states=False),
            small_to_mtp_projection=nn.Identity(), lm_head=heads,
            get_input_embeddings=lambda:embeddings, loss_function=ForCausalLMLoss,
            model=lambda inputs_embeds, **kw: types.SimpleNamespace(last_hidden_state=inputs_embeds.cumsum(1)))
        cp.forward_finetune = types.MethodType(
            Qwen3TTSTalkerCodePredictorModelForConditionalGeneration.forward_finetune, cp)
        talker = types.SimpleNamespace(
            config=types.SimpleNamespace(num_code_groups=16,hidden_size=8),
            code_predictor=cp, get_input_embeddings=lambda:embeddings[0])
        codes = torch.arange(32).reshape(2,16)
        hidden = torch.randn(2,8,requires_grad=True)
        logits, _ = Qwen3TTSTalkerForConditionalGeneration.forward_sub_talker_finetune(talker,codes,hidden)
        logits.retain_grad()
        outputs = dict(talker_loss=hidden.sum()*0,sub_talker_logits=logits,sub_talker_labels=codes[:,1:])
        Qwen3TTSLoss()({},outputs)['loss'].backward()
        expected = logits.detach().softmax(-1)
        expected.scatter_add_(-1,codes[:,1:,None],-torch.ones(2,15,1))
        torch.testing.assert_close(logits.grad, expected*(0.3/30))
        for head in heads:
            self.assertGreater(head.weight.grad.abs().sum().item(),0)

    @unittest.skipIf(ISOLATED, 'Official qwen_tts not installed in isolated CPU run')
    def test_official_mel_numerical_parity(self):
        from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
        wave = np.sin(np.arange(8192, dtype=np.float32)*0.03)
        expected = mel_spectrogram(torch.from_numpy(wave)[None], n_fft=1024,
            num_mels=128,sampling_rate=24000,hop_size=256,win_size=1024,fmin=0,fmax=12000).transpose(1,2)[0]
        np.testing.assert_allclose(prepare.extract_mel(wave), expected.numpy(), rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    print('CPU contracts / Twinkle infrastructure: ' + ('ISOLATED DOUBLES' if ISOLATED else 'INSTALLED LIBRARY'))
    unittest.main(verbosity=2)
