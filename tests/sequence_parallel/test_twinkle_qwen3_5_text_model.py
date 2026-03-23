# Copyright (c) ModelScope Contributors. All rights reserved.
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from twinkle.model.transformers.models.qwen3_5 import modeling_qwen3_5 as tw_qwen35
from twinkle.model.transformers.strategy.sequence_parallel import SequenceParallel, SequenceParallelContext


def _build_text_config(layer_types=None) -> Qwen3_5TextConfig:
    layer_types = layer_types or ['full_attention']
    return Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        hidden_act='silu',
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        linear_conv_kernel_dim=3,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=layer_types,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


class _ContextReceiver:

    def __init__(self):
        self.context = None

    def set_sequence_parallel_context(self, context):
        self.context = context


class TestTwinkleQwen35TextModel(unittest.TestCase):

    def test_rejects_non_text_config(self):
        with self.assertRaises(TypeError):
            tw_qwen35.TwinkleQwen3_5ForCausalLM(Qwen3_5Config())

    def test_text_model_accepts_sequence_parallel_context(self):
        model = tw_qwen35.TwinkleQwen3_5TextModel(_build_text_config(['full_attention']))
        context = SequenceParallelContext(
            sp_group=None,
            sp_world_size=2,
            rank=0,
            world_size=2,
            real_position_ids=torch.tensor([[0, 1, 2]], dtype=torch.long),
            is_packed=False,
        )
        model.set_sequence_parallel_context(context)
        self.assertIs(model._sequence_parallel_context, context)

    def test_from_pretrained_loads_text_only_weights(self):
        config = _build_text_config(['full_attention'])
        hf_model = Qwen3_5ForCausalLM(config).eval()
        with tempfile.TemporaryDirectory() as temp_dir:
            hf_model.save_pretrained(temp_dir)
            tw_model = tw_qwen35.TwinkleQwen3_5ForCausalLM.from_pretrained(temp_dir).eval()

            input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                hf_outputs = hf_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=True,
                )
                tw_outputs = tw_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=True,
                )

            torch.testing.assert_close(tw_outputs.logits, hf_outputs.logits, rtol=0, atol=0)

    def test_sequence_parallel_prepare_inputs_injects_cu_seq_lens(self):
        sp = SequenceParallel()
        sp.world_size = 2
        sp.sp_world_size = 2
        sp.requires_cu_seq_lens_q = True
        receiver = _ContextReceiver()
        sp._bound_llm_model = receiver
        inputs = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long),
            'position_ids': torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long),
        }

        outputs = sp.prepare_inputs(inputs)

        self.assertIn('cu_seq_lens_q', outputs)
        self.assertTrue(torch.equal(outputs['cu_seq_lens_q'], torch.tensor([0, 5, 6], dtype=torch.int32)))
        self.assertIsNotNone(receiver.context)
        self.assertFalse(receiver.context.is_packed)
        self.assertTrue(torch.equal(receiver.context.real_position_ids, inputs['position_ids']))

    def test_linear_attention_requires_fast_path_dependencies(self):
        with self.assertRaises(ImportError):
            tw_qwen35.TwinkleQwen3_5TextModel(_build_text_config(['linear_attention']))

    def test_linear_attention_sp_uses_cu_seq_lens_and_keeps_z_local(self):
        captured = {
            'cu_seqlens': None,
            'seq_to_head_calls': 0,
            'head_to_seq_calls': 0,
            'norm_z_shape': None,
        }

        def fake_conv(x, weight, bias, activation, seq_idx=None, backend=None, cu_seqlens=None):
            del weight, bias, activation, seq_idx, backend
            captured['cu_seqlens'] = cu_seqlens.clone() if cu_seqlens is not None else None
            return x

        def fake_chunk_rule(query, key, value, g, beta, initial_state=None, output_final_state=False,
                            use_qk_l2norm_in_kernel=False, cu_seqlens=None):
            del query, key, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel
            captured['cu_seqlens'] = cu_seqlens.clone() if cu_seqlens is not None else None
            return value, None

        def fake_recurrent_rule(query, key, value, g, beta, initial_state=None, output_final_state=False,
                                use_qk_l2norm_in_kernel=False):
            del query, key, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel
            return value, None

        def fake_seq_to_head(tensor, context):
            del context
            captured['seq_to_head_calls'] += 1
            return tensor

        def fake_head_to_seq(tensor, context):
            del context
            captured['head_to_seq_calls'] += 1
            return tensor

        class DummyNorm(torch.nn.Module):

            def forward(self, x, z):
                captured['norm_z_shape'] = tuple(z.shape)
                return x + z

        with patch.object(tw_qwen35, '_FLA_CAUSAL_CONV1D_FN', fake_conv), \
                patch.object(tw_qwen35, '_FLA_CAUSAL_CONV1D_UPDATE', lambda *args, **kwargs: args[0]), \
                patch.object(tw_qwen35, '_FLA_CHUNK_GATED_DELTA_RULE', fake_chunk_rule), \
                patch.object(tw_qwen35, '_FLA_FUSED_RECURRENT_GATED_DELTA_RULE', fake_recurrent_rule), \
                patch.object(tw_qwen35, '_HAS_CAUSAL_CONV1D', True), \
                patch.object(tw_qwen35, '_seq_to_head_shard', side_effect=fake_seq_to_head), \
                patch.object(tw_qwen35, '_head_to_seq_shard', side_effect=fake_head_to_seq):
            config = _build_text_config(['linear_attention'])
            module = tw_qwen35.TwinkleQwen3_5GatedDeltaNet(config, layer_idx=0)
            module.norm = DummyNorm()
            hidden_states = torch.randn(1, 2, config.hidden_size)
            attention_mask = torch.ones(1, 2, dtype=torch.int64)
            cu_seq_lens_q = torch.tensor([0, 2], dtype=torch.int32)
            context = SequenceParallelContext(
                sp_group='dummy_group',
                sp_world_size=2,
                rank=0,
                world_size=2,
                real_position_ids=torch.tensor([[0, 1]], dtype=torch.long),
                is_packed=False,
            )

            output = module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                cu_seq_lens_q=cu_seq_lens_q,
                sequence_parallel_context=context,
            )

        self.assertEqual(captured['seq_to_head_calls'], 5)
        self.assertEqual(captured['head_to_seq_calls'], 1)
        self.assertTrue(torch.equal(captured['cu_seqlens'], cu_seq_lens_q))
        self.assertEqual(captured['norm_z_shape'], (hidden_states.shape[0] * hidden_states.shape[1] * config.linear_num_value_heads, config.linear_value_head_dim))
        self.assertEqual(tuple(output.shape), (1, 2, config.hidden_size))


if __name__ == '__main__':
    unittest.main()
