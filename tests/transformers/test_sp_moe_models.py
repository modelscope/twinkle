"""Actual text and image MoE forwards with the production SP input boundaries."""
import copy
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _model_worker(rank, directory, family):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM, Qwen3VLMoeConfig, Qwen3VLMoeForConditionalGeneration
    from twinkle.model.transformers.strategy.sequence_parallel import SequenceParallelStrategy
    from twinkle.utils import DeviceMesh

    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{directory}/pg', rank=rank, world_size=2,
                            timeout=timedelta(seconds=90))
    try:
        torch.manual_seed(379)
        text = dict(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=8, moe_intermediate_size=16,
                    num_experts=4, num_experts_per_tok=2, output_router_logits=True, use_cache=False,
                    max_position_embeddings=64)
        if family == 'qwen3':
            model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**text))
        else:
            text['rope_parameters'] = dict(rope_type='default', rope_theta=10000., mrope_section=[1, 1, 2])
            vision = dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=32,
                          patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
                          num_position_embeddings=16, deepstack_visual_indexes=[])
            model = Qwen3VLMoeForConditionalGeneration(Qwen3VLMoeConfig(
                text_config=text, vision_config=vision, image_token_id=29, video_token_id=30,
                vision_start_token_id=27, vision_end_token_id=28))
        model.set_attn_implementation('sdpa')
        model.eval()
        baseline = copy.deepcopy(model)
        mesh = DeviceMesh.from_sizes(world_size=2, fsdp_size=2, ulysses_size=2, device_type='cpu')
        strategy = SequenceParallelStrategy(device_mesh=mesh, sp_config={'enabled': True, 'ulysses_size': 2},
                                            model=model, tokenizer_id=None)
        strategy._tokenizer = SimpleNamespace(pad_token_id=0)
        strategy.initialize()
        inputs = dict(input_ids=torch.tensor([[1, 27, 29, 28, 3, 4, 5, 6, 7],
                                              [1, 27, 29, 28, 8, 9, 0, 0, 0]]),
                      attention_mask=torch.tensor([[1]*9, [1]*6+[0]*3]), output_router_logits=True)
        if family != 'qwen3':
            inputs.update(pixel_values=torch.randn(8, 24), image_grid_thw=torch.tensor([[1, 2, 2], [1, 2, 2]]),
                          mm_token_type_ids=(inputs['input_ids'] == 29).long())
        reference = baseline(**copy.deepcopy(inputs))
        reference.aux_loss.backward()
        prepared = strategy.prepare_model_inputs(copy.deepcopy(inputs))
        with strategy.context():
            output = model(**prepared)
        torch.testing.assert_close(output.aux_loss, reference.aux_loss, atol=1e-6, rtol=1e-5)
        output.aux_loss.backward()
        errors = []
        before = {}
        for name, param in model.named_parameters():
            if name.endswith('mlp.gate.weight'):
                assert param.grad is not None
                dist.all_reduce(param.grad)
                param.grad.div_(2)
                reference_grad = dict(baseline.named_parameters())[name].grad
                torch.testing.assert_close(param.grad, reference_grad, atol=1e-6, rtol=1e-4)
                errors.append(float((param.grad-reference_grad).abs().max()))
                before[name] = param.detach().clone()
        assert errors
        torch.optim.SGD(model.parameters(), lr=.01).step()
        update = sum(float((p.detach()-before[n]).abs().sum()) for n, p in model.named_parameters() if n in before)
        assert update > 0
        row = dict(family=family, batch_size=2, sequence_length=9, images=2 if family != 'qwen3' else 0,
                   aux_loss=float(output.aux_loss.detach()), reference_aux_loss=float(reference.aux_loss.detach()),
                   loss_diff=float((output.aux_loss-reference.aux_loss).abs().detach()),
                   gradient_max_diff=max(errors), router_update_l1=update, status='PASS')
        if rank == 0:
            print('MODEL_RESULT ' + json.dumps(row), flush=True)
            output_dir = Path(os.environ.get('PR379_RESULTS_DIR', directory))
            (output_dir / f'model_{family}.json').write_text(json.dumps(row, indent=2))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('family', ['qwen3', 'qwen3_vl_moe'])
def test_moe_model_aux_loss_matches_unsharded(tmp_path, family):
    mp.spawn(_model_worker, args=(str(tmp_path), family), nprocs=2, join=True)
