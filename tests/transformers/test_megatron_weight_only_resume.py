import sys
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from twinkle.model.megatron.megatron import MegatronModel
from twinkle.model.megatron.multi_lora_megatron import MultiLoraMegatronModel


class MasterParameterOptimizer:
    """Use real Adam with the documented MCore model/master parameter contract."""
    def __init__(self, model_parameter):
        self.model_parameter = model_parameter
        self.master = torch.nn.Parameter(model_parameter.detach().float().clone())
        self.adam = torch.optim.Adam([self.master], lr=0.1)

    def reload_model_params(self):
        self.master.data.copy_(self.model_parameter)

    def step(self):
        self.master.grad = torch.ones_like(self.master)
        self.adam.step()
        self.model_parameter.data.copy_(self.master.to(self.model_parameter.dtype))


@pytest.mark.parametrize('model_class', [MegatronModel, MultiLoraMegatronModel])
def test_weights_only_load_refreshes_target_master_params_and_preserves_adam_state(tmp_path, model_class):
    parameter = torch.nn.Parameter(torch.tensor([7.], dtype=torch.bfloat16))
    optimizer = MasterParameterOptimizer(parameter)
    optimizer.step()
    before = deepcopy(optimizer.adam.state_dict())
    expected = MasterParameterOptimizer(torch.nn.Parameter(torch.tensor([2.], dtype=torch.bfloat16)))
    expected.adam.load_state_dict(deepcopy(before))
    expected.step()
    other = MasterParameterOptimizer(torch.nn.Parameter(torch.tensor([11.], dtype=torch.bfloat16)))
    group = SimpleNamespace(optimizer=optimizer)

    def load_weights(*args, **kwargs):
        parameter.data.fill_(2.)

    bridge = SimpleNamespace(
        load_weights=load_weights, _convert_post_process=lambda *args: None, _peft_format=False,
        hf_score_key='score.weight', config=SimpleNamespace(task_type='causal_lm'))
    wrapper = SimpleNamespace(
        _get_default_group=lambda: 'target', model=[parameter],
        strategy=SimpleNamespace(unwrap_model=lambda model: model, bridge=bridge),
        optimizer_group={'target': group, 'other': SimpleNamespace(optimizer=other)},
        multi_adapter=SimpleNamespace(save_context=lambda adapter: nullcontext('slot'), load_lora_converter=None),
        _load_task_head_checkpoint=lambda *args, **kwargs: None,
    )
    model_class.load(wrapper, str(tmp_path), adapter_name='target', load_optimizer=False)
    assert parameter.item() == 2.
    assert optimizer.adam.state_dict()['state'][0]['step'] == before['state'][0]['step']
    optimizer.step()
    torch.testing.assert_close(parameter, expected.model_parameter)
    assert other.master.item() == other.model_parameter.item() == 11.


@pytest.mark.parametrize('api', ['legacy', 'current'])
@pytest.mark.parametrize('sharding', ['dp_reshardable', 'fsdp_dtensor'])
def test_mcore_optimizer_save_uses_versioned_torch_dist_strategy(tmp_path, monkeypatch, api, sharding):
    pytest.importorskip('megatron.core', reason='requires the optional Megatron backend')
    from megatron.core import dist_checkpointing
    from megatron.core import parallel_state as mpu

    captured = {}

    def record(strategy, async_save, **kwargs):
        captured.update(strategy=strategy, async_save=async_save, **kwargs)

    def legacy_save(state_dict, checkpoint_dir, sharded_strategy=None, common_strategy=None,
                    validate_access_integrity=True, async_sharded_save=False,
                    preprocess_common_before_consistancy_check=None):
        record(sharded_strategy, async_sharded_save)

    def current_save(state_dict, checkpoint_dir, sharded_strategy=None, common_strategy=None,
                     validate_access_integrity=True, async_sharded_save=False,
                     preprocess_common_before_consistancy_check=None, content_metadata=None,
                     async_strategy='nvrx'):
        record(sharded_strategy, async_sharded_save, metadata=content_metadata, async_strategy=async_strategy)

    monkeypatch.setattr(dist_checkpointing, 'save', legacy_save if api == 'legacy' else current_save)
    monkeypatch.setattr(mpu, 'get_data_parallel_world_size', lambda **kwargs: 1)
    metadata = {'sharded': True, 'distrib_optim_sharding_type': sharding}
    wrapper = SimpleNamespace(
        model=object(),
        strategy=SimpleNamespace(get_sharded_sd_metadata=lambda: metadata),
        _generate_state_dict=lambda **kwargs: {'optimizer': {}},
    )
    optimizer = SimpleNamespace(cur_step=1, optimizer=object(), lr_scheduler=None)

    MegatronModel._save_mcore_optimizer(
        wrapper, str(tmp_path), optimizer, no_save_rng=True, thread_count=3, async_save=True)

    assert captured['strategy'].backend == 'torch_dist'
    assert captured['strategy'].version == 1
    assert captured['strategy'].thread_count == 3
    assert captured['async_save'] is False
    assert metadata == {'sharded': True, 'distrib_optim_sharding_type': sharding}
    if api == 'current':
        assert captured['metadata'] is metadata
        assert captured['async_strategy'] == 'mcore'
    else:
        assert 'metadata' not in captured
        assert 'async_strategy' not in captured
    assert (tmp_path / 'latest_checkpointed_iteration.txt').read_text() == '1'


def test_mixed_precision_optimizer_checkpoint_uses_the_actual_main_param_order():
    pytest.importorskip('megatron.core', reason='requires the optional Megatron backend')
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    from twinkle.model.megatron.megatron import _align_distributed_optimizer_param_order

    low = torch.nn.Parameter(torch.ones(8, dtype=torch.bfloat16))
    head = torch.nn.Parameter(torch.ones(3))
    master = torch.nn.Parameter(low.float().clone())
    distributed = object.__new__(DistributedOptimizer)
    distributed.model_float16_groups = [[low]]
    distributed.model_fp32_groups = [[head]]
    distributed.model_param_group_index_map = {low: (0, 0), head: (0, 1)}
    distributed.optimizer = torch.optim.Adam([head, master])
    distributed.ddp_config = SimpleNamespace(use_custom_fsdp=False)
    distributed.config = SimpleNamespace(
        use_precision_aware_optimizer=False, use_precision_aware_optimizer_no_fp8_or_ds_fp8=False)

    _align_distributed_optimizer_param_order(SimpleNamespace(chained_optimizers=[distributed]))

    for param, expected in [(low, master), (head, head)]:
        assert distributed._get_main_param_and_optimizer_states(param)['param'] is expected


def test_mcore_persisted_classifier_config(monkeypatch):
    pytest.importorskip('megatron.core', reason='requires the optional Megatron backend')
    from twinkle import Platform
    from twinkle.model.megatron.bridge.mcore import MCoreBridgeBackend

    bridge = SimpleNamespace(
        ModelConfig=lambda **kwargs: kwargs,
        hf_to_mcore_config=lambda config: {'untie_embeddings_and_output_weights': False},
    )
    monkeypatch.setitem(sys.modules, 'mcore_bridge', bridge)
    monkeypatch.setattr(Platform, 'get_platform', lambda: SimpleNamespace(
        configure_model_config=lambda config: None))
    strategy = SimpleNamespace(params_type='float32', sequence_parallel=False, variable_seq_lengths=False)
    hf_config = SimpleNamespace(architectures=['Qwen2ForSequenceClassification'], num_labels=3)

    config = MCoreBridgeBackend().build_model_config(
        hf_config, {}, strategy, untie_embeddings_and_output_weights=False)

    assert config['task_type'] == 'seq_cls'
    assert config['num_labels'] == 3
    assert config['untie_embeddings_and_output_weights'] is True
    with pytest.raises(ValueError, match='task_type'):
        MCoreBridgeBackend().build_model_config(hf_config, {}, strategy, task_type='causal_lm')
    with pytest.raises(ValueError, match='num_labels'):
        MCoreBridgeBackend().build_model_config(hf_config, {}, strategy, num_labels=2)


def test_mcore_persisted_classifier_uses_independent_output_head():
    pytest.importorskip('mcore_bridge', reason='requires the optional Megatron bridge')
    from mcore_bridge.bridge.gpt_bridge import GPTBridge

    bridge = object.__new__(GPTBridge)
    bridge.config = SimpleNamespace(
        task_type='seq_cls', untie_embeddings_and_output_weights=True, padded_vocab_size=13)
    bridge.is_multimodal = False
    bridge.tp_size = bridge.etp_size = 1
    bridge.tp_rank = bridge.etp_rank = 0
    bridge._peft_format = False
    mg_model = SimpleNamespace(
        output_layer=SimpleNamespace(weight=torch.nn.Parameter(torch.zeros(1, 8))),
        decoder=SimpleNamespace(final_layernorm=SimpleNamespace(weight=torch.nn.Parameter(torch.zeros(8)))),
    )
    score = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    embedding = score.repeat(13, 1) + 100
    norm = torch.ones(8)
    def lazy(value):
        return SimpleNamespace(load=lambda: value)

    bridge._convert_post_process(mg_model, {
        'score.weight': lazy(score),
        'model.embed_tokens.weight': lazy(embedding),
        'model.norm.weight': lazy(norm),
    }, '', True)

    torch.testing.assert_close(mg_model.output_layer.weight, score)
    torch.testing.assert_close(mg_model.decoder.final_layernorm.weight, norm)
