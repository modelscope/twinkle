from contextlib import contextmanager

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils import get_peft_model_state_dict
from torch import nn

from twinkle.model.transformers.hybrid.fft_slots import HybridFftSlots
from twinkle.model.transformers.hybrid.spectral_allocation import (
    CANDIDATE_TYPES,
    allocate_spectral_modules,
    build_spectral_lora_config,
    build_spectral_param_groups,
    compute_spectral_scores,
    compute_spectral_metrics,
    load_spectral_allocation,
    resolve_spectral_config_path,
    select_spectral_targets,
)


class TinyDecoder(nn.Module):

    def __init__(self, num_layers=2, dim=8):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.q_proj = nn.Linear(dim, dim, bias=False)
            layer.self_attn.k_proj = nn.Linear(dim, dim, bias=False)
            layer.self_attn.v_proj = nn.Linear(dim, dim, bias=False)
            layer.self_attn.o_proj = nn.Linear(dim, dim, bias=False)
            layer.mlp = nn.Module()
            layer.mlp.gate_proj = nn.Linear(dim, dim, bias=False)
            layer.mlp.up_proj = nn.Linear(dim, dim, bias=False)
            layer.mlp.down_proj = nn.Linear(dim, dim, bias=False)
            self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            hidden = layer.self_attn.q_proj(inputs)
            inputs = layer.mlp.down_proj(layer.mlp.up_proj(hidden))
        return inputs


def _fft_slot_module(hybrid, allocated_name, slot=0):
    wrapper = hybrid._get_fft_wrapper(allocated_name)
    return wrapper.modules_to_save[f'fft_{slot}']


@contextmanager
def _adapter(manager, hybrid, adapter_name, disable_lora=False):
    with manager.adapter(adapter_name, disable_lora=disable_lora):
        if disable_lora:
            hybrid.deactivate_fft_slots()
        else:
            hybrid.activate_fft_slot(adapter_name)
        try:
            yield
        finally:
            hybrid.deactivate_fft_slots()


def _install_hybrid(manager, model, s_fft):
    hybrid = HybridFftSlots(manager, s_fft)
    manager.module = model
    hybrid.install_fft_slots()
    return hybrid


def _register_hybrid(manager, hybrid, adapter_name, config):
    config.target_modules = set(hybrid.resolve_lora_targets(config.target_modules))
    config.modules_to_save = list(hybrid.s_fft)
    manager.acquire_lora(adapter_name, config)
    hybrid.register_adapter(adapter_name)


@pytest.mark.parametrize('bind_device,expects_device_id', [(None, True), (lambda _backend: False, False)])
def test_initialize_process_group_preserves_backend_device_binding(monkeypatch, bind_device, expects_device_id):
    import torch.distributed as dist
    from twinkle import Platform, torch_util
    from twinkle.model.base import initialize_process_group

    calls = []
    monkeypatch.setattr(dist, 'is_initialized', lambda: False)
    monkeypatch.setattr(dist, 'init_process_group', lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(Platform, 'get_world_size', lambda: 2)
    monkeypatch.setattr(Platform, 'get_rank', lambda: 0)
    monkeypatch.setattr(Platform, 'get_local_device', lambda: 'cpu')
    monkeypatch.setattr(Platform, 'device_backend', lambda: 'nccl')
    monkeypatch.setattr(torch_util, 'set_device', lambda: None)

    initialize_process_group(bind_device)

    assert len(calls) == 1
    assert ('device_id' in calls[0]) is expects_device_id


def test_spectral_metrics_match_weighted_formula():
    singular_values = torch.tensor([4.0, 2.0, 1.0, 0.5], dtype=torch.float64)
    metrics = compute_spectral_metrics(singular_values, r=1)

    probabilities = singular_values / singular_values.sum()
    effective_rank = torch.exp(-(probabilities * probabilities.log()).sum()).item()
    rank_coverage = (singular_values[0].square() / singular_values.square().sum()).item()
    condition_number = 8.0
    decay = torch.diff(singular_values.log()).abs().mean().item()
    expected = (
        0.3 * (effective_rank / singular_values.numel())
        + 0.3 * (1.0 - rank_coverage)
        + 0.2 * (torch.log1p(torch.tensor(condition_number)).item() / 10.0)
        + 0.2 * (1.0 / decay)
    )

    assert metrics['effective_rank'] == pytest.approx(effective_rank)
    assert metrics['rank_coverage'] == pytest.approx(rank_coverage)
    assert metrics['condition_number'] == pytest.approx(condition_number)
    assert metrics['decay'] == pytest.approx(decay)
    assert metrics['score'] == pytest.approx(expected)


@pytest.mark.parametrize('singular_values,r,message', [
    (torch.ones(4), 0, 'rank r must be positive'),
    (torch.empty(0), 1, 'non-empty singular-value vector'),
])
def test_spectral_metrics_validate_inputs(singular_values, r, message):
    with pytest.raises(ValueError, match=message):
        compute_spectral_metrics(singular_values, r=r)


def test_select_targets_covers_supported_module_types():
    model = TinyDecoder(num_layers=2)
    config = LoraConfig(r=4, target_modules=list(CANDIDATE_TYPES.values()))

    targets = select_spectral_targets(model, config)

    assert len(targets) == 14
    assert all(name.rsplit('.', 1)[-1] in CANDIDATE_TYPES.values() for name in targets)


def test_scores_cache_pretrained_singular_values(tmp_path, monkeypatch):
    model = TinyDecoder(num_layers=1)
    config = LoraConfig(r=2, target_modules=['q_proj', 'down_proj'])
    original_svdvals = torch.linalg.svdvals
    calls = []

    def record_svdvals(weight):
        calls.append(tuple(weight.shape))
        return original_svdvals(weight)

    monkeypatch.setattr(torch.linalg, 'svdvals', record_svdvals)
    scores = compute_spectral_scores(
        model, config, r=2, cache_dir=tmp_path, cache_key='tiny', log_interval=0)
    cached_scores = compute_spectral_scores(
        model, config, r=2, cache_dir=tmp_path, cache_key='tiny', log_interval=0)

    assert set(scores) == {'layers.0.mlp.down_proj', 'layers.0.self_attn.q_proj'}
    assert scores == cached_scores
    assert len(calls) == 2
    assert len(list(tmp_path.glob('spectrum-*.pt'))) == 2
    assert all('effective_rank' in scores.metrics[name] for name in scores)


def test_spectral_scores_can_skip_distributed_broadcast(monkeypatch):
    from twinkle.model.transformers.hybrid import spectral_allocation

    model = TinyDecoder(num_layers=1)
    config = LoraConfig(r=2, target_modules=['q_proj'])
    monkeypatch.setattr(spectral_allocation.dist, 'is_available', lambda: True)
    monkeypatch.setattr(spectral_allocation.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(spectral_allocation.dist, 'get_rank', lambda: 0)
    monkeypatch.setattr(
        spectral_allocation.dist,
        'broadcast_object_list',
        lambda *_args, **_kwargs: pytest.fail('broadcast should be disabled'),
    )

    scores = compute_spectral_scores(model, config, r=2, log_interval=0, broadcast=False)

    assert set(scores) == {'layers.0.self_attn.q_proj'}


def test_config_path_reuses_an_existing_explicit_config(tmp_path):
    config_path = tmp_path / 'allocation.json'
    config_path.write_text('{}', encoding='utf-8')

    resolved, should_load = resolve_spectral_config_path(str(config_path), tmp_path / 'output')

    assert resolved == config_path
    assert should_load is True


def test_config_path_computes_when_explicit_config_is_missing(tmp_path):
    config_path = tmp_path / 'missing.json'

    resolved, should_load = resolve_spectral_config_path(str(config_path), tmp_path / 'output')

    assert resolved == config_path
    assert should_load is False


def test_config_path_computes_to_default_when_not_configured(tmp_path):
    resolved, should_load = resolve_spectral_config_path(None, tmp_path)

    assert resolved == tmp_path / 'spectral_hybrid_lora_config.json'
    assert should_load is False


def test_allocation_prioritizes_high_scores_within_budget():
    scores = {'a': 0.2, 'b': 0.8, 'c': 0.4, 'd': 0.6}
    counts = {name: 100 for name in scores}

    s_fft, s_lora = allocate_spectral_modules(scores, counts, fft_ratio=0.25)

    assert s_fft == ['b']
    assert set(s_lora) == {'a', 'c', 'd'}


def test_allocation_uses_strict_ranked_prefix():
    scores = {'big': 0.9, 'small_a': 0.8, 'small_b': 0.7}
    counts = {'big': 80, 'small_a': 30, 'small_b': 15}

    s_fft, s_lora = allocate_spectral_modules(scores, counts, fft_ratio=0.7)

    assert s_fft == ['big']
    assert set(s_lora) == {'small_a', 'small_b'}


def test_allocation_rejects_full_fft_budget():
    with pytest.raises(ValueError, match=r'\[0, 1\)'):
        allocate_spectral_modules({'module': 1.0}, {'module': 10}, fft_ratio=1.0)


def test_config_requires_a_lora_target():
    with pytest.raises(ValueError, match='at least one LoRA module'):
        build_spectral_lora_config([], ['layers.0.self_attn.q_proj'])


def test_config_and_param_groups_cover_every_trainable_parameter():
    config = build_spectral_lora_config(
        s_lora=['layers.0.mlp.down_proj'],
        s_fft=['layers.0.self_attn.q_proj'],
        r=4,
        lora_alpha=8,
    )
    model = get_peft_model(TinyDecoder(num_layers=1), config)

    groups = build_spectral_param_groups(model, lr_lora=2.5e-5, lr_fft=1e-6)

    assert {group['lr'] for group in groups} == {2.5e-5, 1e-6}
    grouped = {id(param) for group in groups for param in group['params']}
    trainable = {id(param) for param in model.parameters() if param.requires_grad}
    assert grouped == trainable


@pytest.mark.parametrize('strategy_cls', [
    pytest.param('accelerate', id='accelerate'),
    pytest.param('native_fsdp', id='native-fsdp'),
])
def test_strategy_adapter_state_includes_full_modules(strategy_cls):
    if strategy_cls == 'accelerate':
        from twinkle.model.transformers.strategy.accelerate import AccelerateStrategy as Strategy
        strategy = object.__new__(Strategy)
        strategy.unwrap_model = lambda model, *args: model
    else:
        from twinkle.model.transformers.strategy.native_fsdp import NativeFSDPStrategy as Strategy
        strategy = object.__new__(Strategy)
        strategy.ep_fsdp_device_mesh = None

    config = build_spectral_lora_config(
        s_lora=['layers.0.mlp.down_proj'],
        s_fft=['layers.0.self_attn.q_proj'],
        r=4,
        lora_alpha=8,
    )
    model = get_peft_model(TinyDecoder(num_layers=1), config)

    state = strategy.get_adapter_state_dict(model, 'default')

    assert any('.lora_A.' in name for name in state)
    assert any('.modules_to_save.default.' in name for name in state)


def test_accelerate_fsdp_optimizer_uses_full_rank0_state_options():
    from twinkle.model.transformers.strategy.accelerate import AccelerateStrategy

    save_options = AccelerateStrategy._prepare_full_optimizer_state_dict_options(for_load=False)
    load_options = AccelerateStrategy._prepare_full_optimizer_state_dict_options(for_load=True)

    assert save_options.full_state_dict is True
    assert save_options.cpu_offload is True
    assert save_options.broadcast_from_rank0 is False
    assert load_options.full_state_dict is True
    assert load_options.cpu_offload is False
    assert load_options.broadcast_from_rank0 is True


def test_accelerate_fsdp_optimizer_save_and_load_ignore_sharded_plugin_options(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from twinkle.model.transformers.strategy.accelerate import AccelerateStrategy
    import torch.distributed.checkpoint.state_dict as state_dict_api

    strategy = object.__new__(AccelerateStrategy)
    strategy.accelerator = SimpleNamespace(process_index=0)
    strategy._get_fsdp_plugin = lambda: SimpleNamespace(fsdp_version=2)
    calls = {}

    def fake_get(_model, _optimizer, *, options):
        calls['save'] = options
        return {'state': {}, 'param_groups': []}

    def fake_set(_model, _optimizer, state, *, options):
        calls['load'] = (state, options)

    monkeypatch.setattr(state_dict_api, 'get_optimizer_state_dict', fake_get)
    monkeypatch.setattr(state_dict_api, 'set_optimizer_state_dict', fake_set)
    checkpoint = tmp_path / 'optimizer.pt'
    strategy.save_optimizer_checkpoint(object(), object(), str(checkpoint))
    strategy.load_optimizer_checkpoint(object(), object(), str(checkpoint))

    assert calls['save'].full_state_dict is True
    assert calls['save'].cpu_offload is True
    loaded_state, load_options = calls['load']
    assert loaded_state == {'state': {}, 'param_groups': []}
    assert load_options.full_state_dict is True
    assert load_options.broadcast_from_rank0 is True


def test_native_fsdp_full_state_reconstructs_ep_experts(monkeypatch):
    import torch.distributed.checkpoint.state_dict as state_dict_api
    from twinkle import Platform
    from twinkle.model.transformers.strategy import native_fsdp
    from twinkle.model.transformers.strategy.native_fsdp import NativeFSDPStrategy

    class EpDimension:

        @staticmethod
        def size():
            return 2

        @staticmethod
        def get_group():
            return 'ep-group'

    strategy = object.__new__(NativeFSDPStrategy)
    strategy.device_mesh = object()
    strategy.ep_fsdp_device_mesh = {'ep': EpDimension()}
    strategy.unwrap_model = lambda model: model
    captured_options = []

    def get_model_state_dict(_model, *, options):
        captured_options.append(options)
        return {
            'experts.weight': torch.tensor([[1.0]]),
            'dense.weight': torch.tensor([[3.0]]),
        }

    def all_gather(output, value, *, group):
        assert group == 'ep-group'
        output[0].copy_(value)
        output[1].copy_(value + 1)

    monkeypatch.setattr(state_dict_api, 'get_model_state_dict', get_model_state_dict)
    monkeypatch.setattr(native_fsdp, '_detect_ep_expert_names', lambda _model: {'experts.weight'})
    monkeypatch.setattr(native_fsdp.dist, 'all_gather', all_gather)
    monkeypatch.setattr(Platform, 'get_local_device', lambda: 'cpu')
    monkeypatch.setattr(Platform, 'is_master', lambda: True)

    state = strategy.get_full_state_dict(object())

    assert captured_options[0].full_state_dict is True
    assert captured_options[0].cpu_offload is False
    assert torch.equal(state['experts.weight'], torch.tensor([[1.0], [2.0]]))
    assert torch.equal(state['dense.weight'], torch.tensor([[3.0]]))


def test_twinkle_checkpoint_normalization_round_trips_full_modules(tmp_path):
    from safetensors.torch import save_file
    from twinkle.model.transformers.strategy.accelerate import AccelerateStrategy
    from twinkle.model.transformers.transformers import TransformersModel

    torch.manual_seed(0)
    base = TinyDecoder(num_layers=1)
    base_state = {name: value.detach().clone() for name, value in base.state_dict().items()}
    config = build_spectral_lora_config(
        s_lora=['layers.0.mlp.down_proj'],
        s_fft=['layers.0.self_attn.q_proj'],
        r=4,
        lora_alpha=8,
    )
    peft_model = get_peft_model(base, config)
    with torch.no_grad():
        for param in peft_model.parameters():
            if param.requires_grad:
                param.add_(0.05)
    inputs = torch.randn(2, 8)
    expected = peft_model(inputs).detach()

    strategy = object.__new__(AccelerateStrategy)
    strategy.unwrap_model = lambda model, *args: model
    twinkle_model = object.__new__(TransformersModel)
    twinkle_model.strategy = strategy
    twinkle_model.__dict__['model'] = peft_model
    saved = twinkle_model._get_adapter_state_dict_for_save('default')

    assert set(saved) == set(get_peft_model_state_dict(peft_model, adapter_name='default'))
    assert 'base_model.model.layers.0.self_attn.q_proj.weight' in saved

    peft_model.peft_config['default'].save_pretrained(tmp_path)
    save_file({name: value.contiguous() for name, value in saved.items()},
              str(tmp_path / 'adapter_model.safetensors'))
    reloaded_base = TinyDecoder(num_layers=1)
    reloaded_base.load_state_dict(base_state)
    loaded = PeftModel.from_pretrained(reloaded_base, tmp_path)

    assert torch.allclose(loaded(inputs), expected, atol=1e-5)


@pytest.mark.parametrize('adapter_name', ['default', 'spectral_hybrid'])
def test_trainable_parameter_filter_includes_full_modules(adapter_name):
    from twinkle.model.transformers.transformers import TransformersModel

    config = build_spectral_lora_config(
        s_lora=['layers.0.mlp.down_proj'],
        s_fft=['layers.0.self_attn.q_proj'],
        r=4,
        lora_alpha=8,
    )
    peft_model = get_peft_model(TinyDecoder(num_layers=1), config, adapter_name=adapter_name)
    model = object.__new__(TransformersModel)
    model.strategy = type('Strategy', (), {'unwrap_model': lambda _self, inner: inner})()
    model.__dict__['model'] = peft_model

    selected = model._get_trainable_parameters(adapter_name)
    expected = {name for name, param in peft_model.named_parameters() if param.requires_grad}

    assert set(selected) == expected
    assert any('.modules_to_save.' in name for name in selected)


def test_load_fixed_server_allocation_only_requires_fft(tmp_path):
    allocation = tmp_path / 'allocation.json'
    allocation.write_text(
        '{"method":"spectral_hybrid",'
        '"s_fft":["layers.0.self_attn.q_proj"]}',
        encoding='utf-8',
    )

    s_fft = load_spectral_allocation(allocation)

    assert s_fft == ['layers.0.self_attn.q_proj']

    allocation.write_text('{"s_lora":["ignored"]}', encoding='utf-8')
    with pytest.raises(ValueError, match='at least one S_FFT'):
        load_spectral_allocation(allocation)


def test_server_hybrid_config_is_strict():
    from pydantic import ValidationError
    from twinkle.server.config.application_spec import ModelArgs

    base = {
        'model_id': 'tiny',
        'device_group': {},
        'device_mesh': {},
        'backend': 'transformers',
    }
    config = ModelArgs(**base, hybrid={'allocation_path': '/shared/allocation.json'})
    assert config.hybrid.default_lr_fft == pytest.approx(1e-6)

    with pytest.raises(ValidationError, match='extra_forbidden'):
        ModelArgs(**base, spectral_hybrid={'allocation_path': '/shared/allocation.json'})
    with pytest.raises(ValidationError, match='extra_forbidden'):
        ModelArgs(**base, hybrid={
            'allocation_path': '/shared/allocation.json',
            'default_lr_fFt': 1e-5,
        })
    with pytest.raises(ValidationError, match='greater_than'):
        ModelArgs(**base, hybrid={
            'allocation_path': '/shared/allocation.json',
            'default_lr_fft': -1.0,
        })
    with pytest.raises(ValidationError, match='only supported by the transformers backend'):
        ModelArgs(**{**base, 'backend': 'megatron'}, hybrid={
            'allocation_path': '/shared/allocation.json',
        })


def test_lora_config_copy_is_explicit_and_non_mutating():
    from twinkle.model.transformers.multi_lora_transformers import MultiLoraTransformersModel

    original = LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj'])
    copied = MultiLoraTransformersModel._copy_lora_config(original)

    assert copied is not original
    assert copied.to_dict() == original.to_dict()
    with pytest.raises(ValueError, match='model path or hub'):
        MultiLoraTransformersModel._copy_lora_config('/adapter/path')
    with pytest.raises(TypeError, match='LoraConfig'):
        MultiLoraTransformersModel._copy_lora_config(object())


def test_hybrid_rejects_client_owned_fft_and_target_parameter_config():
    from twinkle.model.transformers.hybrid import SpectralHybridTransformersModel

    model = object.__new__(SpectralHybridTransformersModel)
    modules_to_save = LoraConfig(
        r=2,
        target_modules=['q_proj'],
        modules_to_save=['k_proj'],
    )
    with pytest.raises(ValueError, match='modules_to_save is controlled by the server'):
        model.add_adapter_to_model('tenant', modules_to_save, adapter_mode='hybrid')

    target_parameters = LoraConfig(r=2, target_modules=['q_proj'])
    target_parameters.target_parameters = ['weight']
    with pytest.raises(ValueError, match='target_parameters is not supported'):
        model.add_adapter_to_model('tenant', target_parameters, adapter_mode='hybrid')


def test_transformers_server_selects_hybrid_model_only_when_configured(monkeypatch):
    from twinkle.server.model import app
    from twinkle.server.model.backends import transformers_model

    class Regular:

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Hybrid(Regular):
        pass

    monkeypatch.setattr(transformers_model, 'TwinkleCompatTransformersModel', Regular)
    monkeypatch.setattr(transformers_model, 'TwinkleCompatSpectralHybridTransformersModel', Hybrid)

    assert type(app._make_transformers_model({'model_id': 'tiny'})) is Regular
    assert type(app._make_transformers_model({
        'model_id': 'tiny',
        'hybrid': {'allocation_path': '/shared/allocation.json'},
    })) is Hybrid


def test_allocation_rejects_duplicate_modules():
    from twinkle.model.multi_lora import MultiLora

    manager = MultiLora(max_loras=1, max_r=4)
    with pytest.raises(ValueError, match='same module more than once'):
        HybridFftSlots(
            manager,
            ['layers.0.self_attn.q_proj', 'layers.0.self_attn.q_proj'],
        )

    manager = MultiLora(max_loras=1, max_r=4)
    manager.patch(TinyDecoder(num_layers=1), target_modules='all-linear')
    hybrid = HybridFftSlots(manager, ['q_proj'])
    with pytest.raises(ValueError, match='resolved to 0 layers'):
        hybrid.install_fft_slots()


def test_allocation_rejects_aliases_for_the_same_module():
    from twinkle.model.multi_lora import MultiLora

    manager = MultiLora(max_loras=1, max_r=4)
    manager.patch(TinyDecoder(num_layers=1), target_modules='all-linear')
    with pytest.raises(ValueError, match='aliases resolve to the same layer'):
        HybridFftSlots(
            manager,
            ['layers.0.self_attn.q_proj', 'base_model.model.layers.0.self_attn.q_proj'],
        )


def test_fft_modules_do_not_need_lora_preallocation():
    from peft.utils import ModulesToSaveWrapper
    from twinkle.model.multi_lora import MultiLora

    base = TinyDecoder(num_layers=1)
    manager = MultiLora(max_loras=1, max_r=4)
    model = manager.patch(base, target_modules=['down_proj'])
    hybrid = _install_hybrid(manager, model, ['layers.0.self_attn.q_proj'])
    config = LoraConfig(r=2, lora_alpha=4, target_modules=['layers.0.mlp.down_proj'])
    _register_hybrid(manager, hybrid, 'hybrid', config)

    fft_wrapper = hybrid._get_fft_wrapper('layers.0.self_attn.q_proj')
    fft_layer = _fft_slot_module(hybrid, 'layers.0.self_attn.q_proj')
    assert isinstance(fft_wrapper, ModulesToSaveWrapper)
    assert isinstance(fft_layer, nn.Linear)
    assert not any('_twinkle_fft' in name for name, _ in model.named_parameters())
    with torch.no_grad():
        fft_layer.weight.add_(0.1)
    inputs = torch.randn(2, 8)
    with _adapter(manager, hybrid, 'hybrid'):
        expected = model(inputs).detach()
    full_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    merged_state = hybrid.build_merged_state_dict('hybrid', full_state)
    deployed = TinyDecoder(num_layers=1)
    deployed.load_state_dict(merged_state)
    assert torch.allclose(deployed(inputs), expected, atol=1e-5)


def test_hybrid_lora_targets_follow_tenant_config_but_exclude_fft():
    from twinkle.model.multi_lora import MultiLora

    manager = MultiLora(max_loras=1, max_r=4)
    model = manager.patch(TinyDecoder(num_layers=1), target_modules=['down_proj'])
    hybrid = _install_hybrid(manager, model, ['layers.0.self_attn.q_proj'])

    targets = hybrid.resolve_lora_targets('all-linear')

    assert len(targets) == 1
    assert any(name.endswith('layers.0.mlp.down_proj') for name in targets)
    assert not any(name.endswith('layers.0.self_attn.q_proj') for name in targets)

    only_fft = hybrid.resolve_lora_targets(['q_proj'])
    assert only_fft == []


def test_fft_layer_never_stacks_a_lora_delta():
    from twinkle.model.multi_lora import MultiLora

    manager = MultiLora(max_loras=1, max_r=4)
    model = manager.patch(TinyDecoder(num_layers=1), target_modules='all-linear')
    hybrid = _install_hybrid(manager, model, ['layers.0.self_attn.q_proj'])
    config = LoraConfig(r=2, lora_alpha=4, target_modules='all-linear', init_lora_weights=False)
    _register_hybrid(manager, hybrid, 'hybrid', config)
    tenant = manager.find_lora_by_tenant('hybrid')
    q_proj = model.get_submodule(hybrid.allocated_to_layer_name['layers.0.self_attn.q_proj'])
    inputs = torch.randn(2, 8)

    with _adapter(manager, hybrid, 'hybrid'):
        before = model(inputs).detach()
    with torch.no_grad():
        q_proj.lora_A[tenant.adapter_name].weight[:2].fill_(10.0)
        q_proj.lora_B[tenant.adapter_name].weight[:, :2].fill_(10.0)
    with _adapter(manager, hybrid, 'hybrid'):
        after = model(inputs).detach()

    assert torch.allclose(after, before, atol=1e-6)


def test_regular_lora_can_still_train_an_fft_allocated_layer():
    from twinkle.model.multi_lora import MultiLora

    manager = MultiLora(max_loras=1, max_r=4)
    model = manager.patch(TinyDecoder(num_layers=1), target_modules='all-linear')
    hybrid = _install_hybrid(manager, model, ['layers.0.self_attn.q_proj'])
    config = LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj'], init_lora_weights=False)
    manager.acquire_lora('regular', config)
    tenant = manager.find_lora_by_tenant('regular')
    q_proj = model.get_submodule(hybrid.allocated_to_layer_name['layers.0.self_attn.q_proj'])
    inputs = torch.randn(2, 8)

    with manager.adapter('regular'):
        before = model(inputs).detach()
    with torch.no_grad():
        q_proj.lora_A[tenant.adapter_name].weight[:2].fill_(0.5)
        q_proj.lora_B[tenant.adapter_name].weight[:, :2].fill_(0.5)
    with manager.adapter('regular'):
        after = model(inputs).detach()

    assert not torch.allclose(after, before)
    assert hybrid._get_fft_wrapper('layers.0.self_attn.q_proj').active_adapters == []


def _make_multi_tenant_hybrid():
    from twinkle.model.multi_lora import MultiLora

    torch.manual_seed(42)
    base = TinyDecoder(num_layers=1, dim=8)
    manager = MultiLora(max_loras=2, max_r=4)
    model = manager.patch(base, target_modules='all-linear')
    spectral = _install_hybrid(manager, model, ['layers.0.self_attn.q_proj'])
    manager.save_initial_weights()
    hybrid_config = LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=['layers.0.mlp.down_proj'],
        modules_to_save=['layers.0.self_attn.q_proj'],
        init_lora_weights=False,
    )
    regular_config = LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=['layers.0.self_attn.v_proj'],
        init_lora_weights=False,
    )
    _register_hybrid(manager, spectral, 'hybrid', hybrid_config)
    manager.acquire_lora('regular', regular_config)
    return base, model, manager, spectral


def test_multi_tenant_hybrid_isolation_and_non_destructive_merge():
    base, model, manager, spectral = _make_multi_tenant_hybrid()
    inputs = torch.randn(2, 8)
    with manager.adapter('regular'):
        regular_before = model(inputs).detach().clone()

    hybrid = manager.find_lora_by_tenant('hybrid')
    modules = dict(model.named_modules())
    down = next(
        layer for name, layer in modules.items()
        if name.endswith('layers.0.mlp.down_proj')
    )
    q_proj = _fft_slot_module(spectral, 'layers.0.self_attn.q_proj', hybrid.index)
    with torch.no_grad():
        down.lora_A[hybrid.adapter_name].weight[:2].fill_(0.15)
        down.lora_B[hybrid.adapter_name].weight[:, :2].fill_(0.10)
        q_proj.weight.add_(0.05)

    before_export = {name: value.detach().clone() for name, value in model.named_parameters()}
    with _adapter(manager, spectral, 'hybrid'):
        hybrid_output = model(inputs).detach()
    full_state = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
    merged_state = spectral.build_merged_state_dict('hybrid', full_state)

    deployed = TinyDecoder(num_layers=1, dim=8)
    deployed.load_state_dict(merged_state)
    assert torch.allclose(deployed(inputs), hybrid_output, atol=1e-5)
    assert all(torch.equal(before_export[name], value) for name, value in model.named_parameters())

    with manager.adapter('regular'):
        assert torch.allclose(model(inputs), regular_before, atol=1e-6)


def test_disable_lora_disables_hybrid_fft_slot_too():
    _, model, manager, spectral = _make_multi_tenant_hybrid()
    inputs = torch.randn(2, 8)
    hybrid = manager.find_lora_by_tenant('hybrid')
    fft_layer = _fft_slot_module(spectral, 'layers.0.self_attn.q_proj', hybrid.index)
    with _adapter(manager, spectral, 'hybrid', disable_lora=True):
        base = model(inputs).detach()
    with torch.no_grad():
        fft_layer.weight.add_(1.0)

    with _adapter(manager, spectral, 'hybrid', disable_lora=True):
        disabled = model(inputs).detach()

    assert torch.allclose(disabled, base, atol=1e-6)


def test_hybrid_training_state_round_trip_and_release_resets_fft_slot():
    _, model, manager, spectral = _make_multi_tenant_hybrid()
    hybrid = manager.find_lora_by_tenant('hybrid')
    fft_layer = _fft_slot_module(spectral, 'layers.0.self_attn.q_proj', hybrid.index)
    with torch.no_grad():
        fft_layer.weight.add_(0.25)
    full_state = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
    saved = spectral.build_training_state_dict('hybrid', full_state)

    with torch.no_grad():
        fft_layer.weight.zero_()
    manager.set_state_dict('hybrid', saved)
    spectral.set_fft_state_dict('hybrid', saved)
    assert torch.equal(
        fft_layer.weight,
        saved['base_model.model.layers.0.self_attn.q_proj.weight'],
    )

    spectral.reset_adapter_slot('hybrid')
    spectral.unregister_adapter('hybrid')
    manager.release_lora('hybrid')
    wrapper = spectral._get_fft_wrapper('layers.0.self_attn.q_proj')
    assert torch.equal(fft_layer.weight, wrapper.original_module.weight)


def test_fft_state_traversal_includes_buffers():
    from twinkle.model.multi_lora import MultiLora

    base = TinyDecoder(num_layers=1)
    q_proj = base.layers[0].self_attn.q_proj
    q_proj.register_buffer('calibration', torch.tensor([1.0, 2.0]))
    manager = MultiLora(max_loras=1, max_r=4)
    model = manager.patch(base, target_modules='all-linear')
    fft_slots = _install_hybrid(manager, model, ['layers.0.self_attn.q_proj'])
    config = LoraConfig(r=2, lora_alpha=4, target_modules=['down_proj'])
    _register_hybrid(manager, fft_slots, 'hybrid', config)
    fft_module = _fft_slot_module(fft_slots, 'layers.0.self_attn.q_proj')
    state_key = 'base_model.model.layers.0.self_attn.q_proj.calibration'

    fft_module.calibration.fill_(3.0)
    saved = fft_slots.get_fft_state_dict('hybrid')
    assert torch.equal(saved[state_key], torch.tensor([3.0, 3.0]))

    fft_module.calibration.zero_()
    fft_slots.set_fft_state_dict('hybrid', saved)
    assert torch.equal(fft_module.calibration, torch.tensor([3.0, 3.0]))

    fft_slots.reset_adapter_slot('hybrid')
    assert torch.equal(fft_module.calibration, torch.tensor([1.0, 2.0]))


def test_multi_tenant_optimizer_parameters_and_learning_rates_are_isolated():
    from twinkle.model.transformers.hybrid import SpectralHybridTransformersModel

    _, model, manager, spectral = _make_multi_tenant_hybrid()
    wrapper = object.__new__(SpectralHybridTransformersModel)
    wrapper.multi_adapter = manager
    wrapper.fft_slots = spectral
    wrapper.strategy = type('Strategy', (), {'unwrap_model': lambda _self, inner: inner})()
    wrapper.__dict__['model'] = model
    wrapper.default_lr_lora = 2.5e-5
    wrapper.default_lr_fft = 1e-6

    regular = wrapper._get_trainable_parameters('regular')
    hybrid = wrapper._get_trainable_parameters('hybrid')
    groups = wrapper._create_param_group('hybrid', weight_decay=0.0)

    assert regular
    assert all('.lora_' in name and '.lora_1.' in name for name in regular)
    assert all('v_proj' in name for name in regular)
    assert any('down_proj' in name and '.lora_0.' in name for name in hybrid)
    assert any('.modules_to_save.fft_0.' in name for name in hybrid)
    assert {group['lr'] for group in groups} == {2.5e-5, 1e-6}
    assert {id(param) for group in groups for param in group['params']} == {id(param) for param in hybrid.values()}


def test_regular_lora_rejects_targets_outside_preallocation():
    _, _, manager, _ = _make_multi_tenant_hybrid()

    with pytest.raises(ValueError, match='outside the preallocated range'):
        manager.validate_tenant_target_modules(['does_not_exist'])


def test_merged_only_hybrid_checkpoint_cannot_resume(tmp_path):
    from twinkle.model.transformers.hybrid import SpectralHybridTransformersModel

    wrapper = object.__new__(SpectralHybridTransformersModel)
    with pytest.raises(ValueError, match='merged-only checkpoint'):
        wrapper._resume_spectral_hybrid(str(tmp_path), 'hybrid', False)


def test_hybrid_save_optimizer_requires_configured_optimizer(tmp_path):
    from types import SimpleNamespace
    from twinkle.model.transformers.hybrid import SpectralHybridTransformersModel

    _, model, manager, spectral = _make_multi_tenant_hybrid()
    wrapper = object.__new__(SpectralHybridTransformersModel)
    wrapper.multi_adapter = manager
    wrapper.fft_slots = spectral
    wrapper.__dict__['model'] = model
    wrapper.optimizer_group = {
        'hybrid': SimpleNamespace(
            cur_step=0,
            gradient_accumulation_steps=1,
            optimizer=None,
            train_status=SimpleNamespace(loss_value=None, num_tokens=0),
        )
    }

    with pytest.raises(ValueError, match='optimizer must be configured'):
        wrapper._validate_hybrid_training_checkpoint_boundary('hybrid')


def test_hybrid_save_writes_deployable_model_and_lossless_training_state(tmp_path):
    from types import SimpleNamespace
    from transformers import PretrainedConfig
    from twinkle.model.transformers.hybrid import SpectralHybridTransformersModel

    _, model, manager, spectral = _make_multi_tenant_hybrid()
    hybrid = manager.find_lora_by_tenant('hybrid')
    fft_layer = _fft_slot_module(spectral, 'layers.0.self_attn.q_proj', hybrid.index)
    with torch.no_grad():
        fft_layer.weight.add_(0.2)

    class Strategy:

        @staticmethod
        def unwrap_model(inner):
            return inner

        @staticmethod
        def get_full_state_dict(inner):
            return {name: value.detach().cpu().clone() for name, value in inner.named_parameters()}

        @staticmethod
        def needs_wrapped_optimizer_state():
            return False

        @staticmethod
        def save_optimizer_checkpoint(_model, optimizer, output_path):
            torch.save(optimizer.state_dict(), output_path)

        @staticmethod
        def load_optimizer_checkpoint(_model, optimizer, input_path):
            optimizer.load_state_dict(torch.load(input_path, map_location='cpu', weights_only=False))

    trainable = {}
    for name, parameter in model.named_parameters():
        if '.lora_0.' in name or '.modules_to_save.fft_0.' in name:
            trainable[name] = parameter
    optimizer = torch.optim.AdamW([{
        'params': list(trainable.values()),
        'param_names': list(trainable),
        'lr': 1e-3,
    }])

    group = SimpleNamespace(
        cur_step=3,
        gradient_accumulation_steps=2,
        adapter_config=hybrid.tenant_config,
        optimizer=optimizer,
        lr_scheduler=None,
        scaler=None,
        train_status=SimpleNamespace(loss_value=None, num_tokens=0),
        do_grad_sync=lambda: True,
    )
    wrapper = object.__new__(SpectralHybridTransformersModel)
    wrapper.multi_adapter = manager
    wrapper.fft_slots = spectral
    wrapper.strategy = Strategy()
    wrapper.__dict__['model'] = model
    wrapper.optimizer_group = {'hybrid': group}
    wrapper.hf_config = PretrainedConfig()
    object.__setattr__(wrapper, '_save_tokenizer', lambda *_args, **_kwargs: None)

    checkpoint = wrapper._save_spectral_hybrid(
        'checkpoint-3',
        str(tmp_path),
        1,
        'hybrid',
        save_optimizer=True,
        consumed_train_samples=17,
    )

    assert (tmp_path / 'checkpoint-3' / 'config.json').is_file()
    assert (tmp_path / 'checkpoint-3' / 'model.safetensors').is_file()
    assert not (tmp_path / 'checkpoint-3' / 'adapter_config.json').exists()
    training_dir = tmp_path / 'checkpoint-3' / 'twinkle_training_state'
    assert (training_dir / 'adapter_config.json').is_file()
    assert '"twinkle_adapter_mode": "hybrid"' in (training_dir / 'adapter_config.json').read_text()
    assert (training_dir / 'adapter_model.safetensors').is_file()
    assert (training_dir / 'optimizer.pt').is_file()
    assert (training_dir / 'trainer_state.json').is_file()
    assert (training_dir / 'rng_state_rank_0.pt').is_file()

    expected_fft = fft_layer.weight.detach().clone()
    with torch.no_grad():
        fft_layer.weight.zero_()
    progress = wrapper._resume_spectral_hybrid(checkpoint, 'hybrid', resume_only_model=True)
    assert torch.equal(fft_layer.weight, expected_fft)
    assert progress == {
        'cur_step': 3,
        'consumed_train_samples': 17,
        'gradient_accumulation_steps': 2,
    }

    hybrid.tenant_config.lora_alpha += 1
    unchanged = fft_layer.weight.detach().clone()
    with pytest.raises(ValueError, match='config does not match checkpoint'):
        wrapper._resume_spectral_hybrid(checkpoint, 'hybrid', resume_only_model=True)
    assert torch.equal(fft_layer.weight, unchanged)

    hybrid.tenant_config.lora_alpha -= 1
    wrapper._save_spectral_hybrid('checkpoint-3', str(tmp_path), 1, 'hybrid', save_optimizer=False)
    assert not training_dir.exists()


def test_hybrid_optimizer_resume_matches_uninterrupted_training(tmp_path):
    from types import SimpleNamespace
    from transformers import PretrainedConfig
    from twinkle.model.transformers.hybrid import SpectralHybridTransformersModel

    class Strategy:

        @staticmethod
        def unwrap_model(inner):
            return inner

        @staticmethod
        def get_full_state_dict(inner):
            return {name: value.detach().cpu().clone() for name, value in inner.state_dict().items()}

        @staticmethod
        def needs_wrapped_optimizer_state():
            return False

        @staticmethod
        def save_optimizer_checkpoint(_model, optimizer, output_path):
            torch.save(optimizer.state_dict(), output_path)

        @staticmethod
        def load_optimizer_checkpoint(_model, optimizer, input_path):
            optimizer.load_state_dict(torch.load(input_path, map_location='cpu', weights_only=False))

    def build_wrapper():
        _, model, manager, spectral = _make_multi_tenant_hybrid()
        wrapper = object.__new__(SpectralHybridTransformersModel)
        wrapper.multi_adapter = manager
        wrapper.fft_slots = spectral
        wrapper.strategy = Strategy()
        wrapper.__dict__['model'] = model
        wrapper.default_lr_lora = 1e-3
        wrapper.default_lr_fft = 2e-4
        params = wrapper._create_param_group('hybrid', weight_decay=0.0)
        optimizer = torch.optim.AdamW(params)
        tenant = manager.find_lora_by_tenant('hybrid')
        group = SimpleNamespace(
            cur_step=0,
            gradient_accumulation_steps=1,
            adapter_config=tenant.tenant_config,
            optimizer=optimizer,
            lr_scheduler=None,
            scaler=None,
            train_status=SimpleNamespace(loss_value=None, num_tokens=0),
            do_grad_sync=lambda: True,
        )
        wrapper.optimizer_group = {'hybrid': group}
        wrapper.hf_config = PretrainedConfig()
        object.__setattr__(wrapper, '_save_tokenizer', lambda *_args, **_kwargs: None)
        wrapper._model_wrapped = False
        return wrapper

    def train_step(wrapper, inputs):
        group = wrapper.optimizer_group['hybrid']
        with wrapper._adapter_context('hybrid'):
            loss = wrapper.model(inputs).float().square().mean()
        loss.backward()
        group.optimizer.step()
        group.optimizer.zero_grad(set_to_none=True)
        group.cur_step += 1

    uninterrupted = build_wrapper()
    train_step(uninterrupted, torch.full((2, 8), 0.25))
    checkpoint = uninterrupted._save_spectral_hybrid(
        'resume', str(tmp_path), 1, 'hybrid', save_optimizer=True, consumed_train_samples=2)

    resumed = build_wrapper()
    progress = resumed._resume_spectral_hybrid(checkpoint, 'hybrid', resume_only_model=False)
    assert progress['cur_step'] == 1

    next_inputs = torch.full((2, 8), -0.4)
    train_step(uninterrupted, next_inputs)
    train_step(resumed, next_inputs)

    uninterrupted_state = uninterrupted.multi_adapter.get_state_dict('hybrid')
    resumed_state = resumed.multi_adapter.get_state_dict('hybrid')
    assert set(uninterrupted_state) == set(resumed_state)
    for name in uninterrupted_state:
        assert torch.equal(uninterrupted_state[name], resumed_state[name]), name

    uninterrupted_optimizer = uninterrupted.optimizer_group['hybrid'].optimizer.state_dict()
    resumed_optimizer = resumed.optimizer_group['hybrid'].optimizer.state_dict()
    assert uninterrupted_optimizer['param_groups'] == resumed_optimizer['param_groups']
    for parameter_id, state in uninterrupted_optimizer['state'].items():
        for state_name, value in state.items():
            restored = resumed_optimizer['state'][parameter_id][state_name]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, restored)
            else:
                assert value == restored


def test_hybrid_checkpoint_reloads_with_auto_model(tmp_path):
    from types import SimpleNamespace
    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
    from twinkle.model.multi_lora import MultiLora
    from twinkle.model.transformers.hybrid import SpectralHybridTransformersModel

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=32,
        tie_word_embeddings=True,
    )
    manager = MultiLora(max_loras=1, max_r=4)
    model = manager.patch(LlamaForCausalLM(config), target_modules='all-linear')
    spectral = _install_hybrid(manager, model, ['model.layers.0.self_attn.q_proj'])
    manager.save_initial_weights()
    tenant_config = LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=['model.layers.0.mlp.down_proj'],
        modules_to_save=['model.layers.0.self_attn.q_proj'],
        init_lora_weights=False,
    )
    _register_hybrid(manager, spectral, 'hybrid', tenant_config)
    tenant = manager.find_lora_by_tenant('hybrid')
    modules = dict(model.named_modules())
    with torch.no_grad():
        _fft_slot_module(spectral, 'model.layers.0.self_attn.q_proj').weight.add_(0.02)
        down = next(
            layer for name, layer in modules.items()
            if name.endswith('model.layers.0.mlp.down_proj')
        )
        down.lora_A[tenant.adapter_name].weight[:2].fill_(0.1)
        down.lora_B[tenant.adapter_name].weight[:, :2].fill_(0.1)
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    with _adapter(manager, spectral, 'hybrid'), torch.no_grad():
        expected = model(input_ids=input_ids).logits

    class Strategy:

        @staticmethod
        def unwrap_model(inner):
            return inner

        @staticmethod
        def get_full_state_dict(inner):
            return {name: value.detach().cpu().clone() for name, value in inner.named_parameters()}

    wrapper = object.__new__(SpectralHybridTransformersModel)
    wrapper.multi_adapter = manager
    wrapper.fft_slots = spectral
    wrapper.strategy = Strategy()
    wrapper.__dict__['model'] = model
    wrapper.optimizer_group = {'hybrid': SimpleNamespace(cur_step=0)}
    wrapper.hf_config = config
    object.__setattr__(wrapper, '_save_tokenizer', lambda *_args, **_kwargs: None)
    checkpoint = wrapper._save_spectral_hybrid(
        'deploy', str(tmp_path), 1, 'hybrid', save_optimizer=False, max_shard_size='1KB')

    assert (tmp_path / 'deploy' / 'model.safetensors.index.json').is_file()
    assert len(list((tmp_path / 'deploy').glob('model-*.safetensors'))) > 1
    deployed = AutoModelForCausalLM.from_pretrained(checkpoint).eval()
    with torch.no_grad():
        actual = deployed(input_ids=input_ids).logits
    assert torch.allclose(actual, expected, atol=1e-5)

    wrapper._save_spectral_hybrid(
        'deploy', str(tmp_path), 1, 'hybrid', save_optimizer=False, max_shard_size='5GB')
    assert not (tmp_path / 'deploy' / 'model.safetensors.index.json').exists()
    assert (tmp_path / 'deploy' / 'model.safetensors').is_file()
    assert not list((tmp_path / 'deploy').glob('model-*.safetensors'))
    reloaded = AutoModelForCausalLM.from_pretrained(checkpoint).eval()
    with torch.no_grad():
        assert torch.allclose(reloaded(input_ids=input_ids).logits, expected, atol=1e-5)
