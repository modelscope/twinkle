import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils import get_peft_model_state_dict
from torch import nn

from twinkle.model.transformers.spectral_hybrid_lora import (
    CANDIDATE_TYPES,
    allocate_spectral_modules,
    build_spectral_lora_config,
    build_spectral_param_groups,
    compute_spectral_scores,
    compute_spectral_metrics,
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
    from twinkle.model.transformers import spectral_hybrid_lora

    model = TinyDecoder(num_layers=1)
    config = LoraConfig(r=2, target_modules=['q_proj'])
    monkeypatch.setattr(spectral_hybrid_lora.dist, 'is_available', lambda: True)
    monkeypatch.setattr(spectral_hybrid_lora.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(spectral_hybrid_lora.dist, 'get_rank', lambda: 0)
    monkeypatch.setattr(
        spectral_hybrid_lora.dist,
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
