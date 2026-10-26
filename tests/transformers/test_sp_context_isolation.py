"""CPU regression tests for process-wide SP patches and exception cleanup."""
import pytest
import torch
import gc
import weakref
from types import SimpleNamespace

import twinkle.model.transformers.strategy.sequence_parallel as sp
from twinkle.model.transformers.strategy.sequence_parallel import linear_attention_sp


@pytest.mark.parametrize('backend', ['sdpa', 'flash_attention_2'])
def test_attention_cache_routes_current_module_and_releases_owners(monkeypatch, backend):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    seen = []

    def origin(module, query, key, value, *args, **kwargs):
        assert '_sp_module' not in kwargs
        seen.append(weakref.ref(module))
        return query, None

    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS, f'{backend}_origin', origin)
    ctx = sp.SequenceParallel()
    ctx.world_size = ctx.sp_world_size = ctx.rp_world_size = 1
    first, second = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    refs = [weakref.ref(first), weakref.ref(second)]
    sp._MODULE_TO_CTX[first] = ctx
    sp._MODULE_TO_CTX[second] = ctx
    x = torch.zeros(1, 1, 2, 2)
    method = ctx._local_sdpa_attn_impl if backend == 'sdpa' else ctx._local_flash_attn_impl
    try:
        method(first, x, x, x, None)
        method(second, x, x, x, None)
        assert [r() for r in seen] == [first, second]
        del first, second, method, ctx
        gc.collect()
        assert all(r() is None for r in refs)
    finally:
        for ref in refs:
            if ref() is not None:
                sp._MODULE_TO_CTX.pop(ref(), None)


@pytest.fixture(autouse=True)
def empty_context_stack():
    saved = list(sp._ctx_stack())
    sp._ctx_stack().clear()
    yield
    sp._ctx_stack()[:] = saved


@pytest.mark.parametrize('failure', ['forward', 'split', 'other_pre_hook'])
def test_exception_restores_enclosing_context(failure):
    outer = object()
    sp._push_ctx(outer)
    ctx = sp.SequenceParallel()
    ctx.world_size = 2 if failure == 'split' else 1

    class Model(torch.nn.Module):
        def forward(self, **kwargs):
            assert sp._current_ctx() is ctx
            raise RuntimeError('forward failed')

    model = Model()
    if failure == 'other_pre_hook':
        def fail(*args):
            assert sp._current_ctx() is ctx
            raise RuntimeError('pre-hook failed')
        model.register_forward_pre_hook(fail)
    ctx._prepare_forward_hook(model)
    with pytest.raises((RuntimeError, KeyError)):
        model()
    assert sp._ctx_stack() == [outer]
    sp._pop_ctx(outer)
    assert sp._current_ctx() is None


def test_input_split_pre_hook_failure_does_not_leak_context():
    ctx = sp.SequenceParallel()
    ctx.world_size = 2

    model = torch.nn.Linear(2, 2)
    ctx._prepare_forward_hook(model)
    with pytest.raises(KeyError, match='position_ids'):
        model(torch.zeros(1, 2))
    assert sp._current_ctx() is None

    # A later model in this thread must not inherit the failed SP model's ctx.
    plain_model = torch.nn.Linear(2, 2)
    assert sp._current_ctx() is None
    plain_model(torch.zeros(1, 2))
    assert sp._current_ctx() is None


def test_gdn_patch_routes_each_owner_and_preserves_non_sp(monkeypatch):
    class Gate(torch.nn.Module):
        def forward(self, hidden_states, **kwargs):
            return ('original', hidden_states)

    monkeypatch.setattr(linear_attention_sp, '_iter_qwen35_gated_delta_net_classes', lambda: [Gate])
    patch_type = linear_attention_sp.Qwen3_5GatedDeltaNetUlyssesPatch
    monkeypatch.setattr(patch_type, '_run_forward', staticmethod(
        lambda mod, hidden_states, **kwargs: kwargs['sequence_parallel_context']))
    policy_ctx = SimpleNamespace(world_size=2, sp_world_size=2, rp_world_size=1)
    ref_ctx = SimpleNamespace(world_size=2, sp_world_size=2, rp_world_size=1)
    policy, ref, plain = Gate(), Gate(), Gate()
    monkeypatch.setitem(sp._MODULE_TO_CTX, policy, policy_ctx)
    monkeypatch.setitem(sp._MODULE_TO_CTX, ref, ref_ctx)
    patch_type()(None, sequence_parallel=policy_ctx)
    patch_type()(None, sequence_parallel=ref_ctx)
    x = torch.zeros(1)
    assert policy(x) is policy_ctx
    assert ref(x) is ref_ctx
    assert policy(x) is policy_ctx
    assert plain(x)[0] == 'original'
    assert plain(x, sequence_parallel_context=ref_ctx) is ref_ctx


def test_router_replay_uses_model_owner_outside_forward(monkeypatch):
    from twinkle.model.transformers.moe.router_replay import RouterReplay as rr

    class Context:
        world_size = sp_world_size = 2
        real_position_ids = None

        def __init__(self, offset):
            self.offset = offset

        def gather(self, tensor, **kwargs):
            return tensor + self.offset

    models = [torch.nn.Linear(1, 1) for _ in range(3)]
    monkeypatch.setitem(sp._MODULE_TO_CTX, models[0], Context(10))
    monkeypatch.setitem(sp._MODULE_TO_CTX, models[1], Context(20))
    monkeypatch.setattr(rr, '_find_moe_blocks_with_names', lambda model: [('block', model)])
    monkeypatch.setattr(rr, '_registry', {'block': SimpleNamespace(recorded_indices=torch.zeros(2, 1))})
    assert sp._current_ctx() is None
    assert rr.get_router_replay_data(models[0]).eq(10).all()
    assert rr.get_router_replay_data(models[1]).eq(20).all()
    assert rr.get_router_replay_data(models[2]).eq(0).all()


@pytest.mark.parametrize('rank', [0, 1])
def test_router_replay_shards_mask_with_routing_indices(monkeypatch, rank):
    from twinkle.model.transformers.moe.router_replay import RouterReplay

    model = torch.nn.Linear(1, 1)
    ctx = sp.SequenceParallel()
    ctx.world_size = ctx.sp_world_size = 2
    ctx.rp_world_size = 1
    ctx._sp_rank = rank
    ctx.extra_kwargs['position_ids'] = torch.arange(5).unsqueeze(0)
    monkeypatch.setattr(RouterReplay, '_find_moe_blocks_with_names', lambda _: [('block', model)])
    state = SimpleNamespace(target_indices=None, target_mask=None)
    monkeypatch.setattr(RouterReplay, '_registry', {'block': state})
    mask = torch.tensor([[True, False, True, False, True]])
    routing = torch.arange(5).reshape(1, 5, 1, 1)
    from twinkle import Platform
    monkeypatch.setattr(Platform, 'get_local_device', lambda: torch.device('cpu'))
    RouterReplay.set_router_replay_data(routing, model, mask, sp_context=ctx)
    expected_indices = torch.tensor([0, 1, 2, 3, 4, 0]).chunk(2)[rank]
    expected_mask = torch.tensor([True, False, True, False, True, False]).chunk(2)[rank]
    torch.testing.assert_close(state.target_indices[:, 0], expected_indices)
    torch.testing.assert_close(state.target_mask, expected_mask)


@pytest.mark.parametrize('nested', [False, True])
def test_prepare_causal_mask_method_does_not_retain_weak_registry_owner(monkeypatch, nested):
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(num_attention_heads=2, _attn_implementation='sdpa')

        def _update_causal_mask(self, mask, *args):
            return mask + self.weight

    model = Backbone()
    if nested:
        wrapper = torch.nn.Module()
        wrapper.language_model = model
        wrapper.config = model.config
        model = wrapper
        del wrapper
    refs = [weakref.ref(module) for module in model.modules()]
    ctx = sp.SequenceParallel()
    monkeypatch.setattr(sp, 'get_llm_model', lambda value: value)
    monkeypatch.setattr(sp.SequenceParallel, '_global_inited', True)
    monkeypatch.setattr(sp.SequenceParallel, '_init_device_mesh', lambda *args: None)
    ctx.prepare(2, model, SimpleNamespace(pad_token_id=0))
    torch.testing.assert_close(ctx.causal_mask_func(torch.zeros(1)), torch.ones(1))
    del model
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert ctx.causal_mask_func is None
