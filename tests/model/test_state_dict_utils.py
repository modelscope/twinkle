# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for twinkle.model.state_dict_utils (adapter-key semantics)."""
import torch
import torch.nn as nn

from twinkle.model.state_dict_utils import (
    collect_local_state,
    is_lora_state_key,
    is_saveable_adapter_key,
    normalize_adapter_keys,
    remap_optimizer_state_names,
    with_tied_weight_aliases,
)


class TestAdapterKeyPredicates:

    def test_lora_state_key_matches_all_three_markers(self):
        assert is_lora_state_key('model.layers.0.lora_A.weight')
        assert is_lora_state_key('model.layers.0.lora_B.weight')
        assert is_lora_state_key('model.lora_embedding_A')

    def test_lora_state_key_is_substring_semantics(self):
        # Marker matching is substring-based (legacy swift semantics): a name merely
        # containing the marker counts, even without dot boundaries.
        assert is_lora_state_key('xxlora_Ayy')

    def test_lora_state_key_rejects_plain_parameter(self):
        assert not is_lora_state_key('model.layers.0.mlp.q_proj.weight')
        assert not is_lora_state_key('')

    def test_saveable_adapter_key_includes_modules_to_save_segment(self):
        assert is_saveable_adapter_key('model.layers.0.mlp.modules_to_save.default.q_proj.weight')

    def test_saveable_adapter_key_requires_dotted_segment(self):
        # MODULES_TO_SAVE_SEGMENT is '.modules_to_save.'; a token merely containing
        # the words without both dot boundaries must not be treated as saveable.
        assert not is_saveable_adapter_key('modules_to_save.weight')
        assert not is_saveable_adapter_key('model.my_modules_to_save.weight')

    def test_saveable_adapter_key_rejects_plain_parameter(self):
        assert not is_saveable_adapter_key('model.layers.0.mlp.q_proj.weight')


class TestCollectLocalState:

    class _Nested(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2, bias=False)
            self.register_buffer('persist', torch.ones(3))
            self.register_buffer('vanish', torch.zeros(3), persistent=False)
            self.child = nn.Identity()
            self.child.register_buffer('persist', torch.full((2, ), 2.0))

    def test_parameters_only_by_default(self):
        owner = self._Nested()
        state = collect_local_state(owner)
        assert set(state) == {'linear.weight'}
        assert state['linear.weight'] is owner.linear.weight

    def test_include_buffers_adds_persistent_only(self):
        owner = self._Nested()
        state = collect_local_state(owner, include_buffers=True)
        assert set(state) == {'linear.weight', 'persist', 'child.persist'}
        assert state['persist'] is owner._buffers['persist']
        assert state['child.persist'] is owner.child._buffers['persist']

    def test_non_persistent_buffer_is_never_collected(self):
        owner = self._Nested()
        for include_buffers in (False, True):
            assert 'vanish' not in collect_local_state(owner, include_buffers=include_buffers)


class TestNormalizeAdapterKeys:

    def test_modules_to_save_segment_collapses(self):
        state = {'model.layers.0.mlp.modules_to_save.default.q_proj.weight': 1}
        assert normalize_adapter_keys(state, 'default') == {'model.layers.0.mlp.q_proj.weight': 1}

    def test_plain_adapter_segment_collapses(self):
        state = {'model.layers.0.mlp.q_proj.lora_A.default.weight': 1}
        assert normalize_adapter_keys(state, 'default') == {'model.layers.0.mlp.q_proj.lora_A.weight': 1}

    def test_keys_without_wrapper_segments_are_untouched(self):
        state = {'model.layers.0.mlp.q_proj.weight': 1}
        assert normalize_adapter_keys(state, 'default') == state

    def test_only_the_named_adapter_segment_is_stripped(self):
        # A modules_to_save segment written under a different adapter name is left
        # alone: the strip is scoped to the requested adapter.
        state = {'a.modules_to_save.other.b': 1}
        assert normalize_adapter_keys(state, 'default') == {'a.modules_to_save.other.b': 1}

    def test_replace_hits_every_occurrence(self):
        state = {'x.default.a.default.b': 1}
        assert normalize_adapter_keys(state, 'default') == {'x.a.b': 1}


class TestRemapOptimizerStateNames:

    def test_empty_mapping_returns_early_without_touching_state(self):
        state_dict = {'state': {'old.x': 1}, 'param_groups': [{'params': [0], 'param_names': ['old.x']}]}
        before = {k: dict(v) if isinstance(v, dict) else list(v) for k, v in state_dict.items()}
        remap_optimizer_state_names(state_dict, {})
        assert state_dict == before

    def test_string_state_keys_remap_and_numeric_ids_survive(self):
        state_dict = {'state': {'old.x': 'v1', 7: 'v2'}}
        remap_optimizer_state_names(state_dict, {'old.x': 'new.x'})
        assert state_dict['state'] == {'new.x': 'v1', 7: 'v2'}

    def test_param_groups_remap_names_but_keep_numeric_params(self):
        state_dict = {'param_groups': [{'params': [0, 1], 'param_names': ['old.a', 'keep.b']}]}
        remap_optimizer_state_names(state_dict, {'old.a': 'new.a'})
        assert state_dict['param_groups'][0]['params'] == [0, 1]
        assert state_dict['param_groups'][0]['param_names'] == ['new.a', 'keep.b']

    def test_missing_state_key_is_tolerated(self):
        state_dict = {'param_groups': [{'params': [0]}]}
        remap_optimizer_state_names(state_dict, {'old.a': 'new.a'})
        assert state_dict == {'param_groups': [{'params': [0]}]}


class TestWithTiedWeightAliases:

    class _Tied(nn.Module):

        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(4, 3)
            self.head = nn.Linear(3, 4, bias=False)
            self.head.weight = self.embed_tokens.weight

    def test_tied_alias_is_expanded_from_the_present_source(self):
        model = self._Tied()
        state_dict = {'embed_tokens.weight': model.embed_tokens.weight}
        expanded = with_tied_weight_aliases(model, state_dict)
        assert set(expanded) == {'embed_tokens.weight', 'head.weight'}
        assert expanded['head.weight'] is model.embed_tokens.weight

    def test_untied_parameter_gains_no_alias(self):
        model = self._Tied()
        model.other = nn.Linear(3, 3, bias=False)
        state_dict = {'embed_tokens.weight': model.embed_tokens.weight}
        expanded = with_tied_weight_aliases(model, state_dict)
        assert 'other.weight' not in expanded

    def test_alias_group_without_a_present_source_is_skipped(self):
        model = self._Tied()
        state_dict = {'other.weight': torch.zeros(1)}
        expanded = with_tied_weight_aliases(model, state_dict)
        assert 'head.weight' not in expanded

    def test_existing_entries_are_not_overwritten(self):
        model = self._Tied()
        replacement = torch.zeros(3, 4)
        state_dict = {'embed_tokens.weight': replacement, 'head.weight': replacement}
        expanded = with_tied_weight_aliases(model, state_dict)
        assert expanded['head.weight'] is replacement
        assert expanded['embed_tokens.weight'] is replacement

    def test_triple_tie_expands_every_name(self):
        model = nn.Module()
        shared = nn.Parameter(torch.randn(2, 2))
        first = nn.Linear(2, 2, bias=False)
        second = nn.Linear(2, 2, bias=False)
        third = nn.Linear(2, 2, bias=False)
        for module, name in ((first, 'first'), (second, 'second'), (third, 'third')):
            module.weight = shared
            setattr(model, name, module)
        state_dict = {'second.weight': shared}
        expanded = with_tied_weight_aliases(model, state_dict)
        assert set(expanded) == {'first.weight', 'second.weight', 'third.weight'}
        assert expanded['first.weight'] is shared
        assert expanded['third.weight'] is shared
