# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for twinkle.model.transformers.target_modules (LoRA target expansion)."""
import re

import pytest
import torch
import torch.nn as nn

from twinkle.model.transformers.target_modules import (
    find_all_linears,
    find_embedding,
    find_layers,
    find_norm,
    get_multimodal_target_regex,
)


class _FusedLinear(nn.Module):
    """A remote-code style fused linear: matched by the class-name heuristic, not isinstance."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 2))


class _RMSNormLike(nn.Module):
    """A remote-code style RMSNorm: matched by class name, not nn.LayerNorm identity."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))


class _Plain(nn.Module):
    """A non-linear container leaf used to force name disambiguation."""

    def __init__(self):
        super().__init__()


class TestFindLayers:

    def test_leaf_names_without_conflict_stay_bare(self):
        model = nn.Module()
        model.only = nn.Linear(2, 2)
        assert find_layers(model, lambda name, module: isinstance(module, nn.Linear)) == ['only']

    def test_conflicting_inner_node_forces_parent_disambiguation(self):
        # 'b.proj' is a non-linear module, so it lands in inner_nodes; the linear leaf
        # 'a.proj' must then be lengthened from bare 'proj' to stay unique for PEFT's
        # endswith matching.
        model = nn.Module()
        model.a = nn.Module()
        model.a.proj = nn.Linear(2, 2)
        model.b = nn.Module()
        model.b.proj = _Plain()
        assert find_layers(model, lambda name, module: isinstance(module, nn.Linear)) == ['a.proj']

    def test_min_name_len_forces_extra_parent_segments(self):
        model = nn.Module()
        model.a = nn.Module()
        model.a.proj = nn.Linear(2, 2)
        assert find_layers(
            model, lambda name, module: isinstance(module, nn.Linear), min_name_len=2) == ['a.proj']
        # Without min_name_len the same leaf resolves to its bare name.
        assert find_layers(model, lambda name, module: isinstance(module, nn.Linear)) == ['proj']

    def test_top_level_leaf_with_min_name_len_does_not_raise(self):
        # A single-segment leaf cannot be lengthened any further; min_name_len
        # must degrade to "as long as possible" instead of popping an empty list.
        model = nn.Module()
        model.only = nn.Linear(2, 2)
        assert find_layers(
            model, lambda name, module: isinstance(module, nn.Linear), min_name_len=2) == ['only']

    def test_sub_module_scopes_search_and_names_stay_relative(self):
        model = nn.Module()
        model.outer = nn.Module()
        model.outer.inner = nn.Module()
        model.outer.inner.proj = nn.Linear(2, 2)
        model.elsewhere = nn.Linear(2, 2)
        assert find_layers(
            model, lambda name, module: isinstance(module, nn.Linear), sub_module='outer.inner') == ['proj']


class TestFindAllLinears:

    def _model(self):
        model = nn.Module()
        language = nn.Module()
        mlp = nn.Module()
        mlp.q_proj = nn.Linear(2, 2)
        mlp.gate_proj = nn.Linear(2, 2)
        layer = nn.Module()
        layer.mlp = mlp
        language.layer = layer
        language.lm_head = nn.Linear(2, 2)
        language.score = nn.Linear(2, 2)
        language.v_head = nn.Linear(2, 2)
        language.classifier = nn.Linear(2, 2)
        language.fused = _FusedLinear()
        model.language = language
        return model

    def test_returns_linear_leaves_and_class_name_matches(self):
        found = set(find_all_linears(self._model()))
        assert found == {'q_proj', 'gate_proj', 'fused'}

    def test_readout_and_head_names_are_excluded(self):
        found = set(find_all_linears(self._model()))
        for excluded in ('lm_head', 'score', 'v_head', 'classifier'):
            assert excluded not in found

    def test_dotted_lm_head_argument_excludes_by_last_segment(self):
        # A loader declaring the head under a path (e.g. 'language.lm_head') still
        # excludes it -- only the final segment is the ignore token.
        found = set(find_all_linears(self._model(), lm_head='language.lm_head'))
        assert 'lm_head' not in found

    def test_sub_module_limits_the_walk(self):
        found = set(find_all_linears(self._model(), sub_module='language.layer'))
        assert found == {'q_proj', 'gate_proj'}


class TestFindNormAndEmbedding:

    def _model(self):
        model = nn.Module()
        model.ln = nn.LayerNorm(2)
        model.rms = _RMSNormLike()
        model.embed = nn.Embedding(4, 2)
        model.linear = nn.Linear(2, 2)
        return model

    def test_find_norm_matches_layernorm_and_rmsnorm_class_name(self):
        assert set(find_norm(self._model())) == {'ln', 'rms'}

    def test_find_embedding_matches_only_embedding(self):
        assert find_embedding(self._model()) == ['embed']


class _Merger(nn.Module):

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(2, 2)


class _VisualTower(nn.Module):

    def __init__(self):
        super().__init__()
        blocks = nn.ModuleList()
        block = nn.Module()
        block.q_proj = nn.Linear(2, 2)
        blocks.append(block)
        self.blocks = blocks
        self.merger = _Merger()
        self.embed_tokens = nn.Embedding(4, 2)


class _LanguageTower(nn.Module):

    def __init__(self):
        super().__init__()
        layers = nn.ModuleList()
        for _ in range(2):
            layer = nn.Module()
            layer.mlp = nn.Module()
            layer.mlp.q_proj = nn.Linear(2, 2)
            layers.append(layer)
        self.layers = layers
        self.lm_head = nn.Linear(2, 2)


class _MultiModal(nn.Module):

    def __init__(self):
        super().__init__()
        self.language = _LanguageTower()
        self.visual = _VisualTower()


def _model_arch():
    return {
        'language_model': ['language'],
        'vision_tower': ['visual'],
        'aligner': ['visual.merger'],
        'lm_head': 'language.lm_head',
    }


class TestGetMultimodalTargetRegex:

    def test_all_towers_frozen_raises(self):
        with pytest.raises(ValueError, match='no tower left to adapt'):
            get_multimodal_target_regex(
                _MultiModal(), _model_arch(), freeze_llm=True, freeze_vit=True, freeze_aligner=True)

    def test_default_partition_targets_only_the_language_tower(self):
        pattern = get_multimodal_target_regex(_MultiModal(), _model_arch())
        assert re.fullmatch(pattern, 'language.layers.0.mlp.q_proj')
        assert re.fullmatch(pattern, 'language.layers.1.mlp.q_proj')
        assert not re.fullmatch(pattern, 'language.lm_head')
        assert not re.fullmatch(pattern, 'visual.blocks.0.q_proj')

    def test_vision_tower_regex_excludes_nested_aligner(self):
        # With only the vision tower unfrozen, the aligner prefix (visual.merger) must
        # be claimed by a negative lookahead so the vit alternative cannot adapt it.
        pattern = get_multimodal_target_regex(
            _MultiModal(), _model_arch(), freeze_llm=True, freeze_vit=False, freeze_aligner=True)
        assert re.fullmatch(pattern, 'visual.blocks.0.q_proj')
        assert not re.fullmatch(pattern, 'visual.merger.q_proj')
        assert not re.fullmatch(pattern, 'language.layers.0.mlp.q_proj')

    def test_unfrozen_aligner_gets_its_own_alternative(self):
        pattern = get_multimodal_target_regex(
            _MultiModal(), _model_arch(), freeze_llm=True, freeze_vit=True, freeze_aligner=False)
        assert re.fullmatch(pattern, 'visual.merger.q_proj')
        assert not re.fullmatch(pattern, 'visual.blocks.0.q_proj')

    def test_head_only_tower_raises_no_linear(self):
        # A tower entry that IS the output head contributes nothing; if it is the only
        # unfrozen tower the expansion must fail loudly instead of adapting the head.
        arch = _model_arch()
        arch['language_model'] = ['language.lm_head']
        with pytest.raises(ValueError, match='found no linear layer'):
            get_multimodal_target_regex(_MultiModal(), arch, freeze_vit=True, freeze_aligner=True)

    def test_include_embedding_pulls_embedding_leaves_into_the_regex(self):
        pattern = get_multimodal_target_regex(
            _MultiModal(), _model_arch(),
            freeze_llm=True,
            freeze_vit=False,
            freeze_aligner=True,
            include_embedding=True)
        assert re.fullmatch(pattern, 'visual.embed_tokens')
        assert not re.fullmatch(pattern, 'language.embed_tokens')

    def test_tower_resolving_to_none_is_warned_and_skipped(self):
        arch = _model_arch()
        arch['language_model'] = ['ghost', 'language']
        # The ghost tower resolves to None and is skipped; the real tower still
        # produces a working regex rather than failing the whole expansion.
        pattern = get_multimodal_target_regex(_MultiModal(), arch)
        assert re.fullmatch(pattern, 'language.layers.0.mlp.q_proj')
        assert not re.fullmatch(pattern, 'ghost.anything')
