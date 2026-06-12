# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for twinkle.cli configuration parser.

Covers:
- dataclass defaults (including newly added model + infra fields)
- CLISource argv parsing (--key val, --key=val, bool flags, --no_xxx)
- DotEnvSource and EnvVarSource
- YamlSource loading
- ConfigRegistry resolution: canonical, alias, prefix, primary disambiguation
- ValueCaster scalar/list coercion
- CLI.from_args resolution priority (CLI > YAML > env > .env > defaults)
- use_megatron shorthand mapping to ``strategy='native_fsdp'``
- Args accessor APIs (get_*_args, get/__getitem__, to_dict, extra)
"""
from __future__ import annotations

import os
import pytest
import textwrap
from dataclasses import fields as dc_fields
from pathlib import Path
from unittest import mock

from twinkle.cli import (CLI, Args, CLISource, ConfigResolver, DotEnvSource, EnvVarSource, ModelArgs, ValueCaster,
                         YamlSource)
from twinkle.cli.cli import ConfigRegistry


def _make_registry() -> ConfigRegistry:
    a = Args()
    groups = {f.name: getattr(a, f.name) for f in dc_fields(a) if f.name != 'extra'}
    return ConfigRegistry(groups)


# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────


class TestDefaults:

    def test_model_defaults(self):
        m = Args().model
        assert m.model_id is None
        assert m.mixed_precision == 'bf16'
        assert m.strategy == 'accelerate'
        assert m.gradient_checkpointing is True

    def test_model_new_megatron_fields(self):
        m = Args().model
        assert m.load_weights is True
        assert m.use_distributed_optimizer is True
        assert m.variable_seq_lengths is True
        assert m.recompute_granularity == 'full'
        assert m.recompute_method == 'uniform'
        assert m.recompute_num_layers == 1
        assert m.recompute_modules is None

    def test_infra_new_fields(self):
        i = Args().infra
        assert i.pp_size is None
        assert i.etp_size is None
        assert i.vpp_size is None
        assert i.ep_fsdp_size is None
        assert i.world_size is None
        assert i.sequence_parallel is False
        assert i.nproc_per_node == 8
        assert i.mode == 'local'

    def test_training_defaults(self):
        t = Args().training
        assert t.padding_free is False
        assert t.seed == 42
        assert t.max_steps == 200


# ──────────────────────────────────────────────────────────────────────────────
# CLISource
# ──────────────────────────────────────────────────────────────────────────────


class TestCLISource:

    def test_value_with_space(self):
        assert CLISource(['--model_id', 'Qwen/Qwen2.5']).load() == {'model_id': 'Qwen/Qwen2.5'}

    def test_value_with_equals(self):
        assert CLISource(['--lr=1e-4']).load() == {'lr': '1e-4'}

    def test_bool_flag(self):
        assert CLISource(['--padding_free']).load() == {'padding_free': True}

    def test_no_prefix_negation(self):
        assert CLISource(['--no_padding_free']).load() == {'padding_free': False}
        assert CLISource(['--no-padding_free']).load() == {'padding_free': False}

    def test_mixed(self):
        out = CLISource(['--model_id', 'X', '--lr=1e-3', '--padding_free', '--no_trust_remote_code']).load()
        assert out == {'model_id': 'X', 'lr': '1e-3', 'padding_free': True, 'trust_remote_code': False}


# ──────────────────────────────────────────────────────────────────────────────
# ValueCaster
# ──────────────────────────────────────────────────────────────────────────────


class TestValueCaster:

    @pytest.mark.parametrize(
        'raw,expected',
        [
            ('true', True),
            ('FALSE', False),
            ('yes', True),
            ('no', False),
            ('null', None),
            ('none', None),
            ('42', 42),
            ('3.14', 3.14),
            ('hello', 'hello'),
        ],
    )
    def test_auto_cast_scalars(self, raw, expected):
        assert ValueCaster.auto_cast(raw) == expected

    def test_auto_cast_csv_list(self):
        assert ValueCaster.auto_cast('q_proj,v_proj,o_proj') == ['q_proj', 'v_proj', 'o_proj']

    def test_auto_cast_passthrough_non_str(self):
        assert ValueCaster.auto_cast(7) == 7
        assert ValueCaster.auto_cast(True) is True

    def test_coerce_to_bool_field(self):
        m = ModelArgs()
        assert ValueCaster.coerce_to_field(m, 'gradient_checkpointing', 'false') is False
        assert ValueCaster.coerce_to_field(m, 'gradient_checkpointing', 'YES') is True

    def test_coerce_to_int_field(self):
        m = ModelArgs()
        assert ValueCaster.coerce_to_field(m, 'recompute_num_layers', '4') == 4


# ──────────────────────────────────────────────────────────────────────────────
# DotEnv + EnvVar
# ──────────────────────────────────────────────────────────────────────────────


class TestDotEnvSource:

    def test_load_simple(self, tmp_path: Path):
        f = tmp_path / '.env'
        f.write_text('MODEL_ID="Qwen/Qwen3"\nLR=1e-4\n# comment\n\nPADDING_FREE=true\n')
        loaded = DotEnvSource(f).load()
        assert loaded == {'MODEL_ID': 'Qwen/Qwen3', 'LR': '1e-4', 'PADDING_FREE': 'true'}

    def test_missing_file(self, tmp_path: Path):
        assert DotEnvSource(tmp_path / 'nope.env').load() == {}


class TestEnvVarSource:

    def test_twinkle_prefix_strip(self):
        registry = _make_registry()
        with mock.patch.dict(os.environ, {'TWINKLE_MODEL_ID': 'X', 'TWINKLE_LR': '5e-5'}, clear=False):
            out = EnvVarSource(registry).load()
        assert out['MODEL_ID'] == 'X'
        assert out['LR'] == '5e-5'

    def test_bare_known_key_picked_unknown_skipped(self, monkeypatch):
        monkeypatch.delenv('TWINKLE_MAX_STEPS', raising=False)
        registry = _make_registry()
        with mock.patch.dict(os.environ, {'MAX_STEPS': '99', 'TOTALLY_UNKNOWN_VAR_XYZ': '1'}, clear=False):
            out = EnvVarSource(registry).load()
        assert 'MAX_STEPS' in out
        assert 'TOTALLY_UNKNOWN_VAR_XYZ' not in out


# ──────────────────────────────────────────────────────────────────────────────
# YAML
# ──────────────────────────────────────────────────────────────────────────────


class TestYamlSource:

    def test_nested_yaml(self, tmp_path: Path):
        pytest.importorskip('omegaconf')
        cfg = tmp_path / 'config.yaml'
        cfg.write_text(
            textwrap.dedent("""
            model:
              model_id: Qwen/Qwen2.5-0.5B
              recompute_num_layers: 4
              variable_seq_lengths: false
            training:
              max_steps: 1000
              padding_free: true
            infra:
              tp_size: 2
              pp_size: 2
              vpp_size: 2
              sequence_parallel: true
            """).strip())
        loaded = YamlSource(cfg).load()
        assert loaded['model']['model_id'] == 'Qwen/Qwen2.5-0.5B'
        assert loaded['infra']['vpp_size'] == 2

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            YamlSource(tmp_path / 'nope.yaml').load()


# ──────────────────────────────────────────────────────────────────────────────
# ConfigRegistry / ConfigResolver
# ──────────────────────────────────────────────────────────────────────────────


class TestRegistry:

    def test_resolve_canonical_case_and_dash(self):
        r = _make_registry()
        assert r.resolve('model_id') == ('model', 'model_id')
        assert r.resolve('MODEL_ID') == ('model', 'model_id')
        assert r.resolve('model-id') == ('model', 'model_id')

    def test_resolve_aliases(self):
        r = _make_registry()
        assert r.resolve('lr') == ('optimizer', 'learning_rate')
        assert r.resolve('num_gpus') == ('infra', 'nproc_per_node')

    def test_resolve_unknown(self):
        assert _make_registry().resolve('totally_made_up') is None

    def test_primary_for_model_id(self):
        # model_id exists in ModelArgs (primary) and TemplateArgs.
        a = Args()
        ConfigResolver(a).apply({'model_id': 'X'}, cast_strings=True)
        assert a.model.model_id == 'X'
        assert a.template.model_id is None

    def test_primary_for_seed(self):
        # seed exists in TrainingArgs (primary) and SamplingArgs.
        a = Args()
        ConfigResolver(a).apply({'seed': '1234'}, cast_strings=True)
        assert a.training.seed == 1234
        assert a.sampling.seed is None


# ──────────────────────────────────────────────────────────────────────────────
# CLI.from_args end-to-end
# ──────────────────────────────────────────────────────────────────────────────


class TestCLIEndToEnd:

    @pytest.fixture(autouse=True)
    def _scrub_env(self, monkeypatch):
        # Strip ambient env vars that could pollute these tests.
        for k in list(os.environ):
            if k.startswith('TWINKLE_'):
                monkeypatch.delenv(k, raising=False)
        for k in ('MODEL_ID', 'LR', 'MAX_STEPS', 'PADDING_FREE'):
            monkeypatch.delenv(k, raising=False)

    def test_cli_argv_basic(self):
        args = CLI.from_args(argv=['--model_id', 'Qwen/Q', '--lr', '5e-4', '--max_steps', '500'])
        assert args.model.model_id == 'Qwen/Q'
        assert args.optimizer.learning_rate == pytest.approx(5e-4)
        assert args.training.max_steps == 500

    def test_cli_new_megatron_fields(self):
        args = CLI.from_args(argv=[
            '--recompute_granularity',
            'selective',
            '--recompute_num_layers',
            '8',
            '--no_variable_seq_lengths',
            '--no_use_distributed_optimizer',
            '--no_load_weights',
        ])
        assert args.model.recompute_granularity == 'selective'
        assert args.model.recompute_num_layers == 8
        assert args.model.variable_seq_lengths is False
        assert args.model.use_distributed_optimizer is False
        assert args.model.load_weights is False

    def test_cli_new_infra_fields(self):
        args = CLI.from_args(argv=[
            '--tp_size',
            '2',
            '--pp_size',
            '2',
            '--etp_size',
            '1',
            '--vpp_size',
            '4',
            '--ep_fsdp_size',
            '8',
            '--sequence_parallel',
            '--world_size',
            '32',
        ])
        assert args.infra.tp_size == 2
        assert args.infra.pp_size == 2
        assert args.infra.etp_size == 1
        assert args.infra.vpp_size == 4
        assert args.infra.ep_fsdp_size == 8
        assert args.infra.sequence_parallel is True
        assert args.infra.world_size == 32

    def test_use_megatron_true_flips_strategy(self):
        args = CLI.from_args(argv=['--use_megatron', 'true'])
        assert args.model.strategy == 'native_fsdp'

    def test_use_megatron_false_is_noop(self):
        args = CLI.from_args(argv=['--use_megatron', 'false'])
        assert args.model.strategy == 'accelerate'

    def test_priority_cli_beats_yaml_beats_env(self, tmp_path: Path):
        cfg = tmp_path / 'c.yaml'
        cfg.write_text('model:\n  model_id: from_yaml\ntraining:\n  max_steps: 100\n')
        env_overrides = {'TWINKLE_MAX_STEPS': '50', 'TWINKLE_MODEL_ID': 'from_env'}
        with mock.patch.dict(os.environ, env_overrides, clear=False):
            args = CLI.from_args(argv=['--config', str(cfg), '--model_id', 'from_cli'])
        # CLI wins for model_id; YAML wins for max_steps (no CLI override).
        assert args.model.model_id == 'from_cli'
        assert args.training.max_steps == 100

    def test_dotenv_lowest_priority(self, tmp_path: Path):
        env_file = tmp_path / '.env'
        env_file.write_text('MODEL_ID=from_dotenv\n')
        args = CLI.from_args(argv=[], env_file=env_file)
        assert args.model.model_id == 'from_dotenv'

    def test_yaml_via_explicit_param(self, tmp_path: Path):
        cfg = tmp_path / 'c.yaml'
        cfg.write_text('model:\n  recompute_num_layers: 16\ninfra:\n  vpp_size: 2\n')
        args = CLI.from_args(argv=[], config_file=cfg)
        assert args.model.recompute_num_layers == 16
        assert args.infra.vpp_size == 2

    def test_extra_dict_for_unknown_keys(self):
        args = CLI.from_args(argv=['--my_custom_flag', 'hello'])
        assert args.extra.get('my_custom_flag') == 'hello'

    def test_lora_target_modules_from_yaml(self, tmp_path: Path):
        cfg = tmp_path / 'c.yaml'
        cfg.write_text('lora:\n  use_lora: true\n  lora_target_modules: [q_proj, v_proj]\n')
        args = CLI.from_args(argv=[], config_file=cfg)
        assert args.lora.use_lora is True
        assert args.lora.lora_target_modules == ['q_proj', 'v_proj']


# ──────────────────────────────────────────────────────────────────────────────
# Args accessors
# ──────────────────────────────────────────────────────────────────────────────


class TestArgsAccessors:

    def test_get_model_args_falls_back_to_template(self):
        a = Args()
        a.template.model_id = 'TmplOnly'
        d = a.get_model_args()
        assert d['model_id'] == 'TmplOnly'

    def test_get_model_args_prefers_self(self):
        a = Args()
        a.model.model_id = 'M'
        a.template.model_id = 'T'
        assert a.get_model_args()['model_id'] == 'M'

    def test_get_optimizer_args_renames_lr(self):
        a = Args()
        a.optimizer.learning_rate = 7e-5
        d = a.get_optimizer_args()
        assert d['lr'] == pytest.approx(7e-5)
        assert 'learning_rate' not in d

    def test_get_lora_args_default_target(self):
        a = Args()
        d = a.get_lora_args()
        assert d['target_modules'] == 'all-linear'
        assert d['r'] == 8

    def test_get_dispatch(self):
        a = Args()
        a.training.max_steps = 7
        # `get` reaches into any group, including for new fields.
        assert a.get('max_steps') == 7
        assert a.get('vpp_size') is None
        assert a.get('does_not_exist', 'fallback') == 'fallback'

    def test_getitem(self):
        a = Args()
        a.model.recompute_num_layers = 3
        assert a['recompute_num_layers'] == 3
        with pytest.raises(KeyError):
            _ = a['absolutely_unknown_key_xyz']

    def test_to_dict_contains_new_fields(self):
        a = Args()
        d = a.to_dict()
        # New fields should show up in the dict export.
        assert 'recompute_granularity' in d
        assert 'use_distributed_optimizer' in d
        assert 'variable_seq_lengths' in d
        assert 'load_weights' in d
        assert 'sequence_parallel' in d
        assert 'pp_size' not in d  # None values are filtered by _to_dict
