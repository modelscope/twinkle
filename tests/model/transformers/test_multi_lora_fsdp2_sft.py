import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import twinkle
from peft import LoraConfig
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from twinkle import DeviceMesh
from twinkle.model.multi_lora import MultiLora
from twinkle.model.transformers.multi_lora_transformers import MultiLoraTransformersModel

TEST_MODEL_ID = os.environ.get('TEST_MODEL_ID')


def build_lora_config(r: int = 4) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=max(8, r * 2),
        target_modules='all-linear',
        init_lora_weights=False,
    )


def build_sft_batch():
    return [{'input_ids': [1, 3, 4, 2], 'labels': [1, 3, 4, 2]}]


def assert_same_lora_state(state_before, state_after):
    assert state_before.keys() == state_after.keys()
    for name, value in state_before.items():
        assert torch.equal(value, state_after[name]), name


def _write_tiny_model_dir(model_dir: Path) -> str:
    model_dir.mkdir(parents=True, exist_ok=True)

    config = LlamaConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    LlamaForCausalLM(config).save_pretrained(model_dir)

    vocab = {
        '<pad>': 0,
        '<bos>': 1,
        '<eos>': 2,
        'hello': 3,
        'world': 4,
        'adapter': 5,
        'state': 6,
        '<unk>': 7,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token='<unk>'))
    tokenizer.pre_tokenizer = Whitespace()
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token='<unk>',
        pad_token='<pad>',
        bos_token='<bos>',
        eos_token='<eos>',
    )
    fast_tokenizer.save_pretrained(model_dir)
    return str(model_dir)


@pytest.fixture
def model_path() -> str:
    if TEST_MODEL_ID:
        return TEST_MODEL_ID
    return _write_tiny_model_dir(make_workspace_temp_dir('tiny-llama'))


def make_workspace_temp_dir(prefix: str) -> Path:
    base_dir = Path.cwd() / '.codex_test_tmp'
    base_dir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f'{prefix}-', dir=base_dir))


def build_device_mesh(fsdp_size: int = 2) -> DeviceMesh:
    if torch.cuda.device_count() >= fsdp_size:
        return DeviceMesh.from_sizes(fsdp_size=fsdp_size, device_type='cuda')
    return DeviceMesh.from_sizes(world_size=1, dp_size=1, device_type='cpu')


def build_multi_lora_model(model_path: str, strategy: str, fsdp_size: int = 2):
    mesh = build_device_mesh(fsdp_size=fsdp_size)
    twinkle.initialize(mode='local', global_device_mesh=mesh)
    return MultiLoraTransformersModel(
        model_id=model_path,
        device_mesh=mesh,
        strategy=strategy,
    )


def build_multi_lora_state() -> MultiLora:
    config = LlamaConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    multi_lora = MultiLora(max_loras=2, max_r=8, max_length=32)
    multi_lora.patch(LlamaForCausalLM(config))
    multi_lora.save_initial_weights()
    multi_lora.acquire_lora('default', build_lora_config(r=4))
    return multi_lora


def test_multi_lora_accelerate_fsdp2_uses_device_mesh(model_path):
    model = build_multi_lora_model(model_path, strategy='accelerate')
    assert model.strategy.device_mesh is not None
    assert model._model_wrapped is False


def test_multi_lora_native_fsdp2_uses_lazy_wrap(model_path):
    model = build_multi_lora_model(model_path, strategy='native_fsdp')
    assert model.strategy.device_mesh is not None
    assert model._model_wrapped is False


def test_multi_lora_state_dict_round_trip_preserves_rank_slices():
    multi_lora = build_multi_lora_state()
    calls = []

    def fake_read(parameter):
        calls.append(tuple(parameter.shape))
        return parameter.detach().clone()

    multi_lora._read_param_tensor = fake_read
    state = multi_lora.get_state_dict('default')

    assert state
    assert all('.default.' not in key for key in state)
    assert calls


def test_multi_lora_set_state_dict_uses_tensor_write_helper():
    multi_lora = build_multi_lora_state()
    state = multi_lora.get_state_dict('default')
    calls = []

    def fake_write(parameter, value):
        calls.append((tuple(parameter.shape), tuple(value.shape)))
        parameter.data.copy_(value)

    multi_lora._write_param_tensor = fake_write
    multi_lora.set_state_dict('default', state)

    assert calls


def test_multi_lora_load_initial_weights_uses_tensor_write_helper():
    multi_lora = build_multi_lora_state()
    calls = []

    def fake_write(parameter, value):
        calls.append((tuple(parameter.shape), tuple(value.shape)))
        parameter.data.copy_(value)

    multi_lora._write_param_tensor = fake_write
    multi_lora._load_initial_weights('lora_0')

    assert calls


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Requires 2+ GPUs')
def test_multi_lora_accelerate_fsdp2_sft_round_trip(model_path):
    output_dir = make_workspace_temp_dir('accelerate-ckpt')
    model = build_multi_lora_model(model_path, strategy='accelerate')
    model.add_adapter_to_model('default', build_lora_config(), gradient_accumulation_steps=1)
    model.set_loss('CrossEntropyLoss', adapter_name='default')
    model.set_optimizer('AdamW', lr=1e-4, adapter_name='default')

    batch = build_sft_batch()
    model.forward_backward(inputs=batch, adapter_name='default')
    model.clip_grad_and_step(adapter_name='default')

    state_before = model.get_state_dict(adapter_name='default')
    model.save('ckpt', output_dir=str(output_dir), adapter_name='default')
    model.remove_adapter('default')
    model.add_adapter_to_model('default', build_lora_config(), gradient_accumulation_steps=1)
    model.load('ckpt', output_dir=str(output_dir), adapter_name='default')
    state_after = model.get_state_dict(adapter_name='default')

    assert_same_lora_state(state_before, state_after)
    shutil.rmtree(output_dir, ignore_errors=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Requires 2+ GPUs')
def test_multi_lora_native_fsdp_sft_round_trip(model_path):
    output_dir = make_workspace_temp_dir('native-ckpt')
    model = build_multi_lora_model(model_path, strategy='native_fsdp')
    model.add_adapter_to_model('default', build_lora_config(), gradient_accumulation_steps=1)
    model.set_loss('CrossEntropyLoss', adapter_name='default')
    model.set_optimizer('AdamW', lr=1e-4, adapter_name='default')

    model.forward_backward(inputs=build_sft_batch(), adapter_name='default')
    model.clip_grad_and_step(adapter_name='default')

    state_before = model.get_state_dict(adapter_name='default')
    model.save('native-ckpt', output_dir=str(output_dir), adapter_name='default')
    model.remove_adapter('default')
    model.add_adapter_to_model('default', build_lora_config(), gradient_accumulation_steps=1)
    model.load('native-ckpt', output_dir=str(output_dir), adapter_name='default')
    state_after = model.get_state_dict(adapter_name='default')

    assert_same_lora_state(state_before, state_after)
    shutil.rmtree(output_dir, ignore_errors=True)
