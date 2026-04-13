from contextlib import nullcontext
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
from twinkle.model.transformers.transformers import TransformersModel

TEST_MODEL_ID = os.environ.get('TEST_MODEL_ID')


def build_lora_config(r: int = 4) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=max(8, r * 2),
        target_modules='all-linear',
        init_lora_weights=False,
    )


def build_sft_batch():
    return [{
        'input_ids': [1, 3, 4, 2],
        'attention_mask': [1, 1, 1, 1],
        'position_ids': [0, 1, 2, 3],
        'labels': [1, 3, 4, 2],
    }]


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


def _npu_device_count() -> int:
    npu = getattr(torch, 'npu', None)
    if npu is None:
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            return 0
        npu = getattr(torch, 'npu', None)

    if npu is None:
        return 0

    try:
        if npu.is_available():
            return npu.device_count()
    except Exception:
        return 0
    return 0


def accelerator_device_count() -> int:
    return max(torch.cuda.device_count(), _npu_device_count())


def build_device_mesh(fsdp_size: int = 2) -> DeviceMesh:
    if torch.cuda.device_count() >= fsdp_size:
        return DeviceMesh.from_sizes(fsdp_size=fsdp_size, device_type='cuda')
    if _npu_device_count() >= fsdp_size:
        return DeviceMesh.from_sizes(fsdp_size=fsdp_size, device_type='npu')
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


def test_build_device_mesh_prefers_npu_when_available(monkeypatch):
    class FakeNPU:

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 0)
    monkeypatch.setattr(torch, 'npu', FakeNPU(), raising=False)

    mesh = build_device_mesh(fsdp_size=2)

    assert mesh.device_type == 'npu'
    assert mesh.fsdp_world_size == 2


def test_accelerator_device_count_uses_npu_when_cuda_absent(monkeypatch):
    class FakeNPU:

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 0)
    monkeypatch.setattr(torch, 'npu', FakeNPU(), raising=False)

    assert accelerator_device_count() == 2


def test_build_sft_batch_includes_processor_fields():
    batch = build_sft_batch()

    assert batch == [{
        'input_ids': [1, 3, 4, 2],
        'attention_mask': [1, 1, 1, 1],
        'position_ids': [0, 1, 2, 3],
        'labels': [1, 3, 4, 2],
    }]


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


def test_multi_lora_write_param_tensor_distributes_leaf_tensor(monkeypatch):
    multi_lora = MultiLora(max_loras=1, max_r=4, max_length=8)
    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    parameter.device_mesh = object()
    parameter.placements = ('shard0', )

    recorded = {}

    def fake_distribute_tensor(value, device_mesh, placements):
        if not value.is_leaf:
            raise RuntimeError('`distribute_tensor` should be used to distribute leaf tensors!')
        recorded['is_leaf'] = value.is_leaf
        recorded['device_mesh'] = device_mesh
        recorded['placements'] = placements
        return value

    monkeypatch.setattr('torch.distributed.tensor.distribute_tensor', fake_distribute_tensor)

    source = torch.ones(2, 2, requires_grad=True) * 3

    multi_lora._write_param_tensor(parameter, source)

    assert recorded['is_leaf'] is True
    assert recorded['device_mesh'] is parameter.device_mesh
    assert recorded['placements'] == parameter.placements
    assert torch.equal(parameter.data, torch.full((2, 2), 3.0))


def _build_stub_multi_lora_model():
    model = object.__new__(MultiLoraTransformersModel)
    model._check_adapter_valid = lambda adapter_name: None
    model.multi_adapter = type(
        'DummyMultiAdapter',
        (),
        {
            'save_context': staticmethod(lambda adapter_name: nullcontext()),
            'set_state_dict': lambda self, adapter_name, state_dict: None,
        },
    )()
    model.strategy = type('DummyStrategy', (), {'unwrap_model': lambda self, wrapped: wrapped})()
    model.model = object()
    model._load_optimizer = lambda checkpoint_dir, adapter_name=None: None
    return model


def test_multi_lora_save_barriers_after_checkpoint_write(monkeypatch):
    model = _build_stub_multi_lora_model()
    events = []

    def fake_save(self, name, output_dir=None, interval=1, **kwargs):
        events.append('save')
        return 'ckpt'

    monkeypatch.setattr(TransformersModel, 'save', fake_save)
    monkeypatch.setattr('torch.distributed.is_initialized', lambda: True)
    monkeypatch.setattr('torch.distributed.barrier', lambda: events.append('barrier'))

    checkpoint_dir = model.save('ckpt', output_dir='output', adapter_name='default')

    assert checkpoint_dir == 'ckpt'
    assert events == ['save', 'barrier']


def test_multi_lora_load_barriers_after_adapter_restore(monkeypatch):
    model = _build_stub_multi_lora_model()
    events = []

    class FakePeftModel:
        pass

    fake_peft_model = FakePeftModel()
    model.model = fake_peft_model

    def fake_set_state_dict(adapter_name, state_dict):
        events.append(('set_state_dict', adapter_name, state_dict))

    model.multi_adapter.set_state_dict = fake_set_state_dict

    monkeypatch.setattr('twinkle.model.transformers.multi_lora_transformers.PeftModel', FakePeftModel)
    monkeypatch.setattr('twinkle.model.transformers.multi_lora_transformers.load_peft_weights',
                        lambda checkpoint_dir, device='cpu': events.append(('load_peft_weights', checkpoint_dir,
                                                                            device)) or {'layer.weight': torch.ones(1)})
    monkeypatch.setattr('torch.distributed.is_initialized', lambda: True)
    monkeypatch.setattr('torch.distributed.barrier', lambda: events.append('barrier'))

    model.load('ckpt', output_dir='output', adapter_name='default')

    assert events == [
        ('load_peft_weights', os.path.join('output', 'ckpt'), 'cpu'),
        ('set_state_dict', 'default', {'layer.weight': torch.ones(1)}),
        'barrier',
    ]


@pytest.mark.skipif(accelerator_device_count() < 2, reason='Requires 2+ CUDA GPUs or NPUs')
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


@pytest.mark.skipif(accelerator_device_count() < 2, reason='Requires 2+ CUDA GPUs or NPUs')
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
