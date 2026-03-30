import concurrent.futures
import sys
import types
import uuid
from pathlib import Path
from types import ModuleType

import pytest
from peft import LoraConfig
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast


class _NoOpProcessPoolExecutor:
    def __init__(self, *args, **kwargs):
        pass

    def submit(self, fn, *args, **kwargs):
        raise RuntimeError('Process pool is disabled in this test environment.')


concurrent.futures.ProcessPoolExecutor = _NoOpProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if 'zmq' not in sys.modules:
    fake_zmq = ModuleType('zmq')
    fake_zmq.Socket = object
    fake_zmq.Context = object
    fake_zmq.REP = 0
    fake_zmq.REQ = 1
    fake_zmq.IPV6 = 2
    fake_zmq.error = types.SimpleNamespace(Again=RuntimeError)
    sys.modules['zmq'] = fake_zmq

from twinkle.model.transformers import MultiLoraTransformersModel, TransformersModel


def build_tiny_tokenizer():
    vocab = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2, '[UNK]': 3, 'hello': 4, 'world': 5}
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token='[PAD]',
        bos_token='[BOS]',
        eos_token='[EOS]',
        unk_token='[UNK]',
    )
    return tokenizer


@pytest.fixture
def tmp_path():
    root = Path(__file__).parent / 'tmp_test_checkpoints'
    root.mkdir(exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir()
    return path


@pytest.fixture
def tiny_local_model_dir(tmp_path):
    model_dir = tmp_path / 'tiny-qwen3'
    model_dir.mkdir()

    tokenizer = build_tiny_tokenizer()
    tokenizer.save_pretrained(model_dir)

    config = Qwen3Config(
        vocab_size=tokenizer.vocab_size,
        num_hidden_layers=1,
        num_attention_heads=1,
        hidden_size=16,
        max_position_embeddings=32,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    Qwen3ForCausalLM(config).save_pretrained(model_dir)
    return model_dir


def build_full_param_model(model_dir):
    return TransformersModel(
        model_cls='Qwen3ForCausalLM',
        model_id=str(model_dir),
        mixed_precision='no',
        grad_scaler_config={},
    )


def build_multi_lora_model(model_dir):
    device_mesh = types.SimpleNamespace(fsdp_world_size=0, data_world_size=1)
    model = MultiLoraTransformersModel(
        model_cls='Qwen3ForCausalLM',
        model_id=str(model_dir),
        mixed_precision='no',
        device_mesh=device_mesh,
        grad_scaler_config={},
    )
    model.add_adapter_to_model('default', LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']))
    return model


def prepare_full_param_checkpoint(tmp_path, model_dir, cur_step=3, consumed_train_samples=6):
    model = build_full_param_model(model_dir)
    model.set_optimizer('AdamW', lr=1e-4)
    model.set_lr_scheduler('LinearLR')
    model.set_grad_scaler(device='cpu')
    model.optimizer_group[''].cur_step = cur_step
    return Path(
        model.save(
            name='full-resume',
            output_dir=str(tmp_path),
            save_optimizer=True,
            consumed_train_samples=consumed_train_samples,
        ))


def prepare_lora_checkpoint(tmp_path, model_dir, cur_step=3, consumed_train_samples=6):
    model = build_multi_lora_model(model_dir)
    model.set_optimizer('AdamW', adapter_name='default', lr=1e-4)
    model.set_lr_scheduler('LinearLR', adapter_name='default')
    model.set_grad_scaler(adapter_name='default', device='cpu')
    model.optimizer_group['default'].cur_step = cur_step
    return Path(
        model.save(
            name='lora-resume',
            output_dir=str(tmp_path),
            adapter_name='default',
            save_optimizer=True,
            consumed_train_samples=consumed_train_samples,
        ))


def _ensure_file_exists(path: Path):
    if path.exists():
        return
    if path.name == 'trainer_state.json':
        path.write_text('{"cur_step": 3}', encoding='utf-8')
    else:
        path.write_bytes(b'placeholder')


def test_save_training_state_writes_split_files(tmp_path, tiny_local_model_dir):
    model = build_full_param_model(tiny_local_model_dir)
    model.set_optimizer('AdamW', lr=1e-4)
    model.set_lr_scheduler('LinearLR')
    model.set_grad_scaler(device='cpu')
    model.optimizer_group[''].cur_step = 7

    ckpt_dir = Path(model.save(name='resume-step', output_dir=str(tmp_path), save_optimizer=True))

    assert (ckpt_dir / 'optimizer.pt').exists()
    assert (ckpt_dir / 'scheduler.pt').exists()
    assert (ckpt_dir / 'scaler.pt').exists()
    assert (ckpt_dir / 'trainer_state.json').exists()
    assert (ckpt_dir / 'rng_state.pt').exists()


@pytest.mark.parametrize(
    'missing_name, expected_pattern',
    [
        ('optimizer.pt', 'optimizer.pt'),
        ('scheduler.pt', 'scheduler.pt'),
        ('scaler.pt', 'scaler.pt'),
        ('rng_state.pt', 'rng_state.pt'),
        ('trainer_state.json', 'trainer_state.json'),
    ],
)
def test_load_training_state_fails_when_required_file_missing(tmp_path, tiny_local_model_dir, missing_name,
                                                             expected_pattern):
    ckpt_dir = prepare_full_param_checkpoint(tmp_path, tiny_local_model_dir)
    _ensure_file_exists(ckpt_dir / missing_name)
    (ckpt_dir / missing_name).unlink()

    restored = build_full_param_model(ckpt_dir)
    restored.set_optimizer('AdamW', lr=1e-4)
    restored.set_lr_scheduler('LinearLR')
    restored.set_grad_scaler(device='cpu')

    with pytest.raises(FileNotFoundError, match=expected_pattern):
        restored.load_training_state(str(ckpt_dir))


def test_load_training_state_fails_for_malformed_trainer_state(tmp_path, tiny_local_model_dir):
    ckpt_dir = prepare_full_param_checkpoint(tmp_path, tiny_local_model_dir)
    (ckpt_dir / 'trainer_state.json').write_text('{"cur_step": 3}', encoding='utf-8')

    restored = build_full_param_model(ckpt_dir)
    restored.set_optimizer('AdamW', lr=1e-4)
    restored.set_lr_scheduler('LinearLR')
    restored.set_grad_scaler(device='cpu')

    with pytest.raises((KeyError, ValueError), match='gradient_accumulation_steps|consumed_train_samples'):
        restored.load_training_state(str(ckpt_dir))


def test_full_parameter_resume_restores_training_state_after_init(tmp_path, tiny_local_model_dir):
    ckpt_dir = prepare_full_param_checkpoint(tmp_path, tiny_local_model_dir, cur_step=9, consumed_train_samples=18)
    saved = build_full_param_model(tiny_local_model_dir)
    saved.set_optimizer('AdamW', lr=1e-4)
    saved.set_lr_scheduler('LinearLR')
    saved.set_grad_scaler(device='cpu')
    saved.optimizer_group[''].cur_step = 9
    saved.optimizer_group[''].gradient_accumulation_steps = 4
    ckpt_dir = Path(
        saved.save(
            name='full-resume',
            output_dir=str(tmp_path),
            save_optimizer=True,
            consumed_train_samples=18,
        ))
    restored = build_full_param_model(ckpt_dir)
    restored.set_optimizer('AdamW', lr=1e-4)
    restored.set_lr_scheduler('LinearLR')
    restored.set_grad_scaler(device='cpu')

    trainer_state = restored.load_training_state(str(ckpt_dir))

    assert trainer_state['cur_step'] == 9
    assert trainer_state['gradient_accumulation_steps'] == 4
    assert trainer_state['consumed_train_samples'] == 18
    assert restored.optimizer_group[''].cur_step == 9
    assert restored.optimizer_group[''].gradient_accumulation_steps == 4


def test_lora_load_does_not_restore_training_state_without_explicit_resume(tmp_path, tiny_local_model_dir):
    saved = build_multi_lora_model(tiny_local_model_dir)
    saved.set_optimizer('AdamW', adapter_name='default', lr=1e-4)
    saved.set_lr_scheduler('LinearLR', adapter_name='default')
    saved.set_grad_scaler(adapter_name='default', device='cpu')
    saved.optimizer_group['default'].cur_step = 5
    saved.optimizer_group['default'].gradient_accumulation_steps = 3
    ckpt_dir = Path(
        saved.save(
            name='lora-resume',
            output_dir=str(tmp_path),
            adapter_name='default',
            save_optimizer=True,
            consumed_train_samples=10,
        ))
    restored = build_multi_lora_model(tiny_local_model_dir)

    assert restored.optimizer_group['default'].cur_step == 0
    assert restored.optimizer_group['default'].gradient_accumulation_steps == 1

    restored.load(name=ckpt_dir.name, output_dir=str(ckpt_dir.parent), adapter_name='default')
    assert restored.optimizer_group['default'].cur_step == 0
    assert restored.optimizer_group['default'].gradient_accumulation_steps == 1

    restored.set_optimizer('AdamW', adapter_name='default', lr=1e-4)
    restored.set_lr_scheduler('LinearLR', adapter_name='default')
    restored.set_grad_scaler(adapter_name='default', device='cpu')

    trainer_state = restored.load_training_state(str(ckpt_dir), adapter_name='default')

    assert trainer_state['cur_step'] == 5
    assert trainer_state['gradient_accumulation_steps'] == 3
    assert trainer_state['consumed_train_samples'] == 10
    assert restored.optimizer_group['default'].cur_step == 5
    assert restored.optimizer_group['default'].gradient_accumulation_steps == 3


def test_read_training_progress_returns_metadata_without_mutating_optimizer_state(tmp_path, tiny_local_model_dir):
    ckpt_dir = prepare_full_param_checkpoint(tmp_path, tiny_local_model_dir, cur_step=6, consumed_train_samples=12)
    restored = build_full_param_model(ckpt_dir)

    progress = restored.read_training_progress(str(ckpt_dir))

    assert progress['cur_step'] == 6
    assert progress['consumed_train_samples'] == 12
    assert restored.optimizer_group[''].cur_step == 0
    assert restored.optimizer_group[''].gradient_accumulation_steps == 1
