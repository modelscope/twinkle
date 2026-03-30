import concurrent.futures
import sys
import types
import uuid
from pathlib import Path
from unittest.mock import Mock

import pytest
from peft import LoraConfig
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast


class _NoOpProcessPoolExecutor:
    def __init__(self, *args, **kwargs):
        pass

    def submit(self, fn, *args, **kwargs):
        raise RuntimeError('Process pool is disabled in this test environment.')


concurrent.futures.ProcessPoolExecutor = _NoOpProcessPoolExecutor

if 'zmq' not in sys.modules:
    zmq_stub = types.ModuleType('zmq')

    class _ZmqError:
        class Again(Exception):
            pass

    class _ZmqSocket:
        def setsockopt(self, *args, **kwargs):
            pass

        def setsockopt_string(self, *args, **kwargs):
            pass

        def bind(self, *args, **kwargs):
            pass

        def connect(self, *args, **kwargs):
            pass

        def send_string(self, *args, **kwargs):
            pass

        def send_pyobj(self, *args, **kwargs):
            pass

        def recv_string(self, *args, **kwargs):
            return ''

        def recv_pyobj(self, *args, **kwargs):
            return None

    class _ZmqContext:
        def socket(self, *args, **kwargs):
            return _ZmqSocket()

    zmq_stub.Context = _ZmqContext
    zmq_stub.Socket = _ZmqSocket
    zmq_stub.REQ = 0
    zmq_stub.REP = 1
    zmq_stub.PUB = 2
    zmq_stub.SUB = 3
    zmq_stub.SNDMORE = 4
    zmq_stub.IPV6 = 5
    zmq_stub.SUBSCRIBE = 6
    zmq_stub.RCVTIMEO = 7
    zmq_stub.SNDTIMEO = 8
    zmq_stub.LINGER = 9
    zmq_stub.error = _ZmqError
    sys.modules['zmq'] = zmq_stub

from twinkle import DeviceMesh
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
    root = Path(r'C:\Users\weika\.codex\memories') / 'twinkle-tests'
    root.mkdir(exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir()
    return path


@pytest.fixture
def tiny_local_model_dir(tmp_path):
    model_dir = tmp_path / 'tiny-gpt2'
    model_dir.mkdir()

    tokenizer = build_tiny_tokenizer()
    tokenizer.save_pretrained(model_dir)

    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_layer=1,
        n_head=1,
        n_embd=16,
        n_positions=32,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    GPT2LMHeadModel(config).save_pretrained(model_dir)
    return model_dir


def build_full_param_model(model_dir):
    return TransformersModel(
        model_cls='GPT2LMHeadModel',
        model_id=str(model_dir),
        mixed_precision='no',
        grad_scaler_config={},
    )


def build_multi_lora_model(model_dir):
    device_mesh = types.SimpleNamespace(fsdp_world_size=0, data_world_size=1)
    model = MultiLoraTransformersModel(
        model_cls='GPT2LMHeadModel',
        model_id=str(model_dir),
        mixed_precision='no',
        device_mesh=device_mesh,
        grad_scaler_config={},
    )
    model.add_adapter_to_model('default', LoraConfig(r=2, lora_alpha=4, target_modules=['c_attn']))
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


def run_resume_only_model_flow(model, dataloader, checkpoint_dir, ignore_data_skip):
    if ignore_data_skip:
        return None
    progress = model.read_training_progress(str(checkpoint_dir))
    dataloader.skip_consumed_samples(progress['consumed_train_samples'])
    model.optimizer_group[''].cur_step = progress['cur_step']
    model.optimizer_group[''].gradient_accumulation_steps = progress['gradient_accumulation_steps']
    return progress


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
    restored = build_full_param_model(ckpt_dir)
    restored.set_optimizer('AdamW', lr=1e-4)
    restored.set_lr_scheduler('LinearLR')
    restored.set_grad_scaler(device='cpu')

    trainer_state = restored.load_training_state(str(ckpt_dir))

    assert trainer_state['cur_step'] == 9
    assert restored.optimizer_group[''].cur_step == 9


def test_lora_resume_keeps_adapter_load_separate_from_training_state(tmp_path, tiny_local_model_dir):
    ckpt_dir = prepare_lora_checkpoint(tmp_path, tiny_local_model_dir, cur_step=5, consumed_train_samples=10)
    restored = build_multi_lora_model(tiny_local_model_dir)
    restored.load(name=ckpt_dir.name, output_dir=str(ckpt_dir.parent), adapter_name='default')
    restored.set_optimizer('AdamW', adapter_name='default', lr=1e-4)
    restored.set_lr_scheduler('LinearLR', adapter_name='default')
    restored.set_grad_scaler(adapter_name='default', device='cpu')

    trainer_state = restored.load_training_state(str(ckpt_dir), adapter_name='default')

    assert trainer_state['cur_step'] == 5


def test_read_training_progress_supports_resume_only_model(tmp_path, tiny_local_model_dir):
    ckpt_dir = prepare_full_param_checkpoint(tmp_path, tiny_local_model_dir, cur_step=6, consumed_train_samples=12)
    restored = build_full_param_model(ckpt_dir)
    dataloader = Mock()

    progress = run_resume_only_model_flow(restored, dataloader, ckpt_dir, ignore_data_skip=False)

    assert progress['cur_step'] == 6
    assert progress['consumed_train_samples'] == 12
    assert restored.optimizer_group[''].cur_step == 6
    dataloader.skip_consumed_samples.assert_called_once_with(12)


def test_resume_only_model_ignore_data_skip_leaves_progress_unrestored(tmp_path, tiny_local_model_dir):
    ckpt_dir = prepare_full_param_checkpoint(tmp_path, tiny_local_model_dir, cur_step=6, consumed_train_samples=12)
    restored = build_full_param_model(ckpt_dir)
    dataloader = Mock()
    restored.read_training_progress = Mock(wraps=restored.read_training_progress)

    progress = run_resume_only_model_flow(restored, dataloader, ckpt_dir, ignore_data_skip=True)

    assert progress is None
    assert restored.optimizer_group[''].cur_step == 0
    restored.read_training_progress.assert_not_called()
    dataloader.skip_consumed_samples.assert_not_called()
