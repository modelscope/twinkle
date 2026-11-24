import pickle
import sys
from types import SimpleNamespace

import pytest

from twinkle.utils.platforms import npu


@pytest.mark.parametrize(
    ('software_version', 'cann_version', 'expected'),
    [
        ('25.3.rc1.2', '8.3.rc1', True),
        ('25.5.t3.b001', '8.3.0', True),
        ('25.2.0', '8.3.rc1', False),
        ('25.3.rc1', '8.2.0', False),
    ],
)
def test_npu_ipc_version_gate(monkeypatch, tmp_path, software_version, cann_version, expected):
    monkeypatch.setattr(
        npu.subprocess,
        'run',
        lambda *args, **kwargs: SimpleNamespace(stdout=f'Software Version : {software_version}\n'),
    )
    monkeypatch.setattr(npu.platform, 'machine', lambda: 'x86_64')
    cann_dir = tmp_path / 'x86_64-linux'
    cann_dir.mkdir()
    (cann_dir / 'ascend_toolkit_install.info').write_text(f'version={cann_version}\n')
    monkeypatch.setenv('ASCEND_HOME_PATH', str(tmp_path))

    assert npu.NPU.is_ipc_supported() is expected


def test_npu_ipc_detection_errors_are_not_hidden(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('npu-smi failed')

    monkeypatch.setattr(npu.subprocess, 'run', fail)

    with pytest.raises(RuntimeError, match='npu-smi failed'):
        npu.NPU.is_ipc_supported()


def test_npu_fallback_identity_matches_physical_device_across_visible_masks(monkeypatch):
    def unsupported(*args):
        raise NotImplementedError

    monkeypatch.setitem(sys.modules, 'vllm.platforms', SimpleNamespace(
        current_platform=SimpleNamespace(get_device_uuid=unsupported)))
    monkeypatch.setattr(npu, '_get_npu_bus_id_from_npu_smi', lambda *args: None)
    monkeypatch.setattr(npu.socket, 'gethostname', lambda: 'node-a')
    monkeypatch.setenv('ASCEND_RT_VISIBLE_DEVICES', '13')
    trainer = npu.NPU.get_vllm_device_uuid(0)
    monkeypatch.setenv('ASCEND_RT_VISIBLE_DEVICES', '6,8,9,13')
    assert npu.NPU.get_vllm_device_uuid(3) == trainer
    assert npu.NPU.get_vllm_device_uuid(0) != trainer
    monkeypatch.setattr(npu.socket, 'gethostname', lambda: 'node-b')
    assert npu.NPU.get_vllm_device_uuid(3) != trainer


def test_npu_native_uuid_is_a_serializable_string(monkeypatch):
    class NativeUUID:
        def __str__(self):
            return '00cc0800-d001-0000-0000-48706f0f1ee7'

        def __reduce__(self):
            raise TypeError("cannot pickle 'torch_npu._C._CUuuid' object")

    native_uuid = NativeUUID()
    monkeypatch.setitem(sys.modules, 'vllm.platforms', SimpleNamespace(
        current_platform=SimpleNamespace(get_device_uuid=lambda device_id: native_uuid)))

    result = npu.NPU.get_vllm_device_uuid(0)

    assert isinstance(result, str)
    assert result == str(native_uuid)
    assert pickle.loads(pickle.dumps(result)) == result
