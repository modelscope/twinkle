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
