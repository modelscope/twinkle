from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SCRIPT = REPO_ROOT / 'cookbook/client/server/megatron/run.sh'


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_run_script_existing_instance_detection_uses_pid_file_only() -> None:
    script = RUN_SCRIPT.read_text()

    assert 'pgrep -f "$RUN_SCRIPT_PATH"' not in script
    assert 'ps -eo pid=' not in script


def test_run_script_without_pid_file_does_not_report_existing_instance(tmp_path: Path) -> None:
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()

    _write_executable(
        fake_bin / 'ray',
        """#!/bin/sh
exit 0
""",
    )
    _write_executable(
        fake_bin / 'redis-server',
        """#!/bin/sh
exit 0
""",
    )
    _write_executable(
        fake_bin / 'redis-cli',
        """#!/bin/sh
case "$*" in
  *ping*) echo PONG ;;
esac
exit 0
""",
    )
    _write_executable(
        fake_bin / 'python',
        """#!/bin/sh
sleep 0.2
exit 0
""",
    )
    _write_executable(
        fake_bin / 'tail',
        """#!/bin/sh
trap 'exit 0' TERM INT
while :; do
  sleep 1
done
""",
    )
    _write_executable(
        fake_bin / 'pkill',
        """#!/bin/sh
exit 0
""",
    )
    _write_executable(
        fake_bin / 'ss',
        """#!/bin/sh
exit 0
""",
    )

    env = os.environ.copy()
    env.update({
        'PATH': f'{fake_bin}:{env["PATH"]}',
        'MODELSCOPE_CACHE': str(tmp_path / 'cache'),
        'TWINKLE_WORK_DIR': str(tmp_path / 'work'),
        'TWINKLE_RUN_PID_FILE': str(tmp_path / 'run.pid'),
        'TWINKLE_RUN_RESTART_REQUEST_FILE': str(tmp_path / 'run.restart'),
    })

    result = subprocess.run(
        [
            'bash',
            str(RUN_SCRIPT),
            '--head',
            '',
            '--gpu-workers',
            '',
            '--cpu-workers',
            '0',
            '--temp-dir',
            str(tmp_path / 'ray_logs'),
            '--save-dir',
            str(tmp_path / 'save'),
            '--server-config',
            str(tmp_path / 'server_config.yaml'),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 1
    assert '已有 run.sh 实例正在运行' not in result.stdout
    assert '检测到已有 run.sh 实例' not in result.stdout
    assert '无法向已有 run.sh 实例发送重启请求' not in result.stdout
    assert not (tmp_path / 'run.pid').exists()
    assert (tmp_path / 'cache').is_dir()
    assert (tmp_path / 'ray_logs').is_dir()
    assert (tmp_path / 'save').is_dir()
