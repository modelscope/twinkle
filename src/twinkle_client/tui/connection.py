# Copyright (c) Twinkle Contributors. All rights reserved.
"""Local file-based connection layer for TUI.

Reads metrics/logs from JSONL files written by the training script.

In Server Mode, training control is done by killing/restarting the client
process. The server retains all model/optimizer state in GPU memory.
- "Pause" = kill client process (SIGKILL)
- "Resume" = start a new client with same adapter_name
- "Stop" = graceful shutdown via SIGTERM (saves checkpoint)

File layout under run_dir (~/.cache/twinkle/{run_id}/):
    metrics.jsonl  — one JSON object per line, written after each step
    logs.jsonl     — one JSON object per line (ts + msg)
    meta.json      — run metadata (model_id, config, status, pid)
    train.py       — current active training script
    train_v{N}.py  — archived previous versions
"""

from __future__ import annotations

import json
from twinkle.utils.logger import get_logger
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

logger = get_logger()

DEFAULT_BASE_DIR = Path.home() / '.cache' / 'twinkle'


class LocalConnection:
    """File-based connection between TUI and training process.

    All monitoring happens through the local filesystem:
    - Metrics and logs are read from JSONL files (tail-style incremental)
    - Training control is via process management (kill/restart)
    """

    def __init__(self, base_dir: Path | str | None = None):
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_BASE_DIR
        self.current_run_id: str | None = None
        self._metrics_offsets: dict[str, int] = {}
        self._logs_offsets: dict[str, int] = {}
        self._crash_detected: set[str] = set()  # run_ids whose crash has already been reported

    # ──────────────────────────────────────────────────────────────────────
    # Meta
    # ──────────────────────────────────────────────────────────────────────

    def get_meta(self, run_id: str) -> dict[str, Any] | None:
        """Read and parse meta.json for a run."""
        meta_file = self.base_dir / run_id / 'meta.json'
        if not meta_file.exists():
            return None
        try:
            return json.loads(meta_file.read_text())
        except Exception:
            return None

    def _write_meta(self, run_id: str, meta: dict[str, Any]) -> None:
        """Write meta dict to meta.json for a run."""
        meta_file = self.base_dir / run_id / 'meta.json'
        meta_file.parent.mkdir(parents=True, exist_ok=True)
        meta_file.write_text(json.dumps(meta, indent=2))

    # ──────────────────────────────────────────────────────────────────────
    # Discovery
    # ──────────────────────────────────────────────────────────────────────

    def list_training_runs(self) -> list[dict[str, Any]]:
        """List all training runs by scanning base directory.

        A valid run directory must contain either meta.json or metrics.jsonl.
        """
        if not self.base_dir.exists():
            return []
        runs = []
        for entry in sorted(self.base_dir.iterdir(), reverse=True):
            if not entry.is_dir():
                continue
            meta_file = entry / 'meta.json'
            metrics_file = entry / 'metrics.jsonl'
            if not (meta_file.exists() or metrics_file.exists()):
                continue
            run_info = {'run_id': entry.name, 'dir': str(entry)}
            if meta_file.exists():
                try:
                    run_info.update(json.loads(meta_file.read_text()))
                except Exception:
                    pass
            runs.append(run_info)
        return runs

    # ──────────────────────────────────────────────────────────────────────
    # Metrics & logs (incremental reading)
    # ──────────────────────────────────────────────────────────────────────

    def get_metrics(self, run_id: str, last_n: int = 200) -> list[dict[str, Any]]:
        """Read metrics from JSONL file (tail last_n entries)."""
        metrics_file = self.base_dir / run_id / 'metrics.jsonl'
        if not metrics_file.exists():
            return []
        try:
            lines = metrics_file.read_text().strip().splitlines()
            recent = lines[-last_n:] if len(lines) > last_n else lines
            return [json.loads(line) for line in recent if line.strip()]
        except Exception:
            return []

    def get_new_metrics(self, run_id: str) -> list[dict[str, Any]]:
        """Read only new metrics since last read (incremental, per-run)."""
        metrics_file = self.base_dir / run_id / 'metrics.jsonl'
        if not metrics_file.exists():
            return []
        try:
            offset = self._metrics_offsets.get(run_id, 0)
            with open(metrics_file, 'r') as f:
                f.seek(offset)
                new_data = f.read()
                self._metrics_offsets[run_id] = f.tell()
            if not new_data.strip():
                return []
            return [json.loads(line) for line in new_data.strip().splitlines() if line.strip()]
        except Exception:
            return []

    def get_new_logs(self, run_id: str) -> list[dict[str, Any]]:
        """Read only new logs since last read (incremental, per-run).

        Also detects process crashes: if the process is dead and status is
        still 'running', reads stderr.log tail and injects it as error logs,
        then updates meta.status to 'error'.
        """
        logs_file = self.base_dir / run_id / 'logs.jsonl'
        entries: list[dict[str, Any]] = []

        # Read new log entries from logs.jsonl
        if logs_file.exists():
            try:
                offset = self._logs_offsets.get(run_id, 0)
                with open(logs_file, 'r') as f:
                    f.seek(offset)
                    new_data = f.read()
                    self._logs_offsets[run_id] = f.tell()
                if new_data.strip():
                    entries = [json.loads(line) for line in new_data.strip().splitlines() if line.strip()]
            except Exception:
                pass

        # Detect crashed process and inject stderr.log content
        if run_id not in self._crash_detected:
            meta = self.get_meta(run_id)
            if meta and meta.get('status') in ('running', 'starting'):
                pid = meta.get('pid')
                if pid and not self._is_process_alive(pid):
                    # Process died — inject stderr as error log
                    self._crash_detected.add(run_id)
                    stderr_file = self.base_dir / run_id / 'stderr.log'
                    error_msg = self._read_stderr_tail(stderr_file)
                    if error_msg:
                        entries.append({'ts': time.time(), 'msg': f'[ERROR] Process crashed:\n{error_msg}'})
                    else:
                        entries.append({'ts': time.time(), 'msg': '[ERROR] Process exited unexpectedly (no stderr output)'})
                    # Update meta status
                    meta['status'] = 'error'
                    self._write_meta(run_id, meta)
                    # Also persist the error into logs.jsonl for future reads
                    for entry in entries[-1:]:
                        self._append_log(run_id, entry['msg'])

        return entries

    def reset_offsets(self, run_id: str) -> None:
        """Reset incremental read offsets for a run (e.g., after switching runs)."""
        self._metrics_offsets.pop(run_id, None)
        self._logs_offsets.pop(run_id, None)
        self._crash_detected.discard(run_id)

    # ──────────────────────────────────────────────────────────────────────
    # Process health helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_process_alive(pid: int) -> bool:
        """Check if a process is still running (POSIX).

        Returns False for zombie processes (state 'Z') since they have
        already exited even though the PID still exists.
        """
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError, OSError):
            return False
        # PID exists — but check if it's a zombie via /proc
        try:
            status_file = Path(f'/proc/{pid}/status')
            if status_file.exists():
                for line in status_file.read_text().splitlines():
                    if line.startswith('State:'):
                        # State: Z (zombie) / R (running) / S (sleeping) / D (disk sleep)
                        return 'Z' not in line
        except Exception:
            pass
        return True

    @staticmethod
    def _read_stderr_tail(stderr_file: Path, max_chars: int = 2000) -> str:
        """Read the tail of stderr.log, focusing on the traceback/error."""
        if not stderr_file.exists():
            return ''
        try:
            content = stderr_file.read_text(errors='replace')
            if not content.strip():
                return ''
            # Try to extract just the traceback (from last 'Traceback' onwards)
            tb_marker = content.rfind('Traceback (most recent call last)')
            if tb_marker >= 0:
                return content[tb_marker:][-max_chars:]
            # No traceback found, return tail
            return content[-max_chars:]
        except Exception:
            return ''

    # ──────────────────────────────────────────────────────────────────────
    # Process management
    # ──────────────────────────────────────────────────────────────────────

    def _launch_script(self, run_id: str) -> dict[str, Any]:
        """Launch the run's train.py as a background subprocess.

        Captures stderr to stderr.log so script errors are diagnosable.
        Returns a dict with launch result (pid or error).
        """
        meta = self.get_meta(run_id)
        if not meta:
            return {'status': 'error', 'run_id': run_id, 'error': f'No meta.json for run {run_id}'}

        script_path = meta.get('script_path')
        if not script_path or not Path(script_path).exists():
            return {'status': 'error', 'run_id': run_id, 'error': f'Script not found: {script_path}'}

        run_dir = self.base_dir / run_id
        stderr_file = run_dir / 'stderr.log'

        try:
            stderr_fh = open(stderr_file, 'w')
        except OSError as e:
            return {'status': 'error', 'run_id': run_id, 'error': f'Cannot open stderr log: {e}'}

        try:
            env = os.environ.copy()
            env['TWINKLE_RUN_ID'] = run_id
            proc = subprocess.Popen(
                ['python', script_path],
                cwd=str(run_dir),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=stderr_fh,
                start_new_session=True,
            )
        except OSError as e:
            stderr_fh.close()
            return {'status': 'error', 'run_id': run_id, 'error': f'Failed to launch script: {e}'}
        # NOTE: Do NOT close stderr_fh here. The subprocess inherits the fd;
        # closing it in the parent is safe only after Popen duplicates it.
        # On POSIX, Popen dups the fd so we can close safely in parent.
        stderr_fh.close()

        # Non-blocking check: if process already exited (e.g., syntax error)
        retcode = proc.poll()
        if retcode is not None:
            error_msg = stderr_file.read_text().strip()[-500:] if stderr_file.exists() else ''
            meta['status'] = 'error'
            self._write_meta(run_id, meta)
            self._append_log(run_id, f'Script failed (exit={retcode}): {error_msg}')
            return {'status': 'error', 'run_id': run_id, 'error': error_msg or f'Process exited immediately (code={retcode})'}

        meta['pid'] = proc.pid
        meta['status'] = 'running'
        self._write_meta(run_id, meta)
        return {'status': 'running', 'run_id': run_id, 'pid': proc.pid, 'script_path': script_path}

    def _append_log(self, run_id: str, message: str) -> None:
        """Append a log entry to a run's logs.jsonl."""
        logs_file = self.base_dir / run_id / 'logs.jsonl'
        logs_file.parent.mkdir(parents=True, exist_ok=True)
        entry = {'ts': time.time(), 'msg': message}
        with open(logs_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def start_training(self, run_id: str, script_content: str, model_id: str = '') -> dict[str, Any]:
        """Create a new training run and launch the script.

        Args:
            run_id: Unique identifier for the run (timestamp suffix auto-appended).
            script_content: Full Python source of the training script.
            model_id: Model identifier for metadata.
        """
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_id = f'{run_id}_{timestamp}'

        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        train_py = run_dir / 'train.py'
        train_py.write_text(script_content)

        meta = {
            'run_id': run_id,
            'model_id': model_id,
            'status': 'starting',
            'script_path': str(train_py),
            'script_version': 1,
            'start_time': time.time(),
        }
        self._write_meta(run_id, meta)
        self.current_run_id = run_id

        return self._launch_script(run_id)

    def pause_training(self, run_id: str) -> dict[str, Any]:
        """Pause training by killing the client process (SIGKILL).

        Server retains all state — restart the script to continue.
        """
        meta = self.get_meta(run_id)
        pid = meta.get('pid') if meta else None

        if pid:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

        if meta:
            meta['status'] = 'paused'
            self._write_meta(run_id, meta)

        return {'status': 'paused', 'run_id': run_id, 'pid': pid}

    def resume_training(self, run_id: str) -> dict[str, Any]:
        """Resume training by re-launching the stored training script.

        Server state (LoRA weights, optimizer, LR scheduler) is preserved in GPU memory.
        """
        return self._launch_script(run_id)

    def stop_training(self, run_id: str) -> dict[str, Any]:
        """Stop training gracefully via SIGTERM.

        The training script's SIGTERM handler saves checkpoint + dataloader state,
        then exits. Training can later be resumed from checkpoint.
        """
        meta = self.get_meta(run_id)
        pid = meta.get('pid') if meta else None

        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

        if meta:
            meta['status'] = 'stopping'
            self._write_meta(run_id, meta)

        return {'status': 'stopping', 'run_id': run_id, 'pid': pid}

    def update_script(self, run_id: str, new_script_content: str) -> dict[str, Any]:
        """Update the training script with version archiving.

        Archives the current train.py as train_v{N}.py, writes new content.
        Version numbering is based on the actual max version found on disk.
        """
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        train_py = run_dir / 'train.py'

        # Archive existing script with robust version numbering
        version = 1
        if train_py.exists():
            # Find the actual max version number from filenames
            max_v = 0
            for f in run_dir.glob('train_v*.py'):
                m = re.match(r'train_v(\d+)\.py$', f.name)
                if m:
                    max_v = max(max_v, int(m.group(1)))
            archive_v = max_v + 1
            shutil.copy2(train_py, run_dir / f'train_v{archive_v}.py')
            version = archive_v + 1

        train_py.write_text(new_script_content)

        # Update meta
        meta = self.get_meta(run_id) or {'run_id': run_id}
        meta['script_version'] = version
        meta['script_path'] = str(train_py)
        self._write_meta(run_id, meta)

        return {
            'run_id': run_id,
            'script_version': version,
            'script_path': str(train_py),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Status queries
    # ──────────────────────────────────────────────────────────────────────

    def get_status(self, run_id: str) -> str:
        """Get the current status string for a run."""
        meta = self.get_meta(run_id)
        return meta.get('status', 'unknown') if meta else 'unknown'
