# Copyright (c) Twinkle Contributors. All rights reserved.
"""Training runtime utilities for auto-research integration.

This module provides helpers that training scripts import to:
1. Write structured metrics to metrics.jsonl
2. Print log messages to stdout (captured as output.log by launcher)
3. Manage run lifecycle (start/end)
4. Register SIGTERM handler for graceful shutdown with checkpoint

In Server Mode, the client is stateless - killing the client process is
equivalent to "pause" (server retains all optimizer/model state in GPU memory).
Restarting the script with the same adapter_name seamlessly continues training.

Usage in training scripts:
    from twinkle_client.auto.runtime import TrainingRuntime

    rt = TrainingRuntime(run_id='my-grpo-run')
    rt.start(model_id='Qwen/Qwen3.5-4B', config={...})
    rt.register_graceful_shutdown(model, dataloader)

    for step, batch in enumerate(dataloader):
        # ... training logic ...
        rt.log_metrics(step=step, loss=loss, reward=reward, grad_norm=gn, lr=lr)
        rt.log(f'Completed step {step}, loss={loss:.4f}')

    rt.finish()
"""

from __future__ import annotations

import json
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from twinkle_client.model import MultiLoraTransformersModel
    from twinkle.dataloader import DataLoader


DEFAULT_BASE_DIR = Path.home() / '.cache' / 'twinkle'


class TrainingRuntime:
    """Runtime helper for training scripts to integrate with auto-research.

    Manages:
    - Writing metrics.jsonl (structured step data)
    - Printing logs to stdout (captured by launcher as output.log)
    - Run metadata (meta.json)
    - SIGTERM graceful shutdown with checkpoint saving
    """

    def __init__(self, run_id: str | None = None, base_dir: Path | str | None = None):
        """Initialize the training runtime.

        Args:
            run_id: Unique identifier for this training run.
                If None, reads from TWINKLE_RUN_ID environment variable
                (automatically set by launcher).
            base_dir: Base directory for run data. Defaults to ~/.cache/twinkle/
        """
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_BASE_DIR
        if run_id is None:
            run_id = os.environ.get('TWINKLE_RUN_ID', '')
            if not run_id:
                raise ValueError(
                    'run_id must be provided or TWINKLE_RUN_ID env var must be set'
                )
        self.run_id = run_id
        self.run_dir = self.base_dir / run_id

        self._metrics_file: Any = None
        self._started = False
        self._last_progress_save: float = 0.0  # timestamp of last progress write
        self._progress_save_interval: float = 5.0  # write at most every 5 seconds

    def start(
        self,
        model_id: str = '',
        config: dict[str, Any] | None = None,
        script_path: str | Path | None = None,
    ) -> None:
        """Initialize the run directory and write metadata.

        Call this once at the beginning of training.

        Args:
            model_id: Model identifier (e.g. 'Qwen/Qwen3.5-4B').
            config: Training configuration dict (hyperparameters, etc.).
            script_path: Path to the training script. If provided, the script
                will be copied into the run directory as ``train.py`` so that
                resume/restart can re-execute it automatically.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Copy training script into run directory for reproducibility & restart.
        # If train.py already exists (e.g. from a previous failed run), archive it
        # as train_v{N}.py before overwriting.
        stored_script: str | None = None
        script_version = 1
        if script_path is not None:
            import shutil
            src = Path(script_path).resolve()
            dst = self.run_dir / 'train.py'
            if src.exists() and src != dst.resolve():
                # Archive existing train.py if present
                if dst.exists():
                    # Find max existing version number (regex-based, consistent with connection.py)
                    max_v = 0
                    for f in self.run_dir.glob('train_v*.py'):
                        m = re.match(r'train_v(\d+)\.py$', f.name)
                        if m:
                            max_v = max(max_v, int(m.group(1)))
                    archive_v = max_v + 1
                    shutil.copy2(dst, self.run_dir / f'train_v{archive_v}.py')
                    script_version = archive_v + 1
                shutil.copy2(src, dst)
            stored_script = str(dst)

        # Write run metadata
        meta = {
            'run_id': self.run_id,
            'model_id': model_id,
            'config': config or {},
            'start_time': time.time(),
            'status': 'running',
            'pid': os.getpid(),
            'script_path': stored_script,
            'script_version': script_version,
        }
        (self.run_dir / 'meta.json').write_text(json.dumps(meta, indent=2))

        # Open metrics file for append
        self._metrics_file = open(self.run_dir / 'metrics.jsonl', 'a', buffering=1)

        self._started = True
        self.log('Training started')

    def log_metrics(self, **kwargs) -> None:
        """Write a metrics entry to metrics.jsonl.

        All keyword arguments are written as a single JSON line.
        A timestamp is automatically added.
        Values are auto-converted to float where possible; unconvertible values are dropped.

        Example:
            rt.log_metrics(step=10, loss=0.5, reward=1.2, grad_norm=0.8, lr=1e-5)
        """
        if not self._metrics_file:
            return
        entry = {'ts': time.time()}
        for k, v in kwargs.items():
            try:
                entry[k] = float(v)
            except (TypeError, ValueError):
                pass  # skip unconvertible values
        self._metrics_file.write(json.dumps(entry) + '\n')

        # Auto-save training progress to meta.json (for resume after crash)
        step = entry.get('step')
        if step is not None:
            now = time.time()
            if now - self._last_progress_save >= self._progress_save_interval:
                progress = {'last_step': int(step)}
                if 'total_steps' in entry:
                    progress['total_steps'] = int(entry['total_steps'])
                self._save_progress(progress)
                self._last_progress_save = now

    def log(self, message: str) -> None:
        """Print a log message to stdout (captured as output.log by launcher).

        Args:
            message: Human-readable log message.
        """
        print(f'[twinkle] {message}', flush=True)

    def get_resume_info(self) -> dict[str, int]:
        """Get resume info from a previous run (if any).

        Reads the 'progress' field from meta.json, which is auto-saved
        during training by log_metrics().

        Returns:
            dict with 'last_step' (int, default 0) and optionally 'total_steps'.
            Always returns a dict — never None. Fresh start returns {'last_step': 0}.

        Usage:
            resume = rt.get_resume_info()
            global_step = resume['last_step']
            if global_step > 0:
                dataloader.skip_consumed_samples(global_step * BATCH_SIZE)
                print(f'[twinkle] Resuming from step {global_step}')
        """
        meta_path = self.run_dir / 'meta.json'
        if not meta_path.exists():
            return {'last_step': 0}
        try:
            meta = json.loads(meta_path.read_text())
            progress = meta.get('progress')
            if progress and progress.get('last_step', 0) > 0:
                return progress
        except Exception:
            pass
        return {'last_step': 0}

    def _save_progress(self, progress: dict[str, int]) -> None:
        """Save training progress to meta.json (throttled, non-blocking)."""
        meta_path = self.run_dir / 'meta.json'
        try:
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            meta['progress'] = progress
            meta_path.write_text(json.dumps(meta, indent=2))
        except Exception:
            pass  # never crash training for progress bookkeeping

    def finish(self, status: str = 'completed') -> None:
        """Mark training as finished and close files.

        Args:
            status: Final status ('completed', 'stopped', 'error').
        """
        self.log(f'Training finished with status: {status}')

        # Update metadata
        meta_file = self.run_dir / 'meta.json'
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                meta['status'] = status
                meta['end_time'] = time.time()
                meta_file.write_text(json.dumps(meta, indent=2))
            except Exception:
                pass

        # Close files
        if self._metrics_file:
            self._metrics_file.close()
            self._metrics_file = None

        self._started = False

    def register_graceful_shutdown(
        self,
        model: 'MultiLoraTransformersModel',
        dataloader: 'DataLoader | None' = None,
        checkpoint_name: str = 'interrupted',
    ) -> None:
        """Register SIGTERM handler for graceful shutdown with checkpoint.

        When SIGTERM is received (e.g., from stop command), the handler will:
        1. Save model checkpoint (LoRA weights + optimizer state)
        2. Save dataloader position (consumed_train_samples) for exact resume
        3. Log the checkpoint path
        4. Mark training as 'stopped' and exit

        Args:
            model: The MultiLoraTransformersModel instance.
            dataloader: Optional DataLoader with .get_state() support.
            checkpoint_name: Name for the saved checkpoint.

        Usage:
            rt = TrainingRuntime(run_id='my-run')
            rt.start(...)
            rt.register_graceful_shutdown(model, dataloader)
            # ... training loop ...
        """
        def _shutdown_handler(signum, frame):
            self.log('SIGTERM received, saving checkpoint before exit...')
            try:
                save_kwargs = {
                    'name': checkpoint_name,
                    'save_optimizer': True,
                }
                if dataloader is not None:
                    state = dataloader.get_state()
                    save_kwargs['consumed_train_samples'] = state.get('consumed_train_samples', 0)
                    self.log(f'Dataloader state: consumed_train_samples={save_kwargs["consumed_train_samples"]}')

                result = model.save(**save_kwargs)
                self.log(f'Checkpoint saved: {result}')
            except Exception as e:
                self.log(f'Error saving checkpoint during shutdown: {e}')

            self.finish(status='stopped')
            sys.exit(0)

        signal.signal(signal.SIGTERM, _shutdown_handler)
        self.log('Graceful shutdown handler registered (SIGTERM)')
