# Copyright (c) Twinkle Contributors. All rights reserved.
"""Training monitor - LLM-driven periodic health check.

Every poll cycle, the monitor gathers ALL available signals about the current
training run (process status, output logs, metrics) and feeds them to the LLM.
The LLM decides:
- LGTM: everything normal, no action needed
- WARNING: report an observation to the user (metrics anomaly, slow progress, etc.)
- FIX: the script has a bug → LLM outputs a fixed script → monitor applies it and restarts

This unified approach handles crashes, hangs, abnormal training, and metrics
anomalies in a single loop without separate hard-coded detection logic.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from twinkle.utils.logger import get_logger
from twinkle_client.auto.connection import LocalConnection

logger = get_logger()

# Maximum auto-fix attempts per run (prevent infinite retry loops)
_MAX_FIX_ATTEMPTS = 3

MONITOR_SYSTEM_PROMPT = """\
You are an automated ML training health monitor. Every ~30 seconds you receive a \
snapshot of ALL signals from a training run. Your job is to analyze and decide \
what action (if any) to take.

## Signals you will receive

- **Process status**: alive / zombie / exited / unknown
- **output.log tail**: recent process output (stdout+stderr combined, may contain errors or warnings)
- **Metrics**: recent training metrics (loss, reward, lr, etc.)
- **Stall duration**: seconds since last new metric was produced
- **Current train.py**: the full training script source (provided for accurate fixes)

## Decision framework

1. **LGTM** — training is progressing normally.
   - Process alive, metrics flowing, no errors in output, loss trending down.
   - Respond: `LGTM`

2. **WARNING** — something worth noting but not script-breaking.
   - Loss plateau, reward hacking, KL explosion, entropy collapse, stall < 5 min, etc.
   - Respond with a BRIEF (1-3 sentence) observation + suggestion.

3. **FIX** — the script has crashed or is broken and needs code changes.
   - Process dead/zombie with error traceback in output.
   - Server returned an error (400/500) that indicates a code bug.
   - Process stuck > 10 minutes with no metrics AND output shows an error.
   - Respond in this EXACT format:
```diagnosis
<1-2 sentence root cause>
```
```python
<complete fixed training script>
```

## Rules
- Be direct and actionable.
- Respond in the same language as the log content (Chinese or English).
- NEVER start with LGTM if there is any issue.
- For FIX: output the COMPLETE fixed script based on the provided "Current train.py". Only modify the lines that cause the error — do NOT rewrite from scratch or change the overall architecture.
- **MUST preserve resume logic** in fixed scripts: `rt.get_resume_info()` + `dataloader.skip_consumed_samples()`. Never hardcode `global_step = 0` if resume logic exists in the original.
- Common fixes:
  - "Batch size N must be >= data world size M" → increase batch_size to M
  - "save_dir does not exist on the server" → remove the save_dir parameter
  - Import errors → fix the import
  - Connection refused → check base_url
  - "Unknown format code 'f' for object of type 'str'" → remove float format specifiers (:.4f etc.) from print statements
- Do NOT suggest FIX for transient issues (network blip, temporary stall < 5 min).
- If process is alive and metrics are flowing but stale for < 3 min, say LGTM.
"""


class TrainingMonitor:
    """Unified LLM-driven training health monitor.

    Every poll cycle, collects all available signals and asks the LLM
    to analyze. The LLM may respond with LGTM, a warning, or a FIX
    (complete fixed script). The monitor applies fixes automatically.
    """

    _MAX_METRICS_FOR_LLM = 30

    def __init__(
        self,
        connection: LocalConnection,
        on_message: Callable[[str], None],
        llm_client: 'AsyncOpenAI',
        llm_model: str = 'qwen3.5',
        poll_interval: float = 30.0,
    ):
        self.connection = connection
        self.on_message = on_message
        self.llm_model = llm_model
        self.poll_interval = poll_interval
        self._client = llm_client
        self._running = True
        # Track state per run
        self._last_analyzed_run_id: str | None = None
        self._last_metric_time: float = time.time()
        self._last_metric_step: int = -1
        self._fix_attempts: dict[str, int] = {}
        # Avoid spamming: don't re-analyze if nothing changed
        self._last_snapshot_hash: str = ''

    async def run(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f'Monitor cycle error: {e}')
            await asyncio.sleep(self.poll_interval)

    def stop(self) -> None:
        self._running = False

    # ──────────────────────────────────────────────────────────────────────
    # Core: gather signals → ask LLM → act on response
    # ──────────────────────────────────────────────────────────────────────

    async def _check(self) -> None:
        """Single monitoring cycle."""
        if not self.connection.current_run_id:
            return

        run_id = self.connection.current_run_id

        # Reset when switching runs
        if run_id != self._last_analyzed_run_id:
            self._last_analyzed_run_id = run_id
            self._last_metric_time = time.time()
            self._last_metric_step = -1
            self._last_snapshot_hash = ''

        # Skip if already completed/stopped by user
        status = self.connection.get_status(run_id)
        if status in ('completed', 'stopped', 'paused'):
            return

        # Gather all signals
        snapshot = self._gather_snapshot(run_id, status)

        # Dedup: skip if snapshot hasn't meaningfully changed
        snapshot_hash = self._hash_snapshot(snapshot)
        if snapshot_hash == self._last_snapshot_hash:
            return
        self._last_snapshot_hash = snapshot_hash

        # Ask LLM
        llm_response = await self._ask_llm(snapshot)
        if llm_response is None:
            return

        # Parse and act
        await self._act_on_response(run_id, llm_response, snapshot)

    def _gather_snapshot(self, run_id: str, status: str) -> dict[str, Any]:
        """Gather all health signals for the run."""
        run_dir = self.connection.base_dir / run_id

        # Process status
        meta = self.connection.get_meta(run_id) or {}
        pid = meta.get('pid')
        process_status = 'unknown'
        if pid:
            if self.connection._is_process_alive(pid):
                process_status = 'alive'
            else:
                process_status = 'dead'

        # output.log tail (last 1500 chars, focus on errors)
        output_tail = ''
        output_file = run_dir / 'output.log'
        if output_file.exists():
            try:
                content = output_file.read_text(errors='replace')
                # Extract traceback if present
                tb_idx = content.rfind('Traceback (most recent call last)')
                if tb_idx >= 0:
                    output_tail = content[tb_idx:][-1500:]
                elif len(content) > 1500:
                    output_tail = content[-1500:]
                else:
                    output_tail = content
            except Exception:
                pass

        # Metrics
        metrics = self.connection.get_metrics(run_id, last_n=self._MAX_METRICS_FOR_LLM)

        # Update stall tracking
        if metrics:
            latest_step = metrics[-1].get('step', 0)
            if latest_step > self._last_metric_step:
                self._last_metric_step = latest_step
                self._last_metric_time = time.time()

        stall_seconds = time.time() - self._last_metric_time

        # Current train.py script content (for LLM to fix)
        script_content = ''
        script_file = run_dir / 'train.py'
        if script_file.exists():
            try:
                script_content = script_file.read_text(errors='replace')
            except Exception:
                pass

        return {
            'run_id': run_id,
            'meta_status': status,
            'process_status': process_status,
            'output_tail': output_tail.strip(),
            'metrics': metrics,
            'stall_seconds': int(stall_seconds),
            'fix_attempts': self._fix_attempts.get(run_id, 0),
            'script_content': script_content,
        }

    def _hash_snapshot(self, snapshot: dict[str, Any]) -> str:
        """Simple hash to detect meaningful changes between cycles."""
        # Key factors: process status, output tail hash, latest step, stall bucket
        parts = [
            snapshot['process_status'],
            str(len(snapshot['output_tail'])),
            str(snapshot['metrics'][-1].get('step', 0) if snapshot['metrics'] else 0),
            str(snapshot['stall_seconds'] // 60),  # bucket by minute
        ]
        return '|'.join(parts)

    def _format_snapshot(self, snapshot: dict[str, Any]) -> str:
        """Format snapshot into text for LLM."""
        parts = []
        parts.append(f'## Run: {snapshot["run_id"]}')
        parts.append(f'- Meta status: {snapshot["meta_status"]}')
        parts.append(f'- Process: {snapshot["process_status"]}')
        parts.append(f'- Time since last metric: {snapshot["stall_seconds"]}s')
        parts.append(f'- Auto-fix attempts so far: {snapshot["fix_attempts"]}/{_MAX_FIX_ATTEMPTS}')
        parts.append('')

        # output
        if snapshot['output_tail']:
            parts.append('## output.log (tail)')
            parts.append(f'```\n{snapshot["output_tail"]}\n```')
            parts.append('')

        # Metrics
        metrics = snapshot['metrics']
        if metrics:
            keys = [k for k in metrics[0].keys() if k != 'ts']
            parts.append(f'## Metrics ({len(metrics)} entries)')
            parts.append(f'Fields: {", ".join(keys)}')
            # Last 8 entries
            for m in metrics[-8:]:
                row = {k: v for k, v in m.items() if k != 'ts'}
                parts.append(f'  {json.dumps(row, default=str)}')
            # Trend
            if len(metrics) >= 6:
                mid = len(metrics) // 2
                parts.append('')
                parts.append('Trend (first half → second half avg):')
                for key in keys:
                    if key in ('step', 'epoch', 'total_steps'):
                        continue
                    first_vals = [m.get(key) for m in metrics[:mid] if isinstance(m.get(key), (int, float))]
                    last_vals = [m.get(key) for m in metrics[mid:] if isinstance(m.get(key), (int, float))]
                    if first_vals and last_vals:
                        avg_first = sum(first_vals) / len(first_vals)
                        avg_last = sum(last_vals) / len(last_vals)
                        parts.append(f'  {key}: {avg_first:.6g} → {avg_last:.6g}')
        else:
            parts.append('## Metrics: NONE (no metrics produced yet)')

        # Script content (for accurate fixes)
        if snapshot.get('script_content'):
            parts.append('')
            parts.append('## Current train.py')
            parts.append(f'```python\n{snapshot["script_content"]}\n```')

        return '\n'.join(parts)

    async def _ask_llm(self, snapshot: dict[str, Any]) -> str | None:
        """Send snapshot to LLM, return response or None on failure."""
        user_content = self._format_snapshot(snapshot)

        # If fix attempts exhausted, tell LLM not to suggest FIX
        extra = ''
        if snapshot['fix_attempts'] >= _MAX_FIX_ATTEMPTS:
            extra = '\n\nNOTE: Auto-fix attempts exhausted. Do NOT suggest FIX. Only report warnings.'

        try:
            response = await self._client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {'role': 'system', 'content': MONITOR_SYSTEM_PROMPT + extra},
                    {'role': 'user', 'content': user_content},
                ],
                temperature=0.3,
                max_tokens=4096,
            )
            content = (response.choices[0].message.content or '').strip()
            return content if content else None
        except Exception as e:
            logger.debug(f'Monitor LLM call failed: {e}')
            return None

    # ──────────────────────────────────────────────────────────────────────
    # Act on LLM response
    # ──────────────────────────────────────────────────────────────────────

    async def _act_on_response(self, run_id: str, response: str, snapshot: dict[str, Any]) -> None:
        """Parse LLM response and take appropriate action."""
        # Case 1: LGTM — no action
        if response.upper().startswith('LGTM'):
            return

        # Case 2: FIX — contains a ```python block
        diagnosis, fixed_script = self._parse_fix_response(response)
        if fixed_script:
            await self._apply_fix(run_id, diagnosis, fixed_script)
            return

        # Case 3: WARNING — just relay to user
        self.on_message(f'[Monitor] {response}')

    async def _apply_fix(self, run_id: str, diagnosis: str, fixed_script: str) -> None:
        """Apply auto-fix: update script + resume training."""
        attempts = self._fix_attempts.get(run_id, 0)
        if attempts >= _MAX_FIX_ATTEMPTS:
            self.on_message(
                f'[Monitor] 已达最大自动修复次数 ({_MAX_FIX_ATTEMPTS})，不再尝试。'
                '请手动检查或输入指令。'
            )
            return

        self.on_message(f'[Monitor] 检测到问题，正在自动修复 (第{attempts + 1}次)...\n诊断: {diagnosis}')

        try:
            # Update script (archives old version)
            self.connection.update_script(run_id, fixed_script)
            # Resume (re-launch)
            result = self.connection.resume_training(run_id)
            if result.get('status') == 'error':
                self.on_message(f'[Monitor] 重启失败: {result.get("error", "unknown")}')
            else:
                self.on_message(f'[Monitor] 脚本已修复并重启 (PID: {result.get("pid", "?")})')
                # Reset stall tracking for the new attempt
                self._last_metric_time = time.time()
                self._last_metric_step = -1
                self._last_snapshot_hash = ''
        except Exception as e:
            self.on_message(f'[Monitor] 自动修复失败: {e}')

        self._fix_attempts[run_id] = attempts + 1

    @staticmethod
    def _parse_fix_response(response: str) -> tuple[str, str]:
        """Parse LLM response for diagnosis + fixed script.

        Returns (diagnosis, fixed_script). fixed_script is empty if
        no ```python block found (meaning it's a WARNING, not a FIX).
        """
        diagnosis = ''
        fixed_script = ''

        # Extract python code block
        code_match = re.search(r'```python\s*\n(.*?)```', response, re.DOTALL)
        if not code_match:
            return '', ''

        fixed_script = code_match.group(1).strip()

        # Extract diagnosis block
        diag_match = re.search(r'```diagnosis\s*\n(.*?)```', response, re.DOTALL)
        if diag_match:
            diagnosis = diag_match.group(1).strip()
        else:
            # Fallback: text before the python block
            before = response[:response.find('```python')]
            lines = [l.strip() for l in before.splitlines() if l.strip() and not l.startswith('```')]
            diagnosis = lines[-1] if lines else 'Auto-fix applied'

        return diagnosis, fixed_script
