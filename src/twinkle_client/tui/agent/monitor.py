# Copyright (c) Twinkle Contributors. All rights reserved.
"""Training monitor - LLM-driven periodic analysis of metrics and logs."""

from __future__ import annotations

import asyncio
import json
from twinkle.utils.logger import get_logger
from typing import Any, Callable

from openai import AsyncOpenAI

from twinkle_client.tui.agent.prompts import MONITOR_SYSTEM_PROMPT
from twinkle_client.tui.connection import LocalConnection

logger = get_logger()

# Prefix the LLM must use when no issue is found
_NORMAL_PREFIX = 'LGTM'


class TrainingMonitor:
    """Background task that periodically feeds metrics to LLM for analysis.

    Instead of hard-coded rules, the LLM reasons about training signals:
    loss trends, reward dynamics, gradient norms, KL divergence, entropy, etc.

    Uses get_metrics() (full read with last_n) instead of get_new_metrics()
    to avoid offset contention with the UI polling loop.
    """

    # Maximum data points to send to LLM per analysis cycle
    _MAX_METRICS_FOR_LLM = 50

    def __init__(
        self,
        connection: LocalConnection,
        on_message: Callable[[str], None],
        llm_base_url: str = 'http://localhost:11434/v1',
        llm_model: str = 'qwen3.5',
        llm_api_key: str = 'not-needed',
        poll_interval: float = 30.0,
    ):
        self.connection = connection
        self.on_message = on_message
        self.llm_model = llm_model
        self.poll_interval = poll_interval
        self._client = AsyncOpenAI(base_url=llm_base_url, api_key=llm_api_key)
        self._running = True
        self._last_analyzed_step: int = -1
        self._last_analyzed_run_id: str | None = None

    async def run(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._analyze()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f'Monitor cycle error: {e}')
            await asyncio.sleep(self.poll_interval)

    def stop(self) -> None:
        self._running = False

    async def _analyze(self) -> None:
        """Collect recent metrics + logs, ask LLM to analyze.

        Uses get_metrics(last_n=N) which is a full read (no offset).
        This avoids competing with the UI's incremental poll for the same offset.
        The _last_analyzed_step guard prevents re-analyzing unchanged data.
        """
        if not self.connection.current_run_id:
            return

        run_id = self.connection.current_run_id

        # Auto-reset when monitoring a different run
        if run_id != self._last_analyzed_run_id:
            self._last_analyzed_step = -1
            self._last_analyzed_run_id = run_id

        # Full read of last N metrics (does NOT advance any shared offset)
        metrics = self.connection.get_metrics(run_id, last_n=self._MAX_METRICS_FOR_LLM)
        if not metrics:
            return

        # Skip if no new steps since last analysis
        latest_step = metrics[-1].get('step', 0)
        if latest_step <= self._last_analyzed_step:
            return

        # Get status to check if training is still active
        status = self.connection.get_status(run_id)
        if status in ('stopped', 'completed', 'error', 'paused'):
            return  # Don't analyze inactive runs

        # Format and analyze
        data_summary = self._format_for_llm(metrics)
        response = await self._call_llm(data_summary)

        # Always update step marker
        self._last_analyzed_step = latest_step

        if response:
            self.on_message(f'[Monitor] {response}')

    def _format_for_llm(self, metrics: list[dict[str, Any]]) -> str:
        """Format metrics into a concise text block for LLM."""
        parts = []
        keys = [k for k in metrics[0].keys() if k != 'ts']
        parts.append(f'## Recent Metrics ({len(metrics)} entries)')
        parts.append(f'Fields: {", ".join(keys)}')
        parts.append('')

        # Show last 10 in detail
        parts.append('Last 10:')
        for m in metrics[-10:]:
            row = {k: v for k, v in m.items() if k != 'ts'}
            parts.append(f'  {json.dumps(row, default=str)}')

        # Trend summary (first half vs second half)
        if len(metrics) >= 10:
            mid = len(metrics) // 2
            parts.append('')
            parts.append('Trend (first half \u2192 second half):')
            for key in keys:
                if key in ('step', 'epoch'):
                    continue
                first_vals = [m.get(key) for m in metrics[:mid] if isinstance(m.get(key), (int, float))]
                last_vals = [m.get(key) for m in metrics[mid:] if isinstance(m.get(key), (int, float))]
                if first_vals and last_vals:
                    avg_first = sum(first_vals) / len(first_vals)
                    avg_last = sum(last_vals) / len(last_vals)
                    parts.append(f'  {key}: {avg_first:.6g} \u2192 {avg_last:.6g}')

        return '\n'.join(parts)

    async def _call_llm(self, data_summary: str) -> str | None:
        """Call LLM for analysis. Returns diagnosis text or None if normal.

        The prompt instructs the LLM to start with 'LGTM' if everything is normal.
        This makes filtering reliable regardless of the LLM's phrasing style.
        """
        messages = [
            {'role': 'system', 'content': MONITOR_SYSTEM_PROMPT},
            {'role': 'user', 'content': data_summary},
        ]
        try:
            response = await self._client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.3,
                max_tokens=300,
            )
            content = (response.choices[0].message.content or '').strip()
            if not content:
                return None
            # LLM is instructed to start with LGTM when normal
            if content.upper().startswith(_NORMAL_PREFIX):
                return None
            return content
        except Exception:
            return None
