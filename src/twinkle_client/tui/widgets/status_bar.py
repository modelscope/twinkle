# Copyright (c) Twinkle Contributors. All rights reserved.
"""Status bar widget - shows training progress and status."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import Static
from textual.widget import Widget


class StatusBar(Widget):
    """Top status bar showing training state, model, step count, and progress."""

    DEFAULT_CSS = """
    StatusBar {
        layout: horizontal;
        height: 3;
        background: $surface;
        border-bottom: solid $primary;
        padding: 0 2;
    }

    StatusBar > .status-item {
        width: auto;
        padding: 0 2;
        content-align: center middle;
    }

    StatusBar > #status-state {
        color: $success;
        text-style: bold;
    }

    StatusBar > #status-run {
        color: $text-muted;
    }

    StatusBar > #status-model {
        color: $text;
    }

    StatusBar > #status-step {
        color: $warning;
    }

    StatusBar > #status-progress {
        color: $accent;
        width: 1fr;
        text-align: right;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static('⏸ Idle', id='status-state', classes='status-item')
        yield Static('Run: -', id='status-run', classes='status-item')
        yield Static('Model: -', id='status-model', classes='status-item')
        yield Static('Step: 0', id='status-step', classes='status-item')
        yield Static('', id='status-progress', classes='status-item')

    def update_status(
        self,
        state: str | None = None,
        model: str | None = None,
        step: int | None = None,
        total_steps: int | None = None,
        run_id: str | None = None,
    ) -> None:
        """Update status bar fields."""
        if state is not None:
            state_icons = {
                'training': '🚀 Training',
                'paused': '⏸ Paused',
                'idle': '⏸ Idle',
                'error': '❌ Error',
                'completed': '✅ Done',
            }
            self.query_one('#status-state', Static).update(
                state_icons.get(state, f'● {state}')
            )
        if model is not None:
            self.query_one('#status-model', Static).update(f'Model: {model}')
        if run_id is not None:
            self.query_one('#status-run', Static).update(f'Run: {run_id}')
        if step is not None:
            self.query_one('#status-step', Static).update(f'Step: {step}')
        if total_steps is not None and step is not None:
            pct = min(100, int(step / total_steps * 100)) if total_steps > 0 else 0
            bar_len = 20
            filled = int(bar_len * pct / 100)
            bar = '█' * filled + '░' * (bar_len - filled)
            self.query_one('#status-progress', Static).update(
                f'[{bar}] {pct}% ({step}/{total_steps})'
            )
