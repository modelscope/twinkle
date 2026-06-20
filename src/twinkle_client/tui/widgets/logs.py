# Copyright (c) Twinkle Contributors. All rights reserved.
"""Log panel widget - efficient scrolling log display using RichLog."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import RichLog, Static
from textual.widget import Widget


class LogPanel(Widget):
    """Scrolling log panel showing training logs in real-time."""

    DEFAULT_CSS = """
    LogPanel {
        layout: vertical;
        border: solid $accent;
        padding: 0;
    }

    LogPanel > #log-title {
        dock: top;
        height: 1;
        background: $accent;
        color: $text;
        text-align: center;
    }

    LogPanel > #log-content {
        height: 1fr;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static('Logs', id='log-title')
        yield RichLog(id='log-content', max_lines=500, wrap=True, markup=True)

    def append_log(self, message: str) -> None:
        """Append a log message to the panel."""
        self.query_one('#log-content', RichLog).write(message)

    def clear(self) -> None:
        """Clear all log entries."""
        self.query_one('#log-content', RichLog).clear()
