# Copyright (c) Twinkle Contributors. All rights reserved.
"""Log panel widget - efficient scrolling log display using RichLog."""

from __future__ import annotations

import re

from textual.app import ComposeResult
from textual.widgets import RichLog, Static
from textual.widget import Widget

# Regex to strip ANSI escape sequences for accurate width measurement
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


class LogPanel(Widget):
    """Scrolling log panel showing training logs in real-time."""

    DEFAULT_CSS = """
    LogPanel {
        layout: vertical;
        border: solid $accent;
        padding: 0;
        overflow: hidden hidden;
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
        width: 100%;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static('Logs', id='log-title')
        yield RichLog(id='log-content', max_lines=500, wrap=True, markup=False)

    def append_log(self, message: str) -> None:
        """Append a log message to the panel, hard-wrapping to avoid overflow."""
        log_widget = self.query_one('#log-content', RichLog)
        # Get available width (subtract padding); fallback to 60
        width = (log_widget.size.width or 60) - 2
        if width < 20:
            width = 60

        # Strip ANSI codes for display (RichLog with markup=False can't handle them)
        clean = _ANSI_RE.sub('', message)

        # Hard-wrap long lines
        for line in clean.splitlines() or ['']:
            while len(line) > width:
                log_widget.write(line[:width])
                line = line[width:]
            log_widget.write(line)

    def clear(self) -> None:
        """Clear all log entries."""
        self.query_one('#log-content', RichLog).clear()
