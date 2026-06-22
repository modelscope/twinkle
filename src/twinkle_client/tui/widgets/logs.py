# Copyright (c) Twinkle Contributors. All rights reserved.
"""Log panel widget - efficient scrolling log display using RichLog."""

from __future__ import annotations

import re

from textual.app import ComposeResult
from textual.widgets import RichLog, Static
from textual.widget import Widget

# Regex to strip ALL terminal escape sequences (colors, cursor movement, etc.)
_ANSI_RE = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b[^\[\]]')


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

        # Strip ALL terminal control sequences and \r (progress bar carriage returns)
        clean = _ANSI_RE.sub('', message)
        clean = clean.replace('\r', '\n')  # \r from progress bars → treat as newline

        # Hard-wrap long lines to prevent overflow into adjacent panels
        for line in clean.splitlines():
            line = line.rstrip()
            if not line:
                continue
            while len(line) > width:
                log_widget.write(line[:width])
                line = line[width:]
            log_widget.write(line)

    def clear(self) -> None:
        """Clear all log entries."""
        self.query_one('#log-content', RichLog).clear()
