# Copyright (c) Twinkle Contributors. All rights reserved.
"""Metrics panel widget - renders training metrics as ASCII charts."""

from __future__ import annotations

from typing import Any

import plotext as plt
from rich.text import Text

from textual.app import ComposeResult
from textual.widgets import Static
from textual.widget import Widget

# Maximum data points to retain in memory
_MAX_HISTORY = 2000


class MetricsPanel(Widget):
    """Renders training metrics (loss, reward, etc.) as terminal plots."""

    DEFAULT_CSS = """
    MetricsPanel {
        layout: vertical;
        border: solid $warning;
        padding: 0;
    }

    MetricsPanel > #metrics-title {
        dock: top;
        height: 1;
        background: $warning;
        color: $text;
        text-align: center;
    }

    MetricsPanel > #metrics-hint {
        dock: bottom;
        height: 1;
        color: $text-muted;
        text-style: italic;
        padding: 0 1;
    }

    MetricsPanel > #metrics-plot {
        height: 1fr;
        padding: 0 1;
    }
    """

    # Max metric lines on a single chart
    _MAX_DISPLAY_KEYS = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metrics_history: list[dict[str, Any]] = []
        self._selected_keys: list[str] | None = None  # None = auto (first 3)
        self._x_range: tuple[int | None, int | None] = (None, None)
        self._y_range: tuple[float | None, float | None] = (None, None)

    def compose(self) -> ComposeResult:
        yield Static('Metrics', id='metrics-title')
        yield Static('No data yet...', id='metrics-plot')
        yield Static('', id='metrics-hint')

    def update_metrics(self, metrics: list[dict[str, Any]]) -> None:
        """Replace all metrics data and redraw the plot."""
        self._metrics_history = metrics[-_MAX_HISTORY:]
        self._redraw()

    def append_metrics(self, new_metrics: list[dict[str, Any]]) -> None:
        """Append new metrics incrementally and redraw."""
        self._metrics_history.extend(new_metrics)
        # Trim to max history
        if len(self._metrics_history) > _MAX_HISTORY:
            self._metrics_history = self._metrics_history[-_MAX_HISTORY:]
        self._redraw()

    def zoom(self, x_start: int | None = None, x_end: int | None = None,
             y_min: float | None = None, y_max: float | None = None) -> None:
        """Zoom into a specific range of the chart (by step value, not index)."""
        self._x_range = (x_start, x_end)
        self._y_range = (y_min, y_max)
        self._redraw()

    def reset_zoom(self) -> None:
        """Reset zoom to show all data."""
        self._x_range = (None, None)
        self._y_range = (None, None)
        self._redraw()

    def get_available_keys(self) -> list[str]:
        """Return all plottable metric keys from current data."""
        if not self._metrics_history:
            return []
        sample = self._metrics_history[0]
        # Exclude metadata fields that are not meaningful as plotted metrics
        exclude = ('step', 'ts', 'epoch', 'total_steps')
        return [k for k in sample.keys() if k not in exclude]

    def get_selected_keys(self) -> list[str]:
        """Return the currently displayed metric keys."""
        available = self.get_available_keys()
        if self._selected_keys is not None:
            return [k for k in self._selected_keys if k in available]
        return available[:3]

    def select_keys(self, keys: list[str]) -> dict[str, Any]:
        """Select which metrics to display. Returns status with available keys.

        Args:
            keys: Metric key names to show (max _MAX_DISPLAY_KEYS).
                  Pass empty list to reset to auto-select mode.
        """
        available = self.get_available_keys()
        if not keys:
            self._selected_keys = None
        else:
            # Validate & cap
            valid = [k for k in keys if k in available]
            self._selected_keys = valid[:self._MAX_DISPLAY_KEYS] if valid else None
        self._redraw()
        return {
            'selected': self.get_selected_keys(),
            'available': available,
        }

    def _redraw(self) -> None:
        """Redraw the metrics plot."""
        plot_widget = self.query_one('#metrics-plot', Static)
        hint_widget = self.query_one('#metrics-hint', Static)
        if not self._metrics_history:
            plot_widget.update('No data yet...')
            hint_widget.update('')
            return

        available = self.get_available_keys()
        if not available:
            plot_widget.update('No plottable metrics.')
            hint_widget.update('')
            return

        # Determine which keys to plot
        display_keys = self.get_selected_keys()
        hidden_keys = [k for k in available if k not in display_keys]

        # Update hint with hidden metrics
        if hidden_keys:
            hint_widget.update(
                f'[dim]Also available: {" ".join(hidden_keys)} '
                f'(ask agent to switch)[/]'
            )
        else:
            hint_widget.update('')

        # Get plot size from widget dimensions
        width = max(self.size.width - 4, 40)
        height = max(self.size.height - 6, 8)  # account for hint line

        plt.clf()
        plt.plotsize(width, height)
        plt.theme('dark')

        steps = [m.get('step', i) for i, m in enumerate(self._metrics_history)]

        # Apply x-range filter by step VALUE (not array index)
        x_start, x_end = self._x_range
        if x_start is not None or x_end is not None:
            filtered = [
                (s, m) for s, m in zip(steps, self._metrics_history)
                if (x_start is None or s >= x_start) and (x_end is None or s <= x_end)
            ]
            if filtered:
                steps, data_slice = zip(*filtered)
                steps = list(steps)
                data_slice = list(data_slice)
            else:
                plot_widget.update('No data in selected range.')
                return
        else:
            data_slice = self._metrics_history

        # Plot selected metrics
        for key in display_keys:
            values = [m.get(key, 0) for m in data_slice]
            if any(v is not None for v in values):
                plt.plot(steps, values, label=key)

        # Apply y-range
        y_min, y_max = self._y_range
        if y_min is not None or y_max is not None:
            plt.ylim(y_min or plt.ylim()[0], y_max or plt.ylim()[1])

        plt.title('Training Metrics')
        plt.xlabel('Step')

        # Render to string and convert ANSI to Rich Text for Textual
        plot_str = plt.build()
        plot_widget.update(Text.from_ansi(plot_str))
