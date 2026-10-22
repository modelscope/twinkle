# Copyright (c) ModelScope Contributors. All rights reserved.
"""Offline timeline analysis over :meth:`TimeRecorder.trace_events` output.

The recorder emits spans as self-contained event tuples; this module never
records anything, it only TRANSFORMS that stream: Chrome Trace rendering
(:func:`chrome_trace`), the span DAG (:func:`span_dag`), the critical lane
(:func:`critical_lane`) and a one-stop text report (:func:`dag_report`). All
module-level pure functions over plain data -- the same stateless-helper shape
as :mod:`stats`, with no torch and no recorder import, so consumers can also
feed hand-built event lists.

Lane semantics (the unit of "serial time"): a lane is the sequence of ROOT
spans (parent ``None``) on one thread id, ordered by start. Concurrent lanes
overlap without edges between them -- there is no cross-lane causality in the
data, so any such edge would be invented; that boundary is stated honestly in
the report instead of being papered over.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

__all__ = ['chrome_trace', 'span_dag', 'critical_lane', 'dag_report']

_NS_PER_US = 1e3
_NS_PER_MS = 1e6
_NS_PER_SEC = 1e9

#: One closed span as emitted by :meth:`TimeRecorder.trace_events`.
Event = Tuple[Optional[int], Optional[int], str, int, int, int, int, int]

#: Edge kinds in the span DAG: nesting (parent -> child) and same-lane ordering.
_NEST, _SEQ = 'nest', 'seq'


def _nodes(events: Iterable[Event]) -> Dict[int, Dict[str, Any]]:
    """Re-key the event stream as ``sid -> node`` dicts (the shared input shape)."""
    return {
        sid: {
            'sid': sid,
            'parent_sid': parent_sid,
            'name': name,
            'tid': tid,
            'start_ns': start_ns,
            'end_ns': end_ns,
            'self_ns': self_ns,
            'total_ns': total_ns,
        }
        for sid, parent_sid, name, tid, start_ns, end_ns, self_ns, total_ns in events
    }


def _subtree_self(nodes: Dict[int, Dict[str, Any]]) -> Dict[int, int]:
    """Self-time summed over each node's whole nesting subtree.

    ``sid`` is allocated at span construction, so a parent's sid is always
    smaller than its children's. Folding in DESCENDING sid order therefore
    reaches every node only after all of its descendants have folded into it,
    which makes this a single pass over ``sorted`` sids instead of a tree walk.
    """
    acc = {sid: node['self_ns'] for sid, node in nodes.items()}
    for sid in sorted(acc, reverse=True):
        parent_sid = nodes[sid]['parent_sid']
        if parent_sid is not None:
            acc[parent_sid] += acc[sid]
    return acc


def _lanes(nodes: Dict[int, Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """Root spans grouped by thread id, each lane sorted by start time."""
    lanes: Dict[int, List[Dict[str, Any]]] = {}
    for node in nodes.values():
        if node['parent_sid'] is None:
            lanes.setdefault(node['tid'], []).append(node)
    for lane in lanes.values():
        lane.sort(key=lambda node: node['start_ns'])
    return lanes


def _lane_chain_ns(lane: List[Dict[str, Any]], subtree: Dict[int, int]) -> int:
    """A lane's serial cost: Σ station subtree-self plus Σ positive gaps.

    Negative gaps (overlapping root spans on one thread, e.g. interleaved
    coroutines) contribute zero -- the overlapped wall second must not be
    counted twice in a serial chain.
    """
    stations = sum(subtree[station['sid']] for station in lane)
    gaps = sum(max(0, nxt['start_ns'] - prev['end_ns']) for prev, nxt in zip(lane, lane[1:]))
    return stations + gaps


def _fmt_ns(ns: int) -> str:
    """Human-readable duration for a nanosecond value."""
    if ns >= _NS_PER_SEC:
        return f'{ns / _NS_PER_SEC:.3f}s'
    if ns >= _NS_PER_MS:
        return f'{ns / _NS_PER_MS:.3f}ms'
    return f'{ns / _NS_PER_US:.1f}us'


def _fmt_pct(ns: int, wall_ns: int) -> str:
    return f'{(100.0 * ns / wall_ns):.0f}%' if wall_ns > 0 else '-'


def chrome_trace(events: Iterable[Event], *, pid: int = 0) -> List[Dict[str, Any]]:
    """Convert the event stream to Chrome Trace Event Format dicts (µs units).

    One complete event (``ph='X'``) per span, timestamped in microseconds; wrap
    the list as ``{'traceEvents': [...]}`` (or load the JSON file directly) to
    render in chrome://tracing or Perfetto. Nesting is implicit in ``ts``/``dur``
    overlap, so no parent field is emitted; ``sid`` rides along in ``args`` to
    correlate a trace row with the DAG nodes from :func:`span_dag`.
    """
    return [
        {
            'name': name,
            'ph': 'X',
            'pid': pid,
            'tid': tid,
            'ts': start_ns / _NS_PER_US,
            'dur': (end_ns - start_ns) / _NS_PER_US,
            'args': {'sid': sid, 'self_ns': self_ns},
        }
        for sid, parent_sid, name, tid, start_ns, end_ns, self_ns, total_ns in events
    ]


def span_dag(events: Iterable[Event]) -> Tuple[Dict[int, Dict[str, Any]], List[Tuple[int, int, str]]]:
    """The span DAG: ``nodes`` (``sid -> node``) and ``edges`` (from, to, kind).

    Two edge kinds exist: ``'nest'`` (parent -> child, from the recorded
    ``parent_sid``) and ``'seq'`` (same-lane root spans in start order, i.e. "B
    could only start after A finished"). Overlap between DIFFERENT lanes carries
    no edge -- see the module docstring for that boundary.
    """
    nodes = _nodes(events)
    return nodes, _edges(nodes, _lanes(nodes))


def _edges(nodes: Dict[int, Dict[str, Any]], lanes: Dict[int, List[Dict[str, Any]]]) -> List[Tuple[int, int, str]]:
    """DAG edges from nodes and their lanes: nesting edges plus lane-order edges."""
    edges: List[Tuple[int, int, str]] = []
    for node in nodes.values():
        if node['parent_sid'] is not None:
            edges.append((node['parent_sid'], node['sid'], _NEST))
    for lane in lanes.values():
        for prev, nxt in zip(lane, lane[1:]):
            edges.append((prev['sid'], nxt['sid'], _SEQ))
    return edges


def critical_lane(events: Iterable[Event]) -> List[Dict[str, Any]]:
    """The longest serial chain: the lane with maximal ``Σ subtree-self + Σ gaps``.

    Each station is one root span; its cost is the subtree Σself of the nesting
    tree below it, and gaps between stations are part of the chain (waiting is
    time too). This is the longest lane, not a proven critical path --
    concurrent lanes overlap without causal edges, so a truly concurrent
    bottleneck may hide in a shorter lane. Returns ``[]`` for an empty stream.
    """
    nodes = _nodes(events)
    if not nodes:
        return []
    subtree = _subtree_self(nodes)
    best_lane: List[Dict[str, Any]] = []
    best_chain_ns = -1
    for lane in _lanes(nodes).values():
        chain_ns = _lane_chain_ns(lane, subtree)
        if chain_ns > best_chain_ns:
            best_lane, best_chain_ns = lane, chain_ns
    stations: List[Dict[str, Any]] = []
    prev: Optional[Dict[str, Any]] = None
    for station in best_lane:
        # Negative gap_before_ns = overlap (interleaved coroutines on one thread).
        gap_ns = 0 if prev is None else station['start_ns'] - prev['end_ns']
        stations.append({
            'sid': station['sid'],
            'name': station['name'],
            'tid': station['tid'],
            'start_ns': station['start_ns'],
            'end_ns': station['end_ns'],
            'subtree_self_ns': subtree[station['sid']],
            'gap_before_ns': gap_ns,
        })
        prev = station
    return stations


def _mermaid_label(name: str) -> str:
    """Escape a span name for a quoted mermaid node label."""
    return name.replace('"', '#quot;')


def _mermaid(nodes: Dict[int, Dict[str, Any]], edges: List[Tuple[int, int, str]]) -> str:
    """The DAG as mermaid flowchart text: nesting as solid, lane order as dotted."""
    lines = ['flowchart TD']
    for sid in sorted(nodes):
        lines.append(f'  s{sid}["{_mermaid_label(nodes[sid]["name"])}"]')
    for from_sid, to_sid, kind in edges:
        arrow = '-->' if kind == _NEST else '-.->'
        lines.append(f'  s{from_sid} {arrow} s{to_sid}')
    return '\n'.join(lines)


def _station_line(station: Dict[str, Any], subtree_self_ns: int, wall_ns: int, window_start_ns: int) -> str:
    """One report row for a lane station: subtree self-time, share, wall window."""
    offset = station['start_ns'] - window_start_ns
    return (f'  {station["name"]}  subtree-self {_fmt_ns(subtree_self_ns)}'
            f' ({_fmt_pct(subtree_self_ns, wall_ns)} of wall)'
            f'  [{_fmt_ns(offset)} - {_fmt_ns(offset + station["total_ns"])}]')


def _gap_line(gap_ns: int) -> str:
    """One report row for the empty (or overlapped) time between two stations."""
    overlapped = ' (overlapped)' if gap_ns < 0 else ''
    return f'  -> gap {_fmt_ns(gap_ns)}{overlapped}'


def dag_report(events: Iterable[Event]) -> str:
    """One-stop text report: lanes, gaps, self-shares, bottlenecks, mermaid DAG.

    Sections: per-lane serial sequences (each station's subtree self-time and
    share of the window wall, with the gap to the previous station attributed as
    "empty time between A and B"), the critical lane highlighted, bottleneck
    conclusions (largest single self-time span, largest inter-station gap), and
    the mermaid rendering of :func:`span_dag`.
    """
    events = list(events)
    nodes = _nodes(events)
    if not nodes:
        return 'twinkle perf timeline: no events (recorder disabled or window empty).'
    lanes = _lanes(nodes)
    subtree = _subtree_self(nodes)
    critical_sids = {station['sid'] for station in critical_lane(events)}
    window_start_ns = min(node['start_ns'] for node in nodes.values())
    wall_ns = max(node['end_ns'] for node in nodes.values()) - window_start_ns

    lines = [f'twinkle perf timeline: {len(nodes)} events, wall {_fmt_ns(wall_ns)}, {len(lanes)} lane(s)', '']
    for tid in sorted(lanes):
        lane = lanes[tid]
        marker = ' [critical]' if lane[0]['sid'] in critical_sids else ''
        lines.append(f'lane tid={tid}{marker} ({len(lane)} station(s), chain {_fmt_ns(_lane_chain_ns(lane, subtree))}):')
        lines.append(_station_line(lane[0], subtree[lane[0]['sid']], wall_ns, window_start_ns))
        for prev, nxt in zip(lane, lane[1:]):
            lines.append(_gap_line(nxt['start_ns'] - prev['end_ns']))
            lines.append(_station_line(nxt, subtree[nxt['sid']], wall_ns, window_start_ns))
        lines.append('')

    largest_self = max(nodes.values(), key=lambda node: node['self_ns'])
    lines.append('bottlenecks:')
    lines.append(f'  largest self: {largest_self["name"]!r} (sid {largest_self["sid"]}) '
                 f'{_fmt_ns(largest_self["self_ns"])} ({_fmt_pct(largest_self["self_ns"], wall_ns)} of wall)')
    largest_gap: Optional[Tuple[int, Dict[str, Any], Dict[str, Any]]] = None
    for lane in lanes.values():
        for prev, nxt in zip(lane, lane[1:]):
            gap_ns = nxt['start_ns'] - prev['end_ns']
            if largest_gap is None or gap_ns > largest_gap[0]:
                largest_gap = (gap_ns, prev, nxt)
    if largest_gap is None:
        lines.append('  largest gap: none (every lane has a single station)')
    else:
        gap_ns, prev, nxt = largest_gap
        lines.append(f'  largest gap: {_fmt_ns(gap_ns)} after {prev["name"]!r} before {nxt["name"]!r}'
                     f' on lane tid={prev["tid"]}')
    lines.append('')

    lines.append('mermaid:')
    lines.extend('  ' + line for line in _mermaid(nodes, _edges(nodes, lanes)).splitlines())
    return '\n'.join(lines)
