# Copyright (c) ModelScope Contributors. All rights reserved.
"""``run_submit`` must stay a free function.

Reverting it to a decorator would rewrite its signature and deepen the route graph, which
once hit the CPython C-stack recursion limit in ``serve.ingress``'s cloudpickle phase.
Adding a ``self: QueuedDeployment`` annotation is a zero-runtime-cost change, so
the function form must be unchanged.
"""
import inspect

from twinkle.server.lifecycle.protocols import DataParallelDeployment, QueuedDeployment
from twinkle.server.lifecycle.submit import input_metrics, run_submit


def test_run_submit_is_a_free_function():
    assert inspect.isfunction(run_submit)
    assert inspect.iscoroutinefunction(run_submit)
    assert inspect.isfunction(input_metrics)


def test_host_protocols_are_two_layers():
    # DataParallelDeployment is the strictly stronger contract (adds data_world_size).
    # (``issubclass`` is avoided: runtime_checkable Protocols with data members raise.)
    assert QueuedDeployment in DataParallelDeployment.__mro__
    assert hasattr(DataParallelDeployment, 'data_world_size')
    assert not hasattr(QueuedDeployment, 'data_world_size')
