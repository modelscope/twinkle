# Copyright (c) ModelScope Contributors. All rights reserved.
import contextvars
import functools
import os
import threading
import weakref


_PARENT_WRITERS = set()
_OWNER = contextvars.ContextVar('twinkle_engine_core_owner', default=None)
_INSTALL_LOCK = threading.Lock()


def _close_inherited_writers():
    # Forked children must not keep another process's parent-death pipe alive.
    for writer in tuple(_PARENT_WRITERS):
        writer.close()
    _PARENT_WRITERS.clear()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_close_inherited_writers)


def _close_writers(writers):
    for writer in writers:
        _PARENT_WRITERS.discard(writer)
        writer.close()


def _run_with_parent_watch(reader, target, args, kwargs):
    """Start the parent-death monitor before initializing the engine core."""
    finished = threading.Event()

    def monitor():
        try:
            reader.recv()
        except (EOFError, OSError):
            if not finished.is_set():
                os._exit(1)

    threading.Thread(target=monitor, daemon=True, name='engine-parent-watch').start()
    try:
        return target(*args, **kwargs)
    finally:
        finished.set()
        reader.close()


class _OwnedProcessContext:
    """Add a one-way lifetime pipe to processes created by one engine manager."""

    def __init__(self, context, pipes):
        self.context = context
        self.pipes = pipes

    def __getattr__(self, name):
        return getattr(self.context, name)

    def Process(self, *, target, args=(), kwargs=None, **options):
        reader, writer = self.context.Pipe(duplex=False)
        _PARENT_WRITERS.add(writer)
        self.pipes.append((reader, writer))
        return self.context.Process(
            target=_run_with_parent_watch, args=(reader, target, args, kwargs or {}), **options)


def install_engine_core_parent_watch():
    """Guard vLLM-owned EngineCore processes without modifying installed package files."""
    from vllm.v1.engine import utils

    with _INSTALL_LOCK:
        manager = utils.CoreEngineProcManager
        if getattr(manager, '_twinkle_parent_watch', False):
            return
        get_context = utils.get_mp_context
        initialize = manager.__init__
        shutdown = manager.shutdown

        @functools.wraps(get_context)
        def owned_context():
            context = get_context()
            pipes = _OWNER.get()
            return context if pipes is None else _OwnedProcessContext(context, pipes)

        @functools.wraps(initialize)
        def initialize_owned(self, *args, **kwargs):
            pipes = []
            token = _OWNER.set(pipes)
            try:
                initialize(self, *args, **kwargs)
            except BaseException:
                _close_writers([writer for _, writer in pipes])
                raise
            else:
                self._twinkle_parent_pipe_finalizer = weakref.finalize(
                    self, _close_writers, [writer for _, writer in pipes])
            finally:
                _OWNER.reset(token)
                for reader, _ in pipes:
                    reader.close()

        @functools.wraps(shutdown)
        def shutdown_owned(self, *args, **kwargs):
            try:
                return shutdown(self, *args, **kwargs)
            finally:
                finalizer = getattr(self, '_twinkle_parent_pipe_finalizer', None)
                if finalizer is not None:
                    finalizer()

        utils.get_mp_context = owned_context
        manager.__init__ = initialize_owned
        manager.shutdown = shutdown_owned
        manager._twinkle_parent_watch = True
