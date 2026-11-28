# Copyright (c) ModelScope Contributors. All rights reserved.
"""Sequential blocking calls whose results remain owned across caller interrupts."""
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import wraps
from typing import Any


@dataclass
class _Outcome:
    value: Any = None
    error: BaseException | None = None


def _invoke(operation, args, kwargs, on_result, worker):
    # A worker's KeyboardInterrupt is an outcome, not an interruption of the wait.
    worker.active = True
    try:
        value = operation(*args, **kwargs)
        if on_result is not None:
            on_result(value)
        return _Outcome(value=value)
    except BaseException as error:
        return _Outcome(error=error)
    finally:
        worker.active = False


class SequentialOperations:
    """Keep one blocking call alive until its outcome is recorded, then propagate interrupts.

    ``on_result`` records successful side effects (for example a new checkpoint or
    optimizer step) even if the caller was interrupted while waiting. Calls remain
    sequential. The short ``on_result`` callback runs on the worker before completion;
    it must not touch thread-affine UI/event-loop state. This helper provides no
    remote cancellation or completion guarantee.
    """

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._stopped = False
        self._worker = threading.local()

    def run(self, operation, *args, on_result=None, **kwargs):
        if self._stopped:
            raise RuntimeError('operation sequence is stopped')
        future = Future()
        start_lock = threading.Lock()

        def execute():
            # A submit interrupted during thread startup may already have queued this call.
            with start_lock:
                if future.running() or future.done() or not future.set_running_or_notify_cancel():
                    return
            future.set_result(_invoke(operation, args, kwargs, on_result, self._worker))

        control_error = None
        while True:
            try:
                self._executor.submit(execute)
                break
            except (KeyboardInterrupt, SystemExit) as error:
                self._stopped = True
                if control_error is None:
                    control_error = error
                if future.running() or future.done():
                    break
            except BaseException as error:
                self._stopped = True
                if future.cancel():
                    try:
                        self.close()
                    except BaseException as cleanup_error:
                        raise error from cleanup_error
                    raise
                if control_error is None:
                    control_error = error
                break

        while True:
            try:
                outcome = future.result()
            except (KeyboardInterrupt, SystemExit) as error:
                self._stopped = True
                if control_error is None:
                    control_error = error
                continue
            break
        error = outcome.error
        if error is not None:
            self._stopped = True
        if control_error is not None:
            raise control_error from error
        if error is not None:
            raise error
        return outcome.value

    def close(self):
        self._stopped = True
        control_error = None
        while True:
            try:
                self._executor.shutdown(wait=True)
            except (KeyboardInterrupt, SystemExit) as error:
                if control_error is None:
                    control_error = error
                continue
            break
        if control_error is not None:
            raise control_error


class _OperationScope:
    """Guard one client's training/control operations, including local result registration.

    Bound explicitly by RolloutResources; ordinary clients remain unchanged. Nested
    calls run inline on the same worker, so an adapter's handle is registered before
    a caller interrupt is raised. Sampling stays under the rollout's own execution
    tracking and is never serialized through this sequence.
    """

    def __init__(self, operations, close):
        self.operations = operations
        self.close = close
        self._training_releases = {}
        self._lock = threading.RLock()
        self._local = threading.local()
        self.uncertain_errors = []

    def note_request(self, path):
        if getattr(self._local, 'active', False):
            self._local.request_path = path

    def _invoke(self, operation, args, kwargs, *, track_uncertainty=True):
        from requests import HTTPError

        from .exceptions import TaskCancelledError

        self._local.active, self._local.request_path = True, None
        try:
            return operation(*args, **kwargs)
        except BaseException as error:
            # Cancellation confirms a pending task was dropped. A submission 4xx
            # rejects the request; a polling 4xx leaves its earlier execution unknown.
            # Other errors after sending (including a failed task record) do not
            # certify physical settlement of remote execution.
            status = getattr(error.response, 'status_code', 0) if isinstance(error, HTTPError) else 0
            path = self._local.request_path
            rejected = (path is not None and 400 <= (status or 0) < 500 and status not in (408, 499)
                        and not path.rstrip('/').endswith('/retrieve_future'))
            if track_uncertainty and path is not None and not rejected and not isinstance(error, TaskCancelledError):
                self.uncertain_errors.append(error)
            raise
        finally:
            self._local.active = False

    def run(self, operation, *args, **kwargs):
        if getattr(self._local, 'active', False):
            return operation(*args, **kwargs)
        # Also support callers still using the previous explicit operations.run API.
        if getattr(self.operations._worker, 'active', False):
            return self._invoke(operation, args, kwargs)
        with self._lock:
            return self.operations.run(self._invoke, operation, args, kwargs)

    def own_training(self, training_id, release):
        self._training_releases[training_id] = release

    def forget_training(self, training_id):
        self._training_releases.pop(training_id, None)

    def release_trainings(self):
        """Release settled handles after the regular operation sequence has drained."""
        errors = []
        for release in tuple(self._training_releases.values()):
            try:
                # The owned handle stays registered on failure. Retrying its release is safe.
                self._invoke(release, (), {}, track_uncertainty=False)
            except BaseException as error:
                errors.append(error)
        if errors:
            raise BaseExceptionGroup('training handle release is unconfirmed', errors)


def _guarded_operation(operation):
    """Use the owning scope at SDK boundaries; preserve unscoped call semantics."""
    @wraps(operation)
    def guarded(self, *args, **kwargs):
        scope = getattr(self._transport, '_operation_scope', None)
        if isinstance(scope, _OperationScope):
            return scope.run(operation, self, *args, **kwargs)
        return operation(self, *args, **kwargs)
    return guarded
