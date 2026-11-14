# Copyright (c) ModelScope Contributors. All rights reserved.
"""Scoped ownership of a multi-turn rollout's client and execution environments."""
import threading
import traceback

from .operations import SequentialOperations, _OperationScope


class RolloutResources:
    """Create a client and release owned resources after rollout work settles.

    ``with RolloutResources(client_factory) as resources`` creates the client through
    ``operations``, recording ownership even if the caller is interrupted. A real
    TwinkleClient is bound to this scope: ordinary model calls and sampler setup
    automatically wait through interrupts and register results before raising.
    Model methods expose completed_step_calls and last_checkpoint for local accounting.
    Sampling keeps the rollout's concurrent execution path. An interrupted or failed
    managed call stops subsequent managed calls; unscoped clients are unchanged.

    Register the rollout (or the driver running it) with ``own_rollout`` and the Env
    components it uses with ``own_environment``. This scope closes them if setup fails too.

    Exit waits for issued operations, closes the rollout (its episodes and their env
    sessions), then shuts the environments down and closes the client. Unknown execution
    or failed closes retain their resource handles on this scope and raise; they are never
    treated as successful release. Cleanup runs off the caller thread so repeated
    interrupts cannot skip later resources.
    """

    def __init__(self, client_factory, **client_kwargs):
        self.operations = SequentialOperations()
        self._operation_scope = _OperationScope(self.operations)
        self.client = None
        self._client_factory = client_factory
        self._client_kwargs = client_kwargs
        self._rollout = None
        self._environments = {}
        self._lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._accepting = True
        self._entered = False
        self._closed = False

    def __enter__(self):
        with self._lock:
            if self._entered or not self._accepting:
                raise RuntimeError('resource scope cannot be entered again')
            self._entered = True

        def own_client(client):
            from .manager import TwinkleClient
            if isinstance(client, TwinkleClient):
                if client.transport._operation_scope is not None:
                    raise ValueError('client already belongs to a resource scope')
                client.transport._operation_scope = self._operation_scope
            self.client = client

        try:
            self.operations.run(self._client_factory, on_result=own_client, **self._client_kwargs)
        except BaseException as error:
            self.__exit__(type(error), error, error.__traceback__)
            raise
        return self

    def own_rollout(self, rollout):
        """Transfer the rollout (anything with ``close()``) to this scope; return it."""
        with self._lock:
            if not self._accepting:
                raise RuntimeError('resource scope is closing')
            if self._rollout is not None and self._rollout is not rollout:
                raise ValueError('use a separate resource scope for each rollout/client')
            self._rollout = rollout
        return rollout

    def own_environment(self, env):
        """Register an Env component; exit shuts it down after the rollout is closed."""
        with self._lock:
            if not self._accepting:
                raise RuntimeError('resource scope is closing')
            self._environments[id(env)] = env
        return env

    def _finish(self, issues):
        def attempt(operation):
            try:
                operation()
            except BaseException as error:
                issues.append(error)

        attempt(self.operations.close)
        if self._rollout is not None:
            attempt(self._rollout.close)
        if self._operation_scope.uncertain_errors:
            issues.append(BaseExceptionGroup(
                'remote operation settlement is unconfirmed; environments and client retained',
                self._operation_scope.uncertain_errors))
            return
        with self._lock:
            environments = list(self._environments.values())
        for env in environments:
            def shutdown(env=env):
                env.shutdown()
                with self._lock:
                    self._environments.pop(id(env))

            attempt(shutdown)
        if self.client is not None:
            def close_client():
                self.client.close()
                self.client = None

            attempt(close_client)

    def close(self):
        """Attempt all safe releases, once at a time, despite repeated interrupts."""
        with self._close_lock:
            if self._closed:
                return
            with self._lock:
                self._accepting = False
            issues = []
            cleanup = SequentialOperations()
            try:
                cleanup.run(self._finish, issues)
            except BaseException as error:
                issues.append(error)
            finally:
                try:
                    cleanup.close()
                except BaseException as error:
                    issues.append(error)
            if issues:
                raise BaseExceptionGroup('rollout resource cleanup failed', issues)
            self._closed = True

    def __exit__(self, exc_type, error, tb):
        try:
            self.close()
        except BaseException as cleanup_error:
            if error is None:
                raise
            error.add_note('resource cleanup:\n' + ''.join(traceback.format_exception(cleanup_error)))
