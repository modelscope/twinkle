# Copyright (c) ModelScope Contributors. All rights reserved.
"""Sequential blocking calls whose results remain owned across caller interrupts."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any


@dataclass
class _Outcome:
    value: Any = None
    error: BaseException | None = None


def _invoke(operation, args, kwargs, on_result):
    # A worker's KeyboardInterrupt is an outcome, not an interruption of the wait.
    try:
        value = operation(*args, **kwargs)
        if on_result is not None:
            on_result(value)
        return _Outcome(value=value)
    except BaseException as error:
        return _Outcome(error=error)


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

    def run(self, operation, *args, on_result=None, **kwargs):
        if self._stopped:
            raise RuntimeError('operation sequence is stopped')
        future = self._executor.submit(_invoke, operation, args, kwargs, on_result)
        control_error = None
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
