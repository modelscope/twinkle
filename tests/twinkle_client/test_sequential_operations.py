# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest

from twinkle_client.operations import SequentialOperations


def test_repeated_wait_interrupts_still_record_success_once():
    operations = SequentialOperations()
    recorded = []
    original = operations._executor.submit

    class Wait:
        def __init__(self, future):
            self.future, self.remaining = future, 2

        def result(self):
            if self.remaining:
                self.remaining -= 1
                raise KeyboardInterrupt()
            return self.future.result()

        def done(self):
            return True

    operations._executor.submit = lambda *a, **k: Wait(original(*a, **k))
    try:
        with pytest.raises(KeyboardInterrupt):
            operations.run(lambda: 'saved', on_result=recorded.append)
        assert recorded == ['saved']
    finally:
        operations.close()


def test_worker_control_exception_is_terminal():
    operations = SequentialOperations()
    try:
        def operation():
            raise KeyboardInterrupt('worker failure')
        with pytest.raises(KeyboardInterrupt, match='worker failure'):
            operations.run(operation)
    finally:
        operations.close()
