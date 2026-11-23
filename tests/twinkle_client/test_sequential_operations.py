# Copyright (c) ModelScope Contributors. All rights reserved.
import threading
from concurrent.futures import Future

import pytest

from twinkle_client.operations import SequentialOperations


def test_repeated_wait_interrupts_still_record_success_once(monkeypatch):
    operations = SequentialOperations()
    recorded = []

    class InterruptedWait(Future):
        remaining = 2

        def result(self):
            if self.remaining:
                self.remaining -= 1
                raise KeyboardInterrupt
            return super().result()

    monkeypatch.setattr('twinkle_client.operations.Future', InterruptedWait)
    try:
        with pytest.raises(KeyboardInterrupt):
            operations.run(lambda: 'saved', on_result=recorded.append)
        assert recorded == ['saved']
    finally:
        operations.close()


def test_interrupt_during_submit_waits_for_an_already_started_worker(monkeypatch):
    operations = SequentialOperations()
    started, finish = threading.Event(), threading.Event()
    recorded = []
    start_thread = threading.Thread.start

    class Outcome(Future):
        def result(self):
            finish.set()
            return super().result()

    def start_and_interrupt(thread):
        start_thread(thread)
        assert started.wait(5)
        # The worker can execute before ThreadPoolExecutor registers it for shutdown.
        raise KeyboardInterrupt

    def save():
        started.set()
        assert finish.wait(5)
        return 'saved'

    monkeypatch.setattr('twinkle_client.operations.Future', Outcome)
    monkeypatch.setattr(threading.Thread, 'start', start_and_interrupt)
    try:
        with pytest.raises(KeyboardInterrupt):
            operations.run(save, on_result=recorded.append)
        assert recorded == ['saved']
    finally:
        finish.set()
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
