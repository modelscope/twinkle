# Copyright (c) ModelScope Contributors. All rights reserved.
"""zmq PUB/SUB manifest-channel helpers."""
import time

import pytest
import zmq

from twinkle.checkpoint_engine.zmq_pubsub import connect_subscriber, format_tcp_address, start_publisher
from twinkle.utils.network import find_free_port


def test_format_tcp_address():
    assert format_tcp_address('10.0.0.1', 5555) == ('tcp://10.0.0.1:5555', False)
    assert format_tcp_address('::1', 5555) == ('tcp://[::1]:5555', True)


def test_loopback_pub_sub():
    port = find_free_port()
    try:
        pub = start_publisher('127.0.0.1', port)
    except zmq.ZMQError as e:
        pytest.skip(f'zmq loopback unavailable: {e}')
    sub = connect_subscriber('127.0.0.1', port, 'topic')
    try:
        sub.setsockopt(zmq.RCVTIMEO, 100)
        got = None
        deadline = time.time() + 5
        while got is None and time.time() < deadline:  # tolerate slow joiner
            pub.send_string('topic', flags=zmq.SNDMORE)
            pub.send_pyobj({'x': 1})
            try:
                sub.recv_string()
                got = sub.recv_pyobj()
            except zmq.error.Again:
                pass
        assert got == {'x': 1}
    finally:
        sub.close(linger=0)
        pub.close(linger=0)
        sub.context.term()
        pub.context.term()
