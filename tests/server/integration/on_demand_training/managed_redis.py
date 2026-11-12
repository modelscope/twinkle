# Copyright (c) ModelScope Contributors. All rights reserved.
"""An owned Redis subprocess for safe, durable restart fault injection."""
import socket
import subprocess
import time
from pathlib import Path


class ManagedRedis:
    """Never stop an externally supplied Redis server."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            self.port = listener.getsockname()[1]
        self.url = f'redis://127.0.0.1:{self.port}/0'
        self.process = None
        self.log = None

    def start(self):
        import redis
        assert self.process is None
        self.log = (self.directory / 'redis.log').open('a')
        self.process = subprocess.Popen([
            'redis-server', '--bind', '127.0.0.1', '--port', str(self.port),
            '--dir', str(self.directory), '--appendonly', 'yes', '--appendfsync', 'always',
            '--save', '', '--daemonize', 'no'], stdout=self.log, stderr=subprocess.STDOUT)
        client = redis.Redis.from_url(self.url, socket_timeout=1, socket_connect_timeout=1)
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError('Owned test Redis exited during startup')
                try:
                    if client.ping():
                        return
                except redis.exceptions.ConnectionError:
                    pass
                time.sleep(0.1)
            raise TimeoutError('Owned test Redis did not start')
        finally:
            client.close()

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
                raise
            finally:
                self.log.close()
                self.process = None
                self.log = None

    def fault(self, probe):
        self.stop()
        try:
            probe()
        finally:
            self.start()
