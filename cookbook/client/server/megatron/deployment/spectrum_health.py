"""Platform liveness for each rank and business readiness for the head."""
import os
import socket
import sys
import time
from http.client import HTTPConnection, HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def head_listening():
    try:
        with socket.create_connection((os.environ['MASTER_ADDRESS'], 6379), timeout=1):
            return True
    except OSError:
        return False


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/healthz':
            status = 200 if head_listening() else 503
        elif self.path == '/readyz' and os.environ['RANK_ID'] == '0':
            conn = HTTPConnection('127.0.0.1', 8000, timeout=2)
            try:
                conn.request('GET', '/api/v1/twinkle/healthz')
                status = 200 if conn.getresponse().status == 200 else 503
            except (OSError, HTTPException):
                status = 503
            finally:
                conn.close()
        else:
            status = 404
        self.send_response(status)
        self.end_headers()
        self.wfile.write(b'ok\n' if status == 200 else b'unavailable\n')

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    if sys.argv[1:] == ['--wait-for-head']:
        while not head_listening():
            time.sleep(2)
    else:
        ThreadingHTTPServer(('0.0.0.0', int(os.environ.get('PORT', '9000'))), HealthHandler).serve_forever()
