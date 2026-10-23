# Copyright (c) ModelScope Contributors. All rights reserved.
"""Concurrent and soak validation against an already running single-node server."""
import argparse
import concurrent.futures
import json
import requests
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.server.integration.on_demand_training.acceptance_support import TrainingHandle, assert_released


def run_stress(base_url, model, concurrency=2, iterations=100, duration=0, controller=None,
               namespace=None, redis_url=None, prefix=None, progress_path=None, protocols=('twinkle', 'tinker')):
    """Run actual training; duration mode is never truncated by an iteration limit."""
    started, completed = time.monotonic(), 0
    evidence = {'completed': 0, 'failures': [], 'passed': False, 'duration_target': duration}

    def cycle(protocol, index):
        handle = TrainingHandle(base_url, model, protocol, 'stress-' + str(index))
        try:
            handle.step()
            if (index // len(protocols)) % 10 == 0:
                path = handle.save('soak-' + str(index))
                handle.resume(path)
                handle.step()
        finally:
            handle.close()

    # Keep a peer alive so cycles test slot reuse instead of spending hours
    # repeatedly loading the same base model. Its final release must free the PG.
    anchor = TrainingHandle(base_url, model, protocols[0], 'soak-anchor')
    try:
        anchor.step()
        # Measure sustained traffic time, not model cold-start/JIT time.
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            while (time.monotonic() - started < duration) if duration else (completed < iterations):
                count = concurrency if duration else min(concurrency, iterations - completed)
                jobs = [executor.submit(cycle, protocols[(completed + slot) % len(protocols)], completed + slot)
                        for slot in range(count)]
                for job in jobs:
                    job.result()  # Do not convert incomplete cycles into successful evidence.
                completed += count
                # Session heartbeats do not renew the business-idle lease.
                # Keep the warm peer active during a multi-hour soak.
                if completed % 10 == 0:
                    anchor.step()
                if controller is not None:
                    import ray
                    live = [row for row in ray.get(controller.diagnostics.remote(), timeout=30)['instances']
                            if row['state'] != 'STOPPED']
                    assert len(live) == 1 and live[0]['state'] == 'READY'
                    assert live[0]['training_ids'] == [anchor.identity], live
                    assert len(live[0]['devices']) == 2
                evidence.update(completed=completed, seconds=time.monotonic() - started)
                if progress_path:
                    Path(progress_path).write_text(json.dumps(evidence, indent=2))
                print(json.dumps({'phase': 'stress', **evidence}), flush=True)
    finally:
        anchor.close()
    if controller is not None:
        assert_released(controller, namespace, redis_url, prefix)
    evidence['passed'] = True
    if duration:
        assert evidence['seconds'] >= duration
    if progress_path:
        Path(progress_path).write_text(json.dumps(evidence, indent=2))
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', required=True, choices=['transformers', 'megatron'])
    parser.add_argument('--client', choices=['twinkle', 'tinker', 'both'], default='both')
    parser.add_argument('--server-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--concurrency', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--duration', type=float, default=0, help='Run for this many seconds instead of fixed iterations')
    parser.add_argument('--results', required=True)
    args = parser.parse_args()
    if args.concurrency <= 0 or args.iterations <= 0 or args.duration < 0:
        parser.error('concurrency and iterations must be positive; duration must be non-negative')

    protocols = ['twinkle', 'tinker'] if args.client == 'both' else [args.client]
    evidence = run_stress(args.server_url, args.model, args.concurrency, args.iterations, args.duration,
                          progress_path=args.results, protocols=protocols)
    response = requests.get(args.server_url + '/api/v1/model/' + args.model + '/healthz', timeout=30)
    response.raise_for_status()
    assert response.json()['healthy'] and not response.json()['instances']
    print(json.dumps(evidence), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
