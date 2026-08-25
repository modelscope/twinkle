# Copyright (c) ModelScope Contributors. All rights reserved.
"""Reap paused sandboxes on the environment host, which is what keeps it alive.

AgentENV does not discard a sandbox when it ends: it *persists* it, as a paused
sandbox whose memory and disk image live under
``/var/lib/aenv/persisted-sandboxes/artifacts`` at roughly 1GB each. Closing the
sandbox from the client does not change this -- a closed sandbox is a paused
sandbox -- so every episode leaks a gigabyte. A GRPO step that boots
``batch_size x num_generations`` sandboxes leaks that many, and a 40GB root
filesystem is gone in a couple of dozen steps. The failure is not graceful: boots
start returning ``500: backend error: ... No space left on device``, and every
episode in the batch scores zero, which reads like a hard task rather than a
broken host.

Run this on the environment host for the length of a training run::

    setsid nohup python3 reap_paused.py --alias twinkle-rsi-msagent \\
        > /var/log/reap.log 2>&1 &

Only *paused* sandboxes with the given alias are deleted. A running one may be an
episode in flight, and a different alias belongs to a different experiment --
this script never touches either.
"""
import argparse
import json
import subprocess
import time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--alias', default='twinkle-rsi-msagent',
                   help='only reap sandboxes built from this template')
    p.add_argument('--interval', type=int, default=120,
                   help='seconds between sweeps; 0 sweeps once and exits')
    return p.parse_args()


def sweep(alias):
    """Delete every paused sandbox with this alias. Returns (reaped, running)."""
    proc = subprocess.run(['aenv', 'list'], capture_output=True, text=True)
    try:
        rows = json.loads(proc.stdout)
    except ValueError:
        # The server restarting mid-sweep is not worth dying over; the next
        # sweep will see the same sandboxes.
        return 0, -1
    paused = [r['sandboxID'] for r in rows
              if r.get('state') == 'paused' and r.get('alias') == alias]
    for sandbox_id in paused:
        subprocess.run(['aenv', 'delete', sandbox_id], capture_output=True)
    running = sum(1 for r in rows if r.get('state') == 'running')
    return len(paused), running


def main():
    args = parse_args()
    while True:
        reaped, running = sweep(args.alias)
        disk = subprocess.run(['df', '-h', '/'], capture_output=True,
                              text=True).stdout.splitlines()[-1].split()
        stamp = time.strftime('%H:%M:%S')
        print(f'{stamp} reaped={reaped:3d} running={running:3d} '
              f'free={disk[3]} used={disk[4]}', flush=True)
        if args.interval <= 0:
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
