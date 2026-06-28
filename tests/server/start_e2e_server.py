"""One-click: restart Ray cluster + launch Twinkle server + wait until ready.

Usage:
    python start_e2e_server.py                         # default config
    python start_e2e_server.py --config my_config.yaml # custom config
    python start_e2e_server.py --kill-only             # just kill everything
"""
import argparse
import os
import signal
import subprocess
import sys
import time

import requests

# ── Paths ──
RAY = "/mnt/nas2/anaconda3/envs/tinker_myl/bin/ray"
PYTHON = "/mnt/nas2/anaconda3/envs/tinker_myl/bin/python"
WORKDIR = "/mnt/nas2/yunlin.myl/twinkle"
DEFAULT_CONFIG = "tests/server/config/server_config_4b_e2e.yaml"
RAY_TEMP_DIR = "/mnt/nas2/yunlin.myl/ray_logs"
SERVER_LOG = os.path.join(WORKDIR, "server_e2e.log")

# ── Server check ──
SERVER_URL = "http://localhost:9000/-/routes"
READY_KEYWORD = "processor"
TIMEOUT = 600
POLL_INTERVAL = 5


def run(cmd, env=None, check=True):
    """Run a shell command, print it, and return CompletedProcess."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, env=full_env,
                            capture_output=True, text=True)
    if result.returncode != 0 and check:
        print(f"    STDERR: {result.stderr.strip()}")
    return result


def kill_server():
    """Kill any existing twinkle.server processes."""
    print("[1/4] Killing existing Twinkle server processes...")
    result = subprocess.run(
        "ps aux | grep 'twinkle.server' | grep -v grep | awk '{print $2}'",
        shell=True, capture_output=True, text=True
    )
    pids = result.stdout.strip().split()
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGKILL)
            print(f"    Killed PID {pid}")
        except (ProcessLookupError, ValueError):
            pass
    if not pids:
        print("    No existing server found")
    time.sleep(2)


def restart_ray():
    """Stop and restart the Ray cluster (5 GPU + 1 CPU nodes)."""
    print("[2/4] Restarting Ray cluster...")
    run(f"{RAY} stop --force", check=False)
    time.sleep(2)

    # Head node: GPU 0,1,2,3 (4 GPUs for model PP=2 x DP=2)
    run(f"{RAY} start --head --port=6379 --num-gpus=4 "
        f"--disable-usage-stats --temp-dir={RAY_TEMP_DIR}",
        env={"CUDA_VISIBLE_DEVICES": "0,1,2,3"})

    # Worker: GPU 4 (1 GPU for sampler)
    run(f"{RAY} start --address=127.0.0.1:6379 --num-gpus=1",
        env={"CUDA_VISIBLE_DEVICES": "4"})

    # CPU-only worker (processor + server)
    run(f"{RAY} start --address=127.0.0.1:6379 --num-gpus=0",
        env={"CUDA_VISIBLE_DEVICES": ""})

    print("    Ray cluster started (4+1+0 GPUs)")


def launch_server(config: str):
    """Launch the Twinkle server in the background."""
    print(f"[3/4] Launching Twinkle server (config={config})...")
    config_path = os.path.join(WORKDIR, config)
    if not os.path.isfile(config_path):
        print(f"    ERROR: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    cmd = f"{PYTHON} -m twinkle.server launch --config {config_path}"
    log_fd = open(SERVER_LOG, "w")
    subprocess.Popen(
        cmd, shell=True, cwd=WORKDIR, stdout=log_fd, stderr=log_fd,
        env={**os.environ, "TWINKLE_TRUST_REMOTE_CODE": "1"},
        start_new_session=True,
    )
    print(f"    Server starting (log: {SERVER_LOG})")


def wait_ready():
    """Poll until the server is fully ready."""
    print(f"[4/4] Waiting for server to be ready (timeout={TIMEOUT}s)...")
    start = time.time()
    while time.time() - start < TIMEOUT:
        try:
            resp = requests.get(SERVER_URL, timeout=3)
            if resp.ok and READY_KEYWORD in resp.text:
                elapsed = time.time() - start
                print(f"    Server READY ({elapsed:.0f}s)")
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(POLL_INTERVAL)

    print(f"    TIMEOUT: server not ready after {TIMEOUT}s", file=sys.stderr)
    print(f"    Check log: tail -50 {SERVER_LOG}", file=sys.stderr)
    return False


def main():
    parser = argparse.ArgumentParser(description="Restart Ray + launch Twinkle server")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help=f"Server config yaml (default: {DEFAULT_CONFIG})")
    parser.add_argument("--kill-only", action="store_true",
                        help="Only kill server + Ray, don't restart")
    parser.add_argument("--no-ray", action="store_true",
                        help="Skip Ray restart (server only)")
    args = parser.parse_args()

    kill_server()

    if args.kill_only:
        run(f"{RAY} stop --force", check=False)
        print("Done (kill-only)")
        return 0

    if not args.no_ray:
        restart_ray()

    launch_server(args.config)

    if wait_ready():
        print("\n✓ All set. Run your test:")
        print(f"  TWINKLE_TEST_GPU_E2E=1 {PYTHON} -u tests/server/integration/test_full_cycle_e2e.py")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
