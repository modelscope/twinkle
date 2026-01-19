#!/usr/bin/env python
"""Test script for Ray mode with various parallelism configurations.

Records loss, memory usage, and training time.
"""
import os
import sys
import time
import subprocess
import re

# Test configurations: (tp_size, pp_size, num_gpus, name)
CONFIGS = [
    (2, 2, 4, "TP=2_PP=2"),
    (4, 1, 4, "TP=4_PP=1"),
    (1, 4, 4, "TP=1_PP=4"),
    (2, 1, 2, "TP=2_PP=1"),
]

MODEL = "ms://Qwen/Qwen2.5-0.5B-Instruct"
MAX_STEPS = 5
TIMEOUT = 600  # 10 minutes per test

def run_test(mode, tp_size, pp_size, num_gpus, name):
    """Run a single test configuration."""
    env = os.environ.copy()
    env["MEGATRON_LM_PATH"] = "/mnt/nas2/hujinghan.hjh/Megatron-LM"
    env["PYTHONPATH"] = "/mnt/nas2/hujinghan.hjh/Megatron-LM:/mnt/nas2/hujinghan.hjh/twinkle/src:" + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))
    env["TRUST_REMOTE_CODE"] = "1"
    
    log_file = f"/mnt/nas2/hujinghan.hjh/twinkle/test_{mode}_{name}.log"
    
    if mode == "ray":
        cmd = [
            "/mnt/nas2/anaconda3/envs/hjh/bin/python",
            "cookbook/megatron/lora.py",
            "--mode", "ray",
            "--tp_size", str(tp_size),
            "--pp_size", str(pp_size),
            "--num_gpus", str(num_gpus),
            "--model", MODEL,
            "--max_steps", str(MAX_STEPS),
        ]
    else:
        # Find an available port
        import socket
        with socket.socket() as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
        
        cmd = [
            "/mnt/nas2/anaconda3/envs/hjh/bin/python", "-m", "torch.distributed.run",
            "--nproc_per_node", str(num_gpus),
            "--master_port", str(port),
            "cookbook/megatron/lora.py",
            "--tp_size", str(tp_size),
            "--pp_size", str(pp_size),
            "--model", MODEL,
            "--max_steps", str(MAX_STEPS),
        ]
    
    print(f"\n{'='*60}")
    print(f"Running: {mode} mode, {name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Log: {log_file}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    with open(log_file, "w") as f:
        try:
            result = subprocess.run(
                cmd,
                cwd="/mnt/nas2/hujinghan.hjh/twinkle",
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=TIMEOUT,
            )
            success = result.returncode == 0
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT after {TIMEOUT}s")
            success = False
        except Exception as e:
            print(f"  ERROR: {e}")
            success = False
    
    elapsed = time.time() - start_time
    
    # Parse results from log
    losses = []
    memory = None
    
    with open(log_file, "r") as f:
        content = f.read()
        
        # Extract losses
        for match in re.finditer(r"Step (\d+), loss: ([\d.]+)", content):
            step = int(match.group(1))
            loss = float(match.group(2))
            losses.append((step, loss))
        
        # Check for completion
        completed = "Training completed!" in content
    
    return {
        "mode": mode,
        "config": name,
        "tp": tp_size,
        "pp": pp_size,
        "gpus": num_gpus,
        "losses": losses,
        "elapsed": elapsed,
        "success": success and completed,
        "log_file": log_file,
    }


def cleanup():
    """Kill any lingering processes."""
    os.system("pkill -9 -f 'lora.py|MegatronModel|ray' 2>/dev/null")
    time.sleep(5)


def main():
    results = []
    
    for tp, pp, gpus, name in CONFIGS:
        # Test Ray mode
        cleanup()
        ray_result = run_test("ray", tp, pp, gpus, name)
        results.append(ray_result)
        
        # Test Local mode
        cleanup()
        local_result = run_test("local", tp, pp, gpus, name)
        results.append(local_result)
    
    cleanup()
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Mode':<8} {'Config':<15} {'GPUs':<6} {'Status':<10} {'Time(s)':<10} {'Step0 Loss':<12} {'Step5 Loss':<12}")
    print("-"*80)
    
    for r in results:
        status = "✅ OK" if r["success"] else "❌ FAIL"
        step0_loss = r["losses"][0][1] if len(r["losses"]) > 0 else "N/A"
        step5_loss = r["losses"][-1][1] if len(r["losses"]) > 5 else "N/A"
        if isinstance(step0_loss, float):
            step0_loss = f"{step0_loss:.4f}"
        if isinstance(step5_loss, float):
            step5_loss = f"{step5_loss:.4f}"
        print(f"{r['mode']:<8} {r['config']:<15} {r['gpus']:<6} {status:<10} {r['elapsed']:<10.1f} {step0_loss:<12} {step5_loss:<12}")
    
    print("="*80)
    
    # Save results to file
    with open("/mnt/nas2/hujinghan.hjh/twinkle/test_results.txt", "w") as f:
        f.write("Ray Mode Parallelism Test Results\n")
        f.write("="*80 + "\n\n")
        for r in results:
            f.write(f"Mode: {r['mode']}, Config: {r['config']}, GPUs: {r['gpus']}\n")
            f.write(f"Success: {r['success']}, Time: {r['elapsed']:.1f}s\n")
            f.write(f"Losses: {r['losses']}\n")
            f.write(f"Log: {r['log_file']}\n")
            f.write("-"*40 + "\n")


if __name__ == "__main__":
    main()
