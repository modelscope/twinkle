"""Multi-client race condition stress test: reproduce ncclCommSplit hang.

Strategy: 3 concurrent clients hammer the Megatron server in a loop:
- Client A: continuous DPO training (forward_only + forward_backward + step)
- Client B: continuous DPO training with occasional bad data (triggers errors)
- Client C: repeatedly creates/destroys adapters (triggers ncclCommSplit)

Runs multiple rounds. If any operation takes >TIMEOUT seconds, NCCL hang is detected.

Usage:
    python tests/server/integration/test_race_nccl_hang.py
    python tests/server/integration/test_race_nccl_hang.py --rounds 5
"""
import argparse
import os
import sys
import threading
import time

os.environ['TINKER_BASE_URL'] = 'http://localhost:9000'
os.environ['TWINKLE_SERVER_TOKEN'] = 'EMPTY_TOKEN'

from twinkle_client import init_twinkle_client
from twinkle_client.model import MultiLoraTransformersModel
from peft import LoraConfig

SERVER_URL = os.environ.get('TWINKLE_SERVER_URL', 'http://localhost:9000')
TIMEOUT = 90
seq_len = 64


def make_batch(size=4, include_position_ids=True):
    batch = []
    for _ in range(size):
        item = {
            'input_ids': list(range(1, seq_len + 1)),
            'labels': [-100] * 32 + list(range(100, 132)),
            'attention_mask': [1] * seq_len,
        }
        if include_position_ids:
            item['position_ids'] = list(range(seq_len))
        batch.append(item)
    return batch


results = {'hangs': [], 'errors': [], 'steps_ok': 0, 'rounds_ok': 0}
lock = threading.Lock()
stop_event = threading.Event()


def log(msg):
    print(f'[RACE] {msg}', flush=True)


def create_session(name):
    init_twinkle_client(base_url=SERVER_URL, api_key='EMPTY_TOKEN')
    model = MultiLoraTransformersModel(model_id='ms://Qwen/Qwen3.5-4B')
    model.add_adapter_to_model(
        adapter_name=name,
        config=LoraConfig(r=16, target_modules=['q_proj', 'v_proj']),
        gradient_accumulation_steps=1,
    )
    model.set_loss('DPOLoss', init_args={'beta': 0.1})
    model.set_optimizer('Adam', lr=1e-5)
    model.set_template('Qwen3_5Template')
    model.set_processor('InputProcessor', padding_side='right')
    return model


def client_a_training(steps_per_round):
    """Client A: continuous normal DPO training."""
    try:
        model = create_session('client-a')
        log('Client-A: session ready')
        batch = make_batch(4)
        for i in range(steps_per_round):
            if stop_event.is_set():
                return
            start = time.time()
            model.forward_only(inputs=batch, disable_lora=True)
            model.forward_backward(inputs=batch)
            model.clip_grad_and_step()
            elapsed = time.time() - start
            with lock:
                results['steps_ok'] += 1
            if i % 3 == 0:
                log(f'Client-A: step {i + 1}/{steps_per_round} ({elapsed:.1f}s)')
            if elapsed > TIMEOUT:
                with lock:
                    results['hangs'].append(f'Client-A step {i + 1} ({elapsed:.0f}s)')
                stop_event.set()
                return
    except Exception as e:
        with lock:
            results['errors'].append(f'Client-A: {type(e).__name__}: {str(e)[:80]}')
        log(f'Client-A: ERROR {type(e).__name__}: {str(e)[:80]}')


def client_b_mixed_training(steps_per_round):
    """Client B: training with mix of good and bad requests."""
    try:
        model = create_session('client-b')
        log('Client-B: session ready')
        good_batch = make_batch(4)
        bad_batch_no_pos = make_batch(4, include_position_ids=False)  # missing position_ids
        bad_batch_odd = make_batch(3)  # odd size for DPO

        for i in range(steps_per_round):
            if stop_event.is_set():
                return
            start = time.time()
            try:
                # Every 4th request: send bad data to trigger error
                if i % 4 == 3:
                    model.forward_backward(inputs=bad_batch_no_pos)
                elif i % 7 == 6:
                    model.forward_backward(inputs=bad_batch_odd)
                else:
                    model.forward_only(inputs=good_batch, disable_lora=True)
                    model.forward_backward(inputs=good_batch)
                    model.clip_grad_and_step()
                elapsed = time.time() - start
                with lock:
                    results['steps_ok'] += 1
                if i % 3 == 0:
                    log(f'Client-B: step {i + 1}/{steps_per_round} ({elapsed:.1f}s)')
            except Exception:
                elapsed = time.time() - start
                if elapsed > TIMEOUT:
                    with lock:
                        results['hangs'].append(f'Client-B step {i + 1} ({elapsed:.0f}s)')
                    stop_event.set()
                    return
                # Expected errors from bad data - continue
                if i % 5 == 0:
                    log(f'Client-B: step {i + 1} error (expected, {elapsed:.1f}s)')
    except Exception as e:
        with lock:
            results['errors'].append(f'Client-B: {type(e).__name__}: {str(e)[:80]}')
        log(f'Client-B: ERROR {type(e).__name__}: {str(e)[:80]}')


def client_c_adapter_churn(count):
    """Client C: repeatedly create adapters (triggers ncclCommSplit)."""
    time.sleep(0.5)  # let training start first
    for i in range(count):
        if stop_event.is_set():
            return
        name = f'churn-{i}'
        start = time.time()
        try:
            init_twinkle_client(base_url=SERVER_URL, api_key='EMPTY_TOKEN')
            m = MultiLoraTransformersModel(model_id='ms://Qwen/Qwen3.5-4B')
            m.add_adapter_to_model(
                adapter_name=name,
                config=LoraConfig(r=16, target_modules=['q_proj', 'v_proj']),
                gradient_accumulation_steps=1,
            )
            elapsed = time.time() - start
            if i % 2 == 0:
                log(f'Client-C: adapter {name} OK ({elapsed:.1f}s)')
            if elapsed > TIMEOUT:
                with lock:
                    results['hangs'].append(f'Client-C {name} ({elapsed:.0f}s)')
                stop_event.set()
                return
        except Exception as e:
            elapsed = time.time() - start
            if elapsed > TIMEOUT:
                with lock:
                    results['hangs'].append(f'Client-C {name} ({elapsed:.0f}s)')
                stop_event.set()
                return
            log(f'Client-C: {name} error ({elapsed:.1f}s) - continuing')
        time.sleep(0.1)


def run_round(round_num, steps_per_round=10, adapter_churn=5):
    """Run one round of the stress test."""
    global results
    stop_event.clear()

    log(f'--- Round {round_num} (steps={steps_per_round}, churn={adapter_churn}) ---')

    t1 = threading.Thread(target=client_a_training, args=(steps_per_round,))
    t2 = threading.Thread(target=client_b_mixed_training, args=(steps_per_round,))
    t3 = threading.Thread(target=client_c_adapter_churn, args=(adapter_churn,))

    threads = [t1, t2, t3]
    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=300)

    alive = [t for t in threads if t.is_alive()]
    if alive:
        with lock:
            results['hangs'].append(f'Round {round_num}: {len(alive)} threads stuck')
        return False

    if results['hangs']:
        return False

    with lock:
        results['rounds_ok'] += 1
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--churn', type=int, default=5)
    args = parser.parse_args()

    log('=' * 60)
    log('Multi-client Race Condition STRESS Test')
    log(f'TWINKLE_FAIL_FAST = {os.getenv("TWINKLE_FAIL_FAST", "not set")}')
    log(f'Rounds={args.rounds}, Steps/round={args.steps}, Adapter churn={args.churn}')
    log('=' * 60)

    t_start = time.time()
    for r in range(1, args.rounds + 1):
        ok = run_round(r, steps_per_round=args.steps, adapter_churn=args.churn)
        if not ok:
            break

    total = time.time() - t_start
    log('')
    log('=' * 60)
    log(f'FINAL RESULTS ({total:.1f}s total)')
    log('=' * 60)
    log(f'  Rounds OK:  {results["rounds_ok"]}/{args.rounds}')
    log(f'  Steps OK:   {results["steps_ok"]}')
    log(f'  Errors:     {len(results["errors"])}')
    for e in results['errors'][:5]:
        log(f'    - {e}')
    log(f'  Hangs:      {results["hangs"]}')

    if results['hangs']:
        log('')
        log('*** NCCL HANG DETECTED ***')
        return 1
    log('')
    log('ALL ROUNDS PASSED - no hang detected.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
