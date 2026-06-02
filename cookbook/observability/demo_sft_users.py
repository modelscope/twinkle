#!/usr/bin/env python
"""End-to-end demo: 5 users running parallel SFT, full trace + log + metric.

Generates traffic that exercises every layer the spec instruments:
- Gateway / Model spans (HTTP edge)
- ServerState business spans (create_session, register_model, register_replica,
  store_future_status, unload_model)
- Task-queue execution spans
- Per-user logs at INFO/WARN/ERROR with trace_id auto-attached
- HTTP request counters + per-deployment task duration histograms
- Resource gauges (CPU / memory / process RSS)

Run:
    PYTHONPATH=src python cookbook/observability/demo_sft_users.py

Then in Grafana (http://localhost:3000):
- Tempo Search → Service=twinkle-server, Tags: twinkle.session_id=<sid>
  → all spans for that user's whole session
- Loki Explore → {service_name="twinkle-server"} | trace_id = `<id>`
  → every log for that trace
- Prometheus Explore → twinkle_http_requests_total / twinkle_task_execution_seconds
  → request rate + task latencies
"""
from __future__ import annotations

import logging
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from opentelemetry import _logs as _otel_logs, metrics, trace
from opentelemetry.util._once import Once


def _reset_otel_globals() -> None:
    """Clear OTel one-shot guards so init_telemetry runs from a clean slate."""
    trace._TRACER_PROVIDER_SET_ONCE = Once()
    trace._TRACER_PROVIDER = None
    metrics._METER_PROVIDER_SET_ONCE = Once()
    metrics._METER_PROVIDER = None
    if hasattr(_otel_logs, '_LOGGER_PROVIDER_SET_ONCE'):
        _otel_logs._LOGGER_PROVIDER_SET_ONCE = Once()
        _otel_logs._LOGGER_PROVIDER = None


def setup_telemetry(otlp_endpoint: str = 'http://localhost:4317') -> None:
    """Initialize the real production telemetry pipeline."""
    _reset_otel_globals()

    from twinkle.server.telemetry import provider
    from twinkle.server.telemetry.provider import TelemetryConfig, init_telemetry

    provider._initialized = False
    init_telemetry(TelemetryConfig(
        enabled=True,
        debug=False,
        service_name='twinkle-server',
        otlp_endpoint=otlp_endpoint,
        export_interval_ms=1000,
    ))

    # Resource collector (CPU / Mem / GPU)
    from twinkle.server.telemetry.resource_metrics import (
        get_collector, reset_collector_for_tests,
    )
    reset_collector_for_tests()
    get_collector().maybe_start()

    # Trigger MetricsRegistry once so its observable counters / histograms
    # land in the meter provider before the workers start emitting.
    from twinkle.server.telemetry.metrics import MetricsRegistry
    MetricsRegistry.get()


@contextmanager
def _gateway_span(tracer, name: str, attrs: dict):
    """Emulate the Gateway's HTTP-edge span: kind=server, route attrs."""
    with tracer.start_as_current_span(
        name, attributes={'http.method': 'POST', 'http.route': name, **attrs}
    ) as span:
        yield span


def run_sft_for_user(user_idx: int, num_steps: int = 8) -> dict:
    """Run one full SFT session for a single user — exercises every layer."""
    from twinkle.server.telemetry.correlation import (
        BASE_MODEL, MODEL_ID, REPLICA_ID, SESSION_ID, TOKEN_ID,
    )
    from twinkle.server.telemetry.metrics import MetricsRegistry
    from twinkle.server.telemetry.tracing import traced_operation

    log = logging.getLogger(f'twinkle.demo.user{user_idx}')
    log.setLevel(logging.INFO)
    metrics_reg = MetricsRegistry.get()
    tracer = trace.get_tracer('twinkle.gateway')

    sid = f'session_{uuid.uuid4().hex[:8]}'
    token = f'tok_user_{user_idx}'
    base_model = ['Qwen/Qwen3.5-4B', 'Qwen/Qwen3.5-7B', 'Qwen/Qwen3.5-1.8B'][user_idx % 3]
    replica_id = f'replica_{user_idx % 3}'

    # ---- 1. /create_session  ---------------------------------------------
    with _gateway_span(tracer, 'POST /tinker/create_session',
                       {SESSION_ID: sid, TOKEN_ID: token}):
        log.info(f'creating session for user{user_idx}',
                 extra={'twinkle.session_id': sid, 'twinkle.token_id': token})
        with traced_operation('server_state.create_session', attrs={SESSION_ID: sid}):
            time.sleep(random.uniform(0.005, 0.02))
        metrics_reg.requests_total.add(1, {'route': '/tinker/create_session', 'status': '200'})

    # ---- 2. /create_model (registers a base + LoRA, picks a replica) -----
    mid = f'mid_{uuid.uuid4().hex[:8]}'
    with _gateway_span(tracer, 'POST /tinker/create_model',
                       {SESSION_ID: sid, MODEL_ID: mid, TOKEN_ID: token, BASE_MODEL: base_model}):
        log.info(f'register_model base={base_model} replica={replica_id}',
                 extra={'twinkle.session_id': sid, 'twinkle.model_id': mid,
                        'twinkle.token_id': token, 'twinkle.base_model': base_model})
        with traced_operation('server_state.register_replica', attrs={REPLICA_ID: replica_id}):
            time.sleep(random.uniform(0.005, 0.02))
        with traced_operation('server_state.register_model',
                              attrs={SESSION_ID: sid, MODEL_ID: mid, REPLICA_ID: replica_id,
                                     TOKEN_ID: token, BASE_MODEL: base_model}):
            time.sleep(random.uniform(0.01, 0.04))
        metrics_reg.requests_total.add(1, {'route': '/tinker/create_model', 'status': '200'})

    # ---- 3. forward_backward × num_steps (the actual SFT loop) ----------
    losses = []
    for step in range(num_steps):
        with _gateway_span(tracer, 'POST /tinker/forward_backward',
                           {SESSION_ID: sid, MODEL_ID: mid, 'sft.step': step}):
            wait = random.uniform(0.001, 0.015)
            execute = random.uniform(0.05, 0.20)
            metrics_reg.task_wait_seconds.record(wait, {'deployment': 'Model'})
            with traced_operation('task_queue.execute',
                                  attrs={SESSION_ID: sid, MODEL_ID: mid, TOKEN_ID: token}):
                with traced_operation('model.forward_backward',
                                      attrs={SESSION_ID: sid, MODEL_ID: mid}):
                    time.sleep(execute)
                    loss = max(0.05, 2.5 * (0.92 ** step) + random.uniform(-0.05, 0.05))
                    losses.append(loss)
                    if step % 4 == 0:
                        log.info(f'sft step={step} loss={loss:.3f}',
                                 extra={'twinkle.session_id': sid, 'twinkle.model_id': mid,
                                        'sft.step': step, 'sft.loss': loss})
            metrics_reg.task_execution_seconds.record(execute, {'deployment': 'Model'})
            metrics_reg.tasks_total.add(1, {'deployment': 'Model', 'status': 'completed'})
            metrics_reg.requests_total.add(1, {'route': '/tinker/forward_backward', 'status': '200'})

    # Simulate a user that hits the rate limit at step 3 of 8
    if user_idx == 2:
        with _gateway_span(tracer, 'POST /tinker/forward_backward',
                           {SESSION_ID: sid, MODEL_ID: mid, 'sft.step': num_steps}):
            log.warning(f'rate-limit rejection for user{user_idx}',
                        extra={'twinkle.session_id': sid, 'twinkle.token_id': token})
            metrics_reg.rate_limit_rejections.add(1, {'deployment': 'Model'})
            metrics_reg.requests_total.add(1, {'route': '/tinker/forward_backward', 'status': '429'})

    # Simulate a hard failure for user 4
    if user_idx == 4:
        with _gateway_span(tracer, 'POST /tinker/optim_step',
                           {SESSION_ID: sid, MODEL_ID: mid}):
            try:
                with traced_operation('model.optim_step', attrs={SESSION_ID: sid, MODEL_ID: mid}):
                    raise RuntimeError('optimizer NaN at user4 step5')
            except RuntimeError:
                log.exception(f'sft failed sid={sid} mid={mid}',
                              extra={'twinkle.session_id': sid, 'twinkle.model_id': mid})
                metrics_reg.tasks_total.add(1, {'deployment': 'Model', 'status': 'failed'})
                metrics_reg.requests_total.add(1, {'route': '/tinker/optim_step', 'status': '500'})

    # ---- 4. /save_weights (client downloads LoRA) ------------------------
    with _gateway_span(tracer, 'POST /tinker/save_weights',
                       {SESSION_ID: sid, MODEL_ID: mid}):
        log.info(f'save_weights mid={mid}',
                 extra={'twinkle.session_id': sid, 'twinkle.model_id': mid})
        with traced_operation('server_state.store_future_status', attrs={MODEL_ID: mid}):
            time.sleep(random.uniform(0.02, 0.08))
        metrics_reg.requests_total.add(1, {'route': '/tinker/save_weights', 'status': '200'})

    # ---- 5. /unload_model (cleanup) --------------------------------------
    with _gateway_span(tracer, 'POST /tinker/unload_model',
                       {SESSION_ID: sid, MODEL_ID: mid}):
        log.info(f'unload_model mid={mid}',
                 extra={'twinkle.session_id': sid, 'twinkle.model_id': mid})
        with traced_operation('server_state.unload_model', attrs={MODEL_ID: mid}):
            time.sleep(random.uniform(0.005, 0.015))
        metrics_reg.requests_total.add(1, {'route': '/tinker/unload_model', 'status': '200'})

    return {'user_idx': user_idx, 'session_id': sid, 'model_id': mid,
            'token': token, 'base_model': base_model,
            'final_loss': losses[-1] if losses else None,
            'num_steps': num_steps}


def main() -> None:
    setup_telemetry()
    log = logging.getLogger('twinkle.demo')
    log.setLevel(logging.INFO)

    NUM_USERS = 5
    log.info(f'launching {NUM_USERS} concurrent SFT runs')

    with ThreadPoolExecutor(max_workers=NUM_USERS) as pool:
        futures = [pool.submit(run_sft_for_user, i, num_steps=8) for i in range(NUM_USERS)]
        results = [f.result() for f in futures]

    log.info(f'all {NUM_USERS} users finished SFT')
    print('\n=== Per-user summary (use these IDs to query in Grafana) ===')
    for r in results:
        print(f"  user{r['user_idx']}  token={r['token']:14s}  session={r['session_id']}  "
              f"model={r['model_id']}  base={r['base_model']:20s}  "
              f"final_loss={r['final_loss']:.3f}" if r['final_loss'] else "")

    # Drive resource gauges + flush everything
    time.sleep(3)
    trace.get_tracer_provider().force_flush(timeout_millis=10000)
    metrics.get_meter_provider().force_flush(timeout_millis=10000)
    from twinkle.server.telemetry import provider
    provider._logger_provider.force_flush(timeout_millis=10000)
    time.sleep(2)
    print('\nflushed traces + logs + metrics to OTLP')


if __name__ == '__main__':
    main()
