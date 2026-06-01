# Observability

Twinkle Server emits OpenTelemetry traces, metrics, and logs from every Ray
Serve deployment. This guide covers the standardized **correlation keys**,
the Ray Serve **trace-context propagation** mechanism, and an end-to-end
**LGTM** example using the Loki / Grafana / Tempo / Mimir docker-compose
stack shipped under `cookbook/observability/`.

## Correlation keys

Every business-layer span carries a subset of these attributes when the
corresponding identifier is known to the operation. All names share the
`twinkle.` prefix so you can filter Tempo / Loki by a single namespace.

| Attribute                    | Set when the operation is associated with… |
|------------------------------|--------------------------------------------|
| `twinkle.session_id`         | A client session                           |
| `twinkle.model_id`           | A specific registered model                |
| `twinkle.replica_id`         | A specific Ray Serve replica               |
| `twinkle.token_id`           | A user authentication token                |
| `twinkle.sampling_session_id`| A sampling session                         |
| `twinkle.base_model`         | The base model behind a registered model   |

Constants live in `twinkle.server.telemetry.correlation`. Use
`set_correlation_attrs(span, {...})` to attach them — None values are
skipped, so partially-known operations never get empty attributes.

```python
from twinkle.server.telemetry.correlation import (
    SESSION_ID, MODEL_ID, set_correlation_attrs,
)
from twinkle.server.telemetry.tracing import traced_operation

with traced_operation('server_state.register_model',
                      attrs={SESSION_ID: sid, MODEL_ID: mid}):
    ...
```

When the OpenTelemetry SDK is not installed, `traced_operation` becomes a
NoOp context manager: the body runs to completion and returns the same
result it would return when tracing is active.

## Trace-context propagation across deployments

The HTTP edge already injects context into outgoing headers in
`gateway/proxy.py`, and `create_tracing_middleware` extracts it on the
inbound side, so a Tinker request that passes through the Gateway proxy
shares one trace id end to end.

The remaining gap is **Ray Serve `DeploymentHandle` calls** between
deployments — those don't go over HTTP. Use the trace-context carrier
helpers:

```python
from twinkle.server.telemetry.context_carrier import make_carrier, activate_carrier

# caller side (e.g. Model deployment) — pass the carrier with the call
carrier = make_carrier()
result = await sampler_handle.options(...).remote(payload, trace_context=carrier)

# callee side (e.g. Sampler deployment handler)
async def handler(payload, trace_context: dict | None = None):
    with activate_carrier(trace_context):
        with traced_operation('sampler.handle'):
            ...
```

`make_carrier()` returns an empty dict and `activate_carrier(None)` is a
no-op when OTel is missing or the carrier is empty, so the path stays
safe under graceful degradation.

## End-to-end LGTM example

The repository ships a docker-compose stack with Grafana, Tempo (traces),
Loki (logs), and Mimir (metrics) under `cookbook/observability/`.

```bash
# 1. Start the LGTM stack.
docker compose -f cookbook/observability/docker-compose.yml up -d

# 2. Launch the server with telemetry enabled.
cat > /tmp/srv.yaml <<'YAML'
telemetry:
  enabled: true
  service_name: twinkle-server
  otlp_endpoint: http://localhost:4317
persistence: { mode: memory }
applications: []
YAML

python -m twinkle.server launch --config /tmp/srv.yaml &

# 3. Issue some traffic and open Grafana at http://localhost:3000.
#    In Tempo, search by tag: `twinkle.session_id = <your session>`.
```

CPU / memory / GPU metrics show up automatically because the
`ResourceMetricsCollector` is started inside every Ray Serve worker by
`ensure_telemetry_initialized()`. When `psutil` or `pynvml` is missing
(or no GPU is present), the affected gauges report no data and the
worker keeps serving requests.
