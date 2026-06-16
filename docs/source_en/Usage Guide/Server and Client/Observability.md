# Observability

Twinkle Server provides full observability through OpenTelemetry, covering traces, metrics, and logs.

## Quick Start

### 1. Start the Observability Stack

The project includes a one-command Docker Compose setup based on the `grafana/otel-lgtm` image (bundles OTel Collector, Mimir, Tempo, Loki, and Grafana):

```bash
cd cookbook/observability
docker compose up -d
```

Available services after startup:

| Service | URL | Purpose |
|---------|-----|---------|
| Grafana | `http://localhost:3000` | Dashboards and data exploration |
| OTLP gRPC | `localhost:4317` | Point Twinkle's `otlp_endpoint` here |
| OTLP HTTP | `localhost:4318` | Same, HTTP alternative |

### 2. Configure the Server

Enable telemetry in `server_config.yaml`:

```yaml
telemetry:
  enabled: true
  otlp_endpoint: http://localhost:4317
```

### 3. Install Dependencies

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
```

### 4. Launch the Server

```bash
twinkle-server launch -c server_config.yaml
```

### 5. Open Grafana

Navigate to `http://localhost:3000`. Default credentials: `admin` / `admin`.

## telemetry Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Whether to enable the telemetry pipeline |
| `service_name` | str | `twinkle-server` | Reported service name |
| `otlp_endpoint` | str | `http://localhost:4317` | OTel Collector gRPC address |
| `debug` | bool | `false` | When `true`, dumps spans/metrics to console instead of OTLP |
| `export_interval_ms` | int | `30000` | Metrics export interval (milliseconds) |
| `resource_attributes` | dict | `{}` | Additional resource attributes attached to all telemetry |

## Built-in Grafana Dashboard

The provisioned **Twinkle Server Overview** dashboard includes:

- HTTP request rate and P95 latency per deployment (Gateway / Model / Sampler / Processor)
- Active resource counts (sessions, models, sampling sessions, futures)
- Task queue depth, execution P95, wait-time P95
- Rate-limit rejections and task completions by status

## Metric Naming Reference

Twinkle uses dot-notation OpenTelemetry metric names. Prometheus OTLP ingestion converts dots to underscores and appends `_total` to monotonic counters:

| OpenTelemetry Name | Prometheus Name |
|--------------------|-----------------|
| `twinkle.http.requests.total` | `twinkle_http_requests_total` |
| `twinkle.http.request.duration_seconds` | `twinkle_http_request_duration_seconds_bucket` |
| `twinkle.queue.depth` | `twinkle_queue_depth` |
| `twinkle.task.execution_seconds` | `twinkle_task_execution_seconds_bucket` |
| `twinkle.task.wait_seconds` | `twinkle_task_wait_seconds_bucket` |
| `twinkle.rate_limit.rejections.total` | `twinkle_rate_limit_rejections_total` |
| `twinkle.tasks.total` | `twinkle_tasks_total` |
| `twinkle.sessions.active` | `twinkle_sessions_active` |
| `twinkle.models.active` | `twinkle_models_active` |
| `twinkle.sampling_sessions.active` | `twinkle_sampling_sessions_active` |
| `twinkle.futures.active` | `twinkle_futures_active` |

> The `*.active` resource gauges report absolute values. Do NOT wrap them with `rate()` or `increase()`.

## Tracing

Twinkle spans are namespaced under `twinkle.server.<component>` (Gateway / Model / Sampler / Processor). Each request carries `twinkle.session_id` and `trace_id` correlation keys, supporting end-to-end cross-deployment tracing.

In Grafana, switch the datasource to Tempo to search traces by service name or span name.

## Production Deployment

The LGTM all-in-one image in `cookbook/observability` is **for local development and demos only**. For production:

- Deploy Mimir / Tempo / Loki / Grafana separately with persistent storage and replicas
- Place an independent OTel Collector tier in front for sampling and routing
- The `telemetry` config and metric names in `server_config.yaml` transfer without changes

## Troubleshooting

**Grafana shows "No data"**
- Confirm `telemetry.enabled: true` in your config
- Confirm worker logs show `Worker telemetry initialized`
- Set `debug: true` to verify spans appear in the console, then switch back to `debug: false`

**Twinkle can't reach the Collector**
- `otlp_endpoint` must be reachable from the Twinkle process. If Twinkle runs in a separate container, use the Docker network address e.g. `http://twinkle-lgtm:4317`

**Resource gauges stuck at 0**
- Only the cleanup-leader worker pushes resource counts. If gauges remain at 0 for longer than `export_interval_ms × 2` after startup, check logs for "became cleanup leader" messages

## Tear Down

```bash
cd cookbook/observability
docker compose down -v   # -v removes the data volume as well
```
