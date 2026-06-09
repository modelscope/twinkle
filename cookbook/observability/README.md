# Twinkle Observability Stack

A one-container OTLP receiver + dashboard for Twinkle, built on the
[`grafana/otel-lgtm`](https://github.com/grafana/docker-otel-lgtm) image.
That image bundles OTel Collector, Mimir (Prometheus-compatible), Tempo,
Loki, and Grafana with everything pre-wired — no extra config files needed.

## What you get

| Surface | URL | Purpose |
|---|---|---|
| Grafana | `http://localhost:3000` | Dashboards + Explore (metrics / traces / logs) |
| OTLP gRPC | `localhost:4317` | Point Twinkle's `otlp_endpoint` here |
| OTLP HTTP | `localhost:4318` | Same, HTTP alternative |

## Quick start

```bash
# 1. Start the stack
cd cookbook/observability
docker compose up -d

# 2. Make sure Twinkle has the OTLP exporter
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp

# 3. In your server_config.yaml:
#
#    telemetry:
#      enabled: true
#      debug: false                  # debug=true dumps to console instead of OTLP
#      service_name: twinkle-server
#      otlp_endpoint: http://localhost:4317

# 4. Launch Twinkle as usual
twinkle-server launch -c server_config.yaml

# 5. Open Grafana
open http://localhost:3000
```

Anonymous viewer access is on by default; full access is `admin` / `admin`.

The provisioned **Twinkle / Twinkle Server Overview** dashboard shows:

- HTTP request rate and P95 latency per deployment (Gateway / Model / Sampler / Processor)
- Active resources (sessions, models, sampling sessions, futures) — reported by
  the cleanup-leader worker only, so the gauges read absolute values rather
  than the count multiplied by the worker fleet size
- Task queue depth, execution P95, wait-time P95
- Rate-limit rejections and task completions by status

For traces, switch the datasource picker in **Explore** to Tempo and search by
service or span name. Twinkle spans are namespaced under
`twinkle.server.<component>` (Gateway / Model / Sampler / Processor).

## Metric naming reference

Twinkle emits OpenTelemetry metric names with dot notation. Prometheus's OTLP
ingestion converts dots to underscores and appends `_total` to monotonic
counters where missing:

| OpenTelemetry name | OTEL instrument | Prometheus name |
|---|---|---|
| `twinkle.http.requests.total` | Counter | `twinkle_http_requests_total` |
| `twinkle.http.request.duration_seconds` | Histogram | `twinkle_http_request_duration_seconds_bucket` (and `_sum`, `_count`) |
| `twinkle.queue.depth` | UpDownCounter | `twinkle_queue_depth` |
| `twinkle.task.execution_seconds` | Histogram | `twinkle_task_execution_seconds_bucket` |
| `twinkle.task.wait_seconds` | Histogram | `twinkle_task_wait_seconds_bucket` |
| `twinkle.rate_limit.rejections.total` | Counter | `twinkle_rate_limit_rejections_total` |
| `twinkle.tasks.total` | Counter | `twinkle_tasks_total` |
| `twinkle.rate_limiter.active_tokens` | UpDownCounter | `twinkle_rate_limiter_active_tokens` |
| `twinkle.sessions.active` | ObservableGauge | `twinkle_sessions_active` |
| `twinkle.models.active` | ObservableGauge | `twinkle_models_active` |
| `twinkle.sampling_sessions.active` | ObservableGauge | `twinkle_sampling_sessions_active` |
| `twinkle.futures.active` | ObservableGauge | `twinkle_futures_active` |

The four `*.active` resource gauges are reported as ObservableGauges, so
they surface as Prometheus gauges holding the absolute count. Queries that
need a rate (`rate(...)` / `increase(...)`) should NOT wrap these — read
the gauge value directly.

## Tear down

```bash
docker compose down -v   # -v also removes the named volume
```

## Production note

The LGTM all-in-one image is **for local development and demos**. Each backend
runs single-instance and shares one volume. For production, deploy each
component (Mimir / Tempo / Loki / Grafana) separately with proper persistent
storage, replicas, and an OTel Collector tier in front. The OTLP endpoint and
metric names stay the same, so your `server_config.yaml` and dashboards
transfer without changes.

## Troubleshooting

- **Grafana shows "No data"** — confirm `telemetry.enabled: true` in
  your server config and that Twinkle's worker logs show
  `Worker telemetry initialized`. With `debug: true` Twinkle dumps spans /
  metrics to logs instead of OTLP, so set `debug: false` once verified.
- **Twinkle can't reach the collector** — `otlp_endpoint` must be reachable
  from the Twinkle process. If Twinkle runs in another container on the same
  Docker network, use `http://twinkle-lgtm:4317` instead of `localhost`.
- **Resource gauges read 0** — only the cleanup-leader worker pushes counts
  into the ObservableGauge cache. If the gauges sit at 0 for more than
  ~`metrics_update_interval` × 2 seconds after startup, none of the workers
  ever became leader; check `twinkle.utils.logger` for "became cleanup
  leader" log lines.
- **Dashboard panel shows "Datasource not found"** — open the panel, switch
  the datasource dropdown to the LGTM-provisioned Prometheus / Tempo and save.
  This happens when LGTM versions change the default datasource UID; the
  dashboard JSON pins `uid: prometheus`.
