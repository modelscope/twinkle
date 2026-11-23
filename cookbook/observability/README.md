# Twinkle Observability Stack

For local development, build the standalone Twinkle LGTM image on top of the
[`grafana/otel-lgtm`](https://github.com/grafana/docker-otel-lgtm) image.
It bundles OTel Collector, Prometheus, Tempo, Loki, Grafana, and the Twinkle
Server Overview dashboard.

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
docker compose up -d --build

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

To build just the LGTM image, run this command from the repository root.

```sh
docker build -t twinkle-lgtm:0.28.0 cookbook/observability
```

Build the CUDA Twinkle Server image from the repository root with
`docker build --platform linux/amd64 -t twinkle-server:local .`. The build
installs the checkout's source and packages its cookbook startup scripts.
Check out the target branch or tag before building a release image.

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

## Dedicated monitoring host

`docker-compose.production.yaml` runs Collector, Tempo, Loki, Mimir, and Grafana
as separate services on one internal CPU host. Place Docker's named volumes on
supported persistent block storage with capacity for 30 days of telemetry. Use
the platform's approved object storage configuration if the storage available
for telemetry is NFS. Keep Mimir's TSDB and component working directories on
supported local block storage.

1. Provision a private OTLP listener and a Grafana listener on the administrator
   network. Allow TCP 4317 from Runtime and TCP 3000 from the managed network only.
   Keep the business Redis and the three backends on internal networks.
2. Have the Redis administrator provision a dedicated account for GET on the
   three snapshot keys in the operations guide. Use the same Redis database and
   key prefix as Runtime.
3. Copy `production/usage-limits.example.json` to deployment configuration.
   Set default rates and optional overrides keyed by the complete fingerprint.
   Render the dashboard with the existing Runtime Redis key prefix:

   ```sh
   python3 production/render_dashboard.py /path/to/usage-limits.json /path/to/twinkle-multitenant.json \
       --redis-key-prefix '<runtime persistence.key_prefix>'
   ```

4. Set `OBS_OTLP_BIND_IP`, `OBS_GRAFANA_BIND_IP`, `OBS_DASHBOARD_FILE` (the
   absolute output path above), `OBS_REDIS_URL` (address without credentials),
   and `OBS_REDIS_USER`. Inject `OBS_REDIS_PASSWORD`, `OBS_GRAFANA_USER`, and
   `OBS_GRAFANA_PASSWORD` via deployment configuration. The Grafana image
   installs the pinned Redis data source plugin. For an offline host, stage the
   signed ARM64 or AMD64 plugin in Grafana's plugin directory before starting.

   ```sh
   docker compose -f docker-compose.production.yaml up -d
   ```

5. Merge `production/server_config.yaml` into the existing Runtime config,
   select the internal Collector address, and inject the same
   `TWINKLE_TELEMETRY_USER_ID_KEY` into every Runtime worker using a deployment
   secret. With the Megatron container entrypoint, pass
   `--server-config /path/to/deployment.yaml` to load the resulting config.
   The entrypoint supervises Runtime independently of the monitoring stack.
   Confirm all three signals arrive.

See the [multitenant operations guide](../../docs/source_en/Usage%20Guide/Server%20and%20Client/Multitenant-Operations.md)
for Redis ACLs, credential handling, dashboard queries, and release checks.

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
