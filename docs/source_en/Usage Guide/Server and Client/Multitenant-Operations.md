# Multitenant operations

## Configure observability

The [Compose example](https://github.com/modelscope/twinkle/tree/main/cookbook/observability)
runs Collector, Tempo, Loki, Mimir, and Grafana with persistent storage.
Set `telemetry.otlp_endpoint` to a Collector reachable from Runtime. Choose a
retention period and storage backend for your deployment. The example retains
30 days of data. Keep Mimir's TSDB and working directories on supported block
storage. Restrict access to the Collector and Redis, and require authentication
for Grafana.

Set the same `TWINKLE_TELEMETRY_USER_ID_KEY` in every Runtime process. Use a
randomly generated secret separate from `TWINKLE_TOKEN_SALT`. When the key is
absent, tenant attribution and Redis snapshots are disabled. Rotating the key
changes the fingerprint. Existing telemetry and previously attributed sessions
keep their original fingerprints until they expire or the sessions are recreated.
Checkpoint directories keep their existing format.

## Redis access

Provision a dedicated Redis account against the same database as Runtime.
Apply the exact `persistence.key_prefix` when provisioning the dashboard and
the ACL. This example assumes an empty prefix and Redis database 0.

```text
ACL SETUSER twinkle-grafana reset on >DEPLOYMENT_SECRET ~admin:index:sessions ~admin:index:trainings ~admin:index:sampling +get
```

Supply the password through your secret manager. Verify the
Grafana data source can read all three keys with GET. Also verify that the same
account cannot read any business key, write, or enumerate keys with SCAN or KEYS.
The Runtime cleanup leader refreshes the snapshots at `metrics_update_interval`
(15 seconds by default). Each snapshot has a generation time and a TTL of
three refresh intervals, rounded up to seconds. An empty snapshot reports
`EMPTY`. Missing fingerprints appear as `未归属` in the dashboard.

## Investigate usage and latency

- **Current trainings** shows the live state, base model, and user fingerprint.
  Click a base model or training ID in the table to filter task logs and traces by that training. **Training snapshot updated** shows the live snapshot's
  update time or `Unavailable` if it expired. When an update time is present,
  `No trainings` means there are no current trainings and `No data` means
  the fingerprint filter matched no rows. Find finished trainings
  in **Recent training activity** by base model, last enqueue time, and the
  short training ID. Older logs may only show the time and training ID.
- **Training task logs** shows lifecycle events such as enqueue, completion,
  failure, and timeout. Select **Task errors in range** for failures, timeouts,
  and refusals, or select an outcome in **Task events by outcome** to open
  matching logs. Expand a log with a trace and choose **View trace**. Counts
  represent log events in the selected range. A task can emit multiple events. For older Runtime logs,
  use **Logs for this span** on the matching task trace.
- **User fingerprint prefix** accepts the 12-character prefix shown in charts
  or the complete `twinkle.user_id` from a trace. Leave it blank to see all
  users. The filter applies to usage, traces, and Redis details. **Base model**
  filters usage and latency metrics. **Training ID** filters task logs and traces.
  Requests without a token use a stable client ID when available and share an
  anonymous fingerprint otherwise.
- Count one request per successful admission for `forward`, `forward_only`,
  `forward_backward`, `step`, `clip_grad_and_step`, `optim_step`, and ordinary `sample` / `asample`.
  Input token counts use the submission size already available at admission.
  Unencoded inputs without token IDs contribute zero to that counter.
  Run seconds sum task execution time across replicas. The range totals subtract
  the last counter sample before the window from the final sample inside it,
  using zero for a series that starts inside the window. View heartbeats,
  result polling, and Runtime HTTP 429 separately.
- Reference limits come from deployment configuration. Grafana sums five-minute
  rates across replicas and models, applies a per-fingerprint override when
  present, and highlights relative usage. Existing per-replica rate limits
  continue to govern admission.
- Use **Runtime task traces** to find traces for the selected training.
  In **Explore → Tempo**, search by trace ID or use the TraceQL editor with
  `{ span.twinkle.request_id = "req_..." }`. For a reused SDK HTTP request ID,
  search `span.twinkle.http_request_id` and narrow by time and operation.
  Open `task_queue.execute` to see the user's `twinkle.user_id`, queue wait
  (`twinkle.wait_seconds`), execution time (`twinkle.execution_seconds`), and final task outcome.
  Telemetry records the error type when an operation fails.
  HTTP latency measures the Runtime response, including a separate result
  poll when the operation is asynchronous.
- On `task_queue.execute`, use **Logs for this span** to find Runtime logs
  matching the task ID. **Find training logs** opens the selected training's
  logs. To browse a whole service, select the time range in **Explore → Loki**.
  If you know the service and task ID, use
  `{service_name="runtime-service"} |= "req_..."`. The task ID also finds
  completion logs emitted outside the trace context.
- **Drilldown → Traces** shows span rate, errors, and latency grouped by
  attributes such as service name. Select **All spans** to include child spans
  such as `task_queue.execute`. Tempo builds these aggregates from traces
  received after its local-blocks processor is enabled. Earlier traces remain
  searchable in **Explore**. Use **Drilldown → Logs** to browse services and
  labels and find changes in log volume or errors.
- Queue depth comes from `sum by (deployment) (twinkle_queue_depth)`.
  GPU charts show physical cards by node and UUID. In the execution span, open
  **Links** and choose **Node GPU utilization** or **Node CPU utilization** for
  the node and time around that span. New traces also show `twinkle.node` as a
  resource attribute. Workers on a node share a node-local temporary directory
  to elect a CPU and GPU metrics publisher. Process RSS stays per worker.
  Establish exclusive model-to-device ownership before attributing device usage
  to a model.
- **Last Runtime metric received** shows the age of the latest heartbeat per reporting
  node. If it goes missing, check Runtime and Collector connectivity. The heartbeat
  covers the metrics pipeline. Check Loki and Tempo independently when testing ingestion.

## Verify the deployment

Confirm that Runtime metrics, logs, and traces arrive. Check the configured
retention and query stored samples after restarting the backends. Verify Grafana
authentication, Redis permissions, and device identity labels. Use the Twinkle
and Tinker clients to exercise session creation, heartbeat, sampling, and result
retrieval through your deployment's API endpoint.
