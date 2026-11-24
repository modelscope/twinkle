# Multitenant operations

## Configure observability

Follow [Start the training operations dashboard](https://github.com/modelscope/twinkle/tree/main/cookbook/observability#start-the-training-operations-dashboard)
for both local and self-hosted use. The Compose example runs Collector, Tempo,
Loki, Mimir, and Grafana with persistent storage.
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
three refresh intervals, rounded up to seconds. Empty snapshots contain an
empty row list and a count of zero. Requests without a token appear as
**Unattributed** and can be selected with `unassigned`.

## Find a training and investigate an error

1. Select the time range and open **Current trainings** or **Recent training
   activity**. **Training snapshot** shows the last refresh and total live
   trainings before filters. A fresh snapshot with zero trainings is empty.
   A fresh snapshot with a positive count and an empty filtered table has no
   matches. **Unavailable** indicates a missing or expired snapshot. A data
   source error indicates a failed Redis read. Historical logs and traces
   remain independently searchable.
2. Click the **Training ID** to open **Training task logs** with that training
   and time range. Use **Log outcome** to narrow to failures, refusals, or
   timeouts. To search a longer log history, open the logs panel in **Explore**.
3. Expand a log and choose **View trace**, or open **Runtime task traces**.
   The execution span shows queue wait, execution time, outcome and error type.
   For a known task ID, search Tempo with
   `{ span.twinkle.task_id = "req_..." }`, or filter Loki with
   `{service_name="runtime-service"} | twinkle_task_id = "req_..."`.
   **Logs for this span** searches by the task ID. From the span's **Links**,
   open **Node GPU utilization** or **Node CPU utilization** to inspect the
   device's node around that execution.

## Check tenant usage and resources

- **User fingerprint prefix** accepts a fingerprint or its prefix. The
  full value appears as `twinkle.tenant_fingerprint` in a task trace.
  **Base model** and the fingerprint filter apply to training and usage.
  **Training ID** narrows the training and diagnostic panels. **Resource
  node** narrows the node metrics. Queue depth is global across tenants
  and models.
- Usage includes admitted training compute and ordinary sampling requests.
  Retries that replay an accepted request contribute once. Input token counts
  use token IDs known at admission. Execution seconds sum task duration
  across replicas. The time-range totals use Prometheus `increase()`
  estimates. Each time series needs two samples in the selected range.
  Heartbeats, polling and origin HTTP 429 responses appear separately.
- The two deployment references are requests per second and input tokens
  per second. Grafana compares five-minute rates against the same references
  for every tenant and highlights the corresponding panels.
- CPU and GPU metrics are emitted once per node and backend destination.
  RSS is per process. Interpret GPU charts as physical device usage by node
  and UUID. Attribute a device to a model only when its allocation is exclusive.
- **Drilldown → Traces** shows span rates, errors and latency using the
  Tempo local-blocks processor. **Drilldown → Logs** shows service and label
  activity. Use **Last Runtime metric received** to check exporter freshness
  and confirm trace and log ingestion separately.

## Verify the deployment

Confirm that Runtime metrics, logs, and traces arrive. Check the configured
retention and query stored samples after restarting the backends. Verify Grafana
authentication, Redis permissions, and device identity labels. Use the Twinkle
and Tinker clients to exercise session creation, heartbeat, sampling, and result
retrieval through your deployment's API endpoint.
