# On-demand training and device reuse

The catalog, controller, allocation, and SDK routing are implemented. Review fixes
are under local validation. Real-device reruns are deferred at the user's request;
see the [validation record](validation/on-demand-training/README.md).

## Scope and requirements

A fixed device pool serves a catalog larger than its resident capacity. The demand
unit is a training handle; the allocation unit is a model instance on N devices.
Sampling retains its existing deployment lifecycle. This change does not introduce
resource waiting queues, preemption, training batching, or automatic checkpointing.

| Requirement | Behavior |
| --- | --- |
| R01 | Support the existing Transformers and Megatron training backends. |
| R02 | Administrators declare model sources, modes, parallelism, and N devices. |
| R03 | New mode specifications default to lazy startup; `min_replicas` keeps resident capacity. |
| R04 | Initialize residents before accepting dynamic training; all instances use Ray resources. |
| R05 | An instance reserves N devices on one node; independent bases do not share that reservation. |
| R06 | A model may offer both LoRA and full training through separate instances. |
| R07 | Fill compatible LoRA instances first; expand when slots and devices permit. |
| R08 | Each full handle exclusively owns an instance, subject to user quota. |
| R09 | Capacity failures are terminal; callers explicitly issue a new create to retry. |
| R10 | New specifications use business inactivity, defaulting to 1800 seconds. |
| R11 | Heartbeats, polling, health, and information queries do not renew business inactivity. |
| R12 | Remove an empty dynamic instance immediately; retain clean resident capacity. |
| R13 | Expired handles stay invalid; users explicitly recreate. No checkpoint is resumed implicitly. |
| R14 | Worker failure or controller restart invalidates affected handles without replaying training. |
| R15 | Retain the existing instance queue and backend execution gate. |
| R16 | Keep `applications`, old model arguments, and SDK methods; use one controller. |
| R17 | Every create gets a distinct training ID; identical adapter labels cannot reattach old state. |
| R18 | Validate both backends and modes on real devices before marking the PR ready. |

## Configuration and compatibility

The public entry point remains `applications` with `import_path: model`.
The [configuration example](../../cookbook/client/server/on_demand_training.yaml)
shows `args.modes.lora/full`. Each mode sets `num_devices`, `device_mesh`,
`min_replicas`, `max_replicas`, and `idle_timeout_s`. LoRA additionally sets
`max_loras`, `max_r`, and its existing backend options.

- New modes default to `min_replicas=0`, `idle_timeout_s=1800`, five LoRA slots,
  and maximum LoRA rank 32. Full capacity is one handle.
- Mesh world size must equal N; N and capacities are positive integers. Timeouts
  must be finite and positive. Minimum replicas cannot exceed the maximum.
- `modes` cannot be mixed with legacy single-mode settings or deployment replicas.
- Legacy `train_mode`, device counts, mesh, strategy, LoRA parameters, and environment
  variables normalize into the same runtime catalog.
- Existing deployment replica bounds keep their resident-minimum/instance-limit
  meaning. Explicit `args.min_replicas` and `args.max_replicas` take precedence.
- **Legacy `adapter_timeout` remains a session heartbeat timeout.** It is not converted
  to business inactivity. `adapter_max_lifetime` remains an absolute handle lifetime.
  Their existing defaults are 1800 and 36000 seconds respectively.
- A legacy configuration enables business inactivity only by explicitly setting
  `args.idle_timeout_s`. Its heartbeat and absolute lifetime bounds still apply.
- The Native SDK and server must upgrade together for the internal training-ID
  handshake. Native and Tinker retain their existing initialization differences.

Controller allocation/startup/shutdown/sweep bounds default to 3/1800/60/5 seconds
under `training`. Base startup and user initialization each receive a bounded
startup window; waiting for a base does not consume the user's initialization window.
The create future covers both phases without allowing late success.

## Ownership and lifecycle

`TrainingCatalog` performs pure configuration lookup. `TrainingController` is the
single lifecycle writer. `RayInstanceDriver` performs Ray/Serve effects, and
`ModelManagement` executes backend operations on the existing queue.

The controller is a named actor owned by the launcher, not a detached service that
another launcher adopts. Initialization failure cleans and terminates that actor.
While its owner is alive, a controller restart starts a new epoch and invalidates
old handles. Runtime constructors register an incarnation once; Serve cannot
silently replace a process while preserving its old training state.

The state lock covers decisions and commits. Model startup, user initialization,
cleanup, and resident rebuilding run outside it. A resident rebuilding for one
model/mode does not block another specification or the sweep loop.

Every instance owns a named Ray Placement Group with one accelerator/CPU bundle
and `STRICT_PACK`. The allocator bounds `pg.ready()`, obtains actual device IDs
through a scheduled probe, and injects that allocation into workers without a
second reservation. Worker, application, registry, and PG names identify their
server namespace, controller epoch, and instance ID.

A model has a stable CPU router. Creating/releasing handles enters the controller;
training requests route to their fixed instance. Internal model lifecycle HTTP
routes explicitly reject direct creation or unloading. Backend workers do not
restart or replay stateful calls automatically.

| Record | States |
| --- | --- |
| Training | PENDING → ACTIVE → CLOSING → RELEASED / EXPIRED / FAILED / CANCELLED |
| Instance | ALLOCATING → STARTING → READY; RESETTING for resident full reset; DRAINING → STOPPED |
| Unconfirmed shutdown | QUARANTINED, with devices/quota still owned until death is confirmed |

## Creation, isolation, and expiry

Create identity is `(client type, token, session, create key)`: Native uses its UUID
and Tinker uses `model_seq_id`. Identical retries return the original future;
changed parameters are rejected. Pending users count toward quota and LoRA slots.
Compatible pending creates share a base startup. Full handles never share an instance.

A failed user initialization closes only that user. Its slot remains reserved while
queued or late initialization may still execute. Cleanup follows initialization in
the serial lifecycle queue, and a late result cannot publish an ACTIVE handle.
Worker death, base startup failure, or unsafe backend execution can fail an entire
instance; a create RPC timeout alone is not evidence of that failure.

Controller tickets protect accepted operations from expiry. The runtime exposes
queued/executing membership through the compute worker's public interface. Sweep
reconciles both uncommitted admission and committed operations whose completion
notification was lost. A terminal future alone does not establish that a background
call stopped. Reconciled activity uses the original completion time, not sweep time.

Native and Tinker keep bounded sequence retry receipts per training stream. An
expired-sequence fence survives while the handle is live, so expiry cannot cause an
old optimizer operation to run again. Retained receipts tolerate out-of-order
requests; claims are released only when submission never created a future. Retiring
the handle removes its stream ledgers. Redis enumeration uses the existing SCAN API.

Expiry closes inactive users only after their operations drain. Business inactivity
uses executed operations; heartbeat-only clients expire under the new policy.
Legacy heartbeat and maximum-lifetime checks retain their original meanings.

The last user leaving closes a dynamic instance. Clean residents keep their base;
full residents reset weights, optimizer state, gradients, and template hooks before
another user is admitted. A resident modified by an instance-level patch is rebuilt
when empty. A cleanup failure cannot release a slot or kill another active LoRA user
merely because its cleanup RPC timed out.

Shutdown kills owned workers, deletes the owned Serve app, removes the PG, confirms
worker/resource termination, and then retires quota. Failed confirmation preserves
ownership and retries in the background. Recovery never clears a shared registry or
shuts down another component's Ray cluster. Launcher shutdown retains its normal
Serve teardown behavior.

## Acceptance

| Case | Required observation |
| --- | --- |
| A01 | Startup/query/health on a lazy catalog loads no base or training PG. |
| A02 | Concurrent first creates coalesce into the required LoRA slot count. |
| A03 | Full LoRA slots expand with devices available; exhausted capacity fails terminally. |
| A04 | Both backends and SDKs isolate LoRA tenants and exclusive full handles. |
| A05 | Disjoint allocations are released and reused by another model. |
| A06 | Fragmented resources across nodes cannot satisfy one instance. |
| A07 | Heartbeats do not renew new-policy business inactivity; legacy semantics remain unchanged. |
| A08 | Idle LoRA cleanup preserves an active tenant's optimizer continuation. |
| A09 | Queued/running/save operations drain before expiry or release. |
| A10 | Unsafe background execution retains ownership until workers terminate. |
| A11 | Partial startup, cancellation, and allocation failure leave correct quota and ownership. |
| A12 | Worker/controller failure fences old handles and preserves unrelated instances. |
| A13 | Same-label recreation, old requests, and sequence retries cannot mutate another handle. |
| A14 | Explicit recreation starts with clean weights and optimizer state; checkpoint restoration is separate. |
| A15 | Resident full reset leaves a clean base for the next user. |
| A16 | Ten cross-model reload cycles leave no accumulated workers, PGs, or device memory. |
| A17 | Invalid mode/config, expired IDs, and cross-user access fail without allocation. |
| A18 | Related SDK, YAML, and repository CPU regressions pass together. |
| A19 | Late startup/user initialization cannot publish success or prematurely reuse a slot. |
| A20 | Memory and Redis recovery invalidate old state and rebuild quota correctly. |
| A21 | Checkpoint upload needs no base; patched residents rebuild before reuse. |

Review regressions also cover nonblocking resident recovery, launcher initialization
rollback, lost completion notification, and sequence-receipt expiry. CPU/logical Ray
resources are not device training evidence. The standalone resource lifecycle is accepted with DP on both backends.
Checkpoint restoration and FSDP/TP/SP backend repairs are tracked separately in #315.
Raw device evidence is retained outside the public source tree.
