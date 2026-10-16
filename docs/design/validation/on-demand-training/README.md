# On-demand training validation

PR #312 is based directly on main. It does not require the checkpoint,
FSDP optimizer, or Megatron TP/SP repairs in #315. Its backend changes are
limited to resetting resident full-training state and using the supported
Transformers configuration import needed to initialize Megatron.

Collected CPU tests under `tests/server/training/` exercise the controller,
allocation, recovery, and both installed SDKs over HTTP using a mock backend.
The HTTP and allocation probes run in isolated subprocesses. Existing
applications YAML startup can also be checked with memory and Redis state.

Standalone validation programs live in `tools/validation/on_demand_training/`.
The DP resource-lifecycle device suite is explicitly invoked:

```bash
python -m pytest tools/validation/on_demand_training/device_validation.py -vv
```

It runs DP=2, TP=1, and no FSDP for both backends and both SDKs in shared
LoRA/exclusive full modes. Assertions cover actual training and tenant isolation,
release, cross-model reuse, business-idle reclamation, expired handles, and clean
recreation. Checkpoint I/O and FSDP/TP repairs belong to #315, not this acceptance.
Results from older revisions containing #315 do not validate standalone #312.

Device validation requires explicit `TWINKLE_DEVICE_*` configuration, a separate
evidence directory, and exclusively available devices. Record the exact source
SHA, import origins, raw JUnit/logs, numerical assertions, and resource cleanup.
Current results and remaining limitations are recorded in the English PR body;
results from an older revision are not acceptance evidence for the new head.
