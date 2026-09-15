# Server test notes

## Mock backend evidence boundary (spec T8.4 / R9#10)

`server/model/backends/mock_model.py` is a stand-in backend for tests that must run
without a GPU or a real Ray-distributed model. Two properties bound what a test using
it can prove:

1. **Every method takes `**kwargs` and performs no argument validation.** A test that
   drives the mock backend therefore **cannot** be used as evidence for request/argument
   validation behavior — the mock accepts anything.
2. **It never enters a real collective.** The mock does no NCCL communication, so a test
   using it **cannot** be used as evidence for NCCL behavior (asymmetric failure,
   collective mis-pairing, ReduceScatter, etc.).

What the mock backend *can* evidence is exactly the parts that do not depend on the
backend's internals: the **dispatch path** (that a call reaches the backend via
`call_backend` / the task queue) and the **timeout / admission mechanisms** themselves
(that a slow or leaked call is bounded and the event loop stays responsive).

Tests that need to prove validation or NCCL behavior are the GPU-gated end-to-end tests
under `tests/server/integration/test_nccl_safe_*_e2e.py` (run only with
`TWINKLE_TEST_GPU_E2E=1` against a real server configured with the test execution timeout).

The contract suite covers all five apps and recursively snapshots request and response
types. Tinker compatibility assertions follow the 0.29.0 SDK wire values. The blocking
boundary suite uses a real serial Ray actor for the health-probe timing case.
