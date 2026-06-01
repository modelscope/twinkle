# Mock backend — CPU-only quick start

This directory ships an all-mock Twinkle Server configuration so you can
launch the HTTP surface in seconds on a CPU-only laptop, no GPU and no
torch/transformers/vllm/megatron required. Use it for local development,
CI smoke tests, and contract-level HTTP debugging.

> **Not for production.** Mock backends return fixed numpy-derived results
> without performing real model computation or sampling. The training and
> sampling endpoints respond with deterministic synthetic outputs derived
> only from the request shape and a seed.

## Launch

```bash
python -m twinkle.server --config cookbook/client/server/mock/server_config.yaml
```

The launcher should reach the ready state within **30 seconds** on a CPU-only
host (R4.1) — `ModelManagement` and `SamplerManagement` skip the
`twinkle.initialize(mode='ray', ...)` step that the GPU backends would run
(R3.7, R3.8).

## What the mock backends do

- **Model (`backend: mock`)** — numpy-only. Forward / forward-only /
  forward-backward calls return deterministic logprobs and elementwise
  losses keyed by `(model_id, adapter_name, seed, input_shape)`. Step /
  backward / optimizer-update calls are no-ops. Adapter add / remove /
  has are tracked in an in-memory record.
- **Sampler (`sampler_type: mock`)** — numpy-only. `sample` returns one
  `SampleResponse` per input prompt with `num_samples` sequences of length
  `max_tokens`, exactly one logprob entry per emitted token. Repeated calls
  with the same parameters return identical token sequences and logprobs.
  `max_tokens < 1` raises a validation error.

## Verifying determinism

```bash
curl -s -X POST http://localhost:8000/api/v1/model/mock/twinkle/forward_only \
  -H 'Content-Type: application/json' -d @some_payload.json
# Repeat the same request — the response body is byte-for-byte identical.
```
