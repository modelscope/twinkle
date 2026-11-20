# Generate speech with a trained adapter

Run in the cookbook directory with the existing local model and optional dependencies:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python scripts/generate.py \
  --model_path /models/Qwen3-TTS-12Hz-1.7B-Base \
  --checkpoint /data/run/checkpoint-update-70 \
  --reference_audio /data/reference.wav \
  --text "A new sentence to synthesize." \
  --output /data/generated.wav
```

Omit `--checkpoint` to use the base model. The output must be new. `--seed`
(default 42) controls sampling, and `--max_new_tokens` (default 256) bounds
length. The reference must be finite mono 16/24 kHz audio from the intended
speaker. Prefer the reference used to prepare training data.

Auto language conditioning and speaker-only cloning match this cookbook's
training prefix. Sampling uses the pinned official implementation's defaults;
TTS runs in NPU bf16 and the codec in CPU fp32. Models load locally without
a download fallback. WAV values are saved without normalization or clipping.
A valid WAV does not guarantee correct speech: listen for omissions, repetition
and truncation at the token limit.

For the optional real-model inference smoke test, provide `MODEL_PATH` and
`CHECKPOINT_PATH`, then run `python tests/entrypoint/test_inference_npu.py`.
It checks finite audio, not quality. HTML comparison pages, human-score
collection and project acceptance workflows are outside this cookbook.
