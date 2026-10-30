# Qwen3-TTS cookbook — layer 2

This contribution is part 2 of a four-part series. Scope is cumulative:
1. Audio features, padded input processing, two-stage forward and codebook loss.
2. Twinkle model bridge and pretrained LoRA update/save/reload.
3. Accumulation windows and update-boundary checkpoint restoration.
4. Offline manifest preparation and the user-facing training CLI.

Only the features through part 2 are present here. No generation, quality evaluation or changes to Twinkle core are included.

Install Twinkle from the repository root (`pip install -e .`) and optional dependencies with `pip install -r cookbook/qwen3_tts/requirements.txt`. The real-model tests require an existing local Qwen3-TTS 12Hz 1.7B Base and a single Ascend NPU. Development baseline: Twinkle 5337cc3, Qwen 022e286, transformers 4.57.3, torch 2.10 / torch_npu 2.10, bf16. No model download is needed.

From this cookbook directory, run `python tests/batch1/test_contracts.py`. CPU development without Twinkle/Qwen can use `--isolated-twinkle`; this substitutes infrastructure and skips two official package checks. It is not native or NPU validation.

Run `python tests/lora/test_contracts.py` (or add `--isolated-twinkle`). Native pretrained acceptance: `MODEL_PATH=/models/Qwen3-TTS-12Hz-1.7B-Base ASCEND_RT_VISIBLE_DEVICES=0 python tests/lora/test_real_model.py`.

The final series tree is byte-identical to the integrated training baseline 0466bd2 that passed 40 tests on one Ascend NPU. Individual intermediate layers receive CPU checks; they have not independently been rerun on NPU. No convergence or voice-similarity result is claimed. The model bridge depends on Twinkle internal optimizer-group and wrapping APIs; these remain confined to this cookbook.
