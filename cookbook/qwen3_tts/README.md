# Qwen3-TTS LoRA training on one Ascend NPU

This cookbook connects the Qwen3-TTS 12Hz Base model to Twinkle's local
Transformers training API. It provides explicit train/dev preparation,
single-sample micro-batches, LoRA on the Talker and Code Predictor, accumulation
with a final short window, and restart at completed optimizer updates.

It is a single-card reference implementation, not a quality-tuned training
recipe. There are no changes to Twinkle core or its default dependencies.
A minimal adapter inference example is documented in [GENERATION.md](GENERATION.md).

## Installation and supported scope

Run from the repository root with an existing Ascend PyTorch environment:

```bash
pip install -e .
pip install -r cookbook/qwen3_tts/requirements.txt
cd cookbook/qwen3_tts
export MODEL_PATH=/models/Qwen3-TTS-12Hz-1.7B-Base
export ASCEND_RT_VISIBLE_DEVICES=0
```

The model directory must already contain the complete Base weights, config,
text tokenizer, processor and `speech_tokenizer/`. Preparation and training are
offline: missing assets cause an error, not a download. No virtual environment
or model download is created by these commands. Select one physical card with
`ASCEND_RT_VISIBLE_DEVICES`; `npu:0` is relative to that selection.

The development baseline used Twinkle `5337cc3`, Qwen3-TTS `022e286`, Python
3.12, torch 2.10.0, torch_npu 2.10.0.post2, CANN 9.1.0 and an Ascend 910B.
The optional requirements pin the Qwen implementation and its Transformers
4.57.3 / accelerate 1.12.0 dependencies. They are not installed for ordinary
Twinkle users. Match torch/torch_npu to your Ascend image before installing
these extras. Codec and reference mel extraction use CPU fp32; the TTS model
uses NPU bf16 with eager attention. Multi-card, CUDA, padding-free execution
and other Qwen3-TTS model variants are not covered by the current evidence.

## Prepare an existing split

Provide two UTF-8 JSONL files. Each row has `id`, `text`, `audio`, `group_id`
and `speaker_id`. Paths are relative to the input JSONL file. For example:

```json
{"id":"train-001","text":"A training utterance.","audio":"audio/train-001.wav","group_id":"session-a","speaker_id":"speaker-a"}
```

Use a different held-out group and text in the dev file. Audio must be finite
mono 16 or 24 kHz; 16 kHz is resampled to 24 kHz. The preparer requires one
speaker, unique IDs, nonempty train/dev sets and a reference chosen from train.
It rejects cross-split group, normalized text, file, decoded PCM and resampled
PCM duplicates. Group identifiers still need to reflect the real recording
source; these checks do not detect all near-duplicates.

```bash
python scripts/prepare_split_manifest.py \
  --model_path "$MODEL_PATH" \
  --train_jsonl /data/splits/train.jsonl \
  --dev_jsonl /data/splits/dev.jsonl \
  --reference_id train-001 \
  --output_dir /data/derived/qwen3-tts
```

The output directory must be new. The script preserves the assigned split,
text and order; it never silently resplits or edits source audio. Output rows
contain `id`, `text`, integer `audio_codes` of shape `[T,16]` in `[0,2048)`,
and `ref_mel_path`. One train-reference mel `[F,128]` is shared by both splits.
`provenance.json` records input/output hashes and the source mapping.
The dev manifest is prepared for downstream evaluation and is **not** consumed
by this training loop. No dev loss, early stopping or quality score is implied.

## Train and resume

```bash
python scripts/train_qwen3_tts_lora.py \
  --model_path "$MODEL_PATH" \
  --train_jsonl /data/derived/qwen3-tts/train.jsonl \
  --output_dir outputs/run-1 \
  --epochs 3 --grad_accum 4 --shuffle --seed 42 \
  --lr 1e-4 --lr_decay 1.0 --lora_r 8 --lora_dropout 0.1 \
  --save_interval 10
```

These are example hyperparameters, not a convergence recommendation. The
default learning rate stays constant. `--lr_decay` optionally supplies StepLR
gamma per optimizer update; the short resume regression explicitly uses 0.8
to exercise scheduler restoration. LoRA targets `q_proj`/`v_proj` in both
branches, with alpha equal to twice the rank. All base parameters are frozen.

Micro-batch size is one. The loop owns accumulation windows and keeps
Twinkle's group accumulation at one; `clip_grad_and_step` normalizes by the
actual number of micro-batches, including an epoch's short final window.
The speaker encoder runs without gradients. The Talker loss uses its internal
temporal shift; residual-codebook logits are supervised against corresponding
codebooks 1–15 without another shift along the codebook axis. Tests check
last-head gradients and the official residual assembly interface. This loss
choice is not a claim about how official pretrained weights were trained.

For a controlled stop, add `--stop_after_updates 1`. Resume to a **new** output
directory with the same training options:

```bash
python scripts/train_qwen3_tts_lora.py \
  --model_path "$MODEL_PATH" \
  --train_jsonl /data/derived/qwen3-tts/train.jsonl \
  --output_dir outputs/run-2 --resume_from outputs/run-1/checkpoint-update-1 \
  --epochs 3 --grad_accum 4 --shuffle --seed 42 \
  --lr 1e-4 --lr_decay 1.0 --lora_r 8 --lora_dropout 0.1 \
  --save_interval 10
```

Checkpoints include adapter, optimizer, scheduler, RNG and loop progress.
Resume checks content fingerprints of model/manifest/reference mel, sample
plan and settings; it fails on mismatch instead of silently starting over.
Model hashing reads the local weight files and may take time. Only complete,
trusted checkpoints produced by this workflow should be resumed. Saving is
at update boundaries; recovery from arbitrary interruption during a checkpoint
write is not guaranteed. Inspect `trace.jsonl`, `run_config.json` and
`result.json` for the actual update count and checkpoint path.

## Tests and evidence boundaries

With the optional dependencies installed:

```bash
python tests/run.py
MODEL_PATH="$MODEL_PATH" python tests/run.py --npu
```

The first command runs data, forward/loss, projection, manifest, preparation
and accumulation contracts. It includes official mel/residual-interface tests
but uses small CPU networks, not the pretrained TTS model. The second also
loads the real local model for a LoRA update/save/reload, public split
preparation-to-training, and continuous versus independent-process resume.
The two tiny LibriSpeech fixtures are attributed under
`tests/fixtures/librispeech/`; they are regression fixtures, not an independent
quality benchmark. Tests use temporary outputs and never require user data.

For local CPU development without Twinkle or Qwen installed, install torch,
Transformers, NumPy, scipy and soundfile, then run `python tests/run.py
--isolated`. This mode explicitly substitutes Twinkle infrastructure and skips
two official-package checks. It is not NPU or native Twinkle validation.

The CPU suite checks data, losses and boundary contracts. The opt-in NPU
suite exercises local-model parameter updates, reload and process recovery.
Passing these tests does not establish voice quality or convergence on a new
dataset. Select training settings and checkpoints using held-out speech.

## Integration boundaries

The local `tts/` package is added to the script path; it is not registered as
a new global Twinkle task. The subclass reuses forward/backward, optimizer,
clipping, saving and resume APIs. It currently overrides
`_construct_default_optimizer_group` for consistent tokenizer/processor/loss
defaults and calls `_lazy_wrap_model` when inspecting/loading wrapped weights.
These internal dependencies are isolated here and should be reviewed when
Twinkle changes. A public generic TTS abstraction should only be added if
maintainers identify a reusable need; this cookbook does not add one prematurely.
