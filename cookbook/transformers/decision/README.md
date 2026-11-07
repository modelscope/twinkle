# Typed decision training

Twinkle can train and score typed decisions through its existing TransformersModel, InputProcessor,
Loss and PEFT interfaces. Predictions are probabilities over each question's legal options;
there is no autoregressive answer-token generation. A probability is not automatically calibrated:
retain the checkpoint's calibration settings, or fit new settings on a disjoint calibration split.

## 1. Model and task

| Family | Execution | Head | Loss preset |
| --- | --- | --- | --- |
| generic CausalLM | One question per prompt | Tokenizer-derived verbalizer | scoring |
| JEV | One question per prompt; text/image input | Released verbalizer and bias | distill_rps |
| Clef | All questions in a record jointly; text input | Span pooling and joint schema decoder | smooth_brier |
| OmniJev | Cached prefix and separate option branches; text/image/precomposed video | FiLM and ordinal head | ce_rps |
| StartLux | One question per prompt; text input | Last-position FP32 A-Z projection | scoring |

`noul`, `choice`, and `score` use different option orders across families. The preprocessor preserves
those orders and remaps targets. Do not substitute one family's prompt or calibration for another.
Generic/JEV verbalizers require distinct single-token symbols and allow up to 16 choice options.
StartLux allows 2–26 options; wide choices must be explicitly decomposed outside this recipe.

## 2. Validation and scope

| Check | Evidence / scope |
| --- | --- |
| Loss and gradients | Four presets compared against independent unpadded formulas at 1e-12 |
| Accumulation | Uneven question counts match the full-batch normalized gradient |
| LoRA persistence | Tiny CausalLM updates both adapter and head; save/reload logits match exactly on CPU |
| Head contracts | Clef joint spans/padding, OmniJev FiLM/ordinal gradients, JEV image payload |
| Regression suite | Decision loss, encoding, augmentation and checkpoint regression tests; full repository pre-commit checks |
| Native training API | Tiny model forward/backward, optimizer step, save, resume from step 3 to 6 and reload; StartLux-4B two-step training, save and adapter reload/scoring on one Ascend NPU |
| Released-model quality | Full checkpoint-specific accuracy and visual benchmark validation pending |

Tests use synthetic fixtures and do not establish released-checkpoint quality. Clef image span
alignment, packing, sequence/context parallelism and multi-adapter serving are not supported here.
The example uses one device; multi-rank training requires separate sharding and checkpoint validation.

## 3. Installation and reproducibility

Develop from a checkout and install its declared dependencies without changing repository bounds:

```bash
python -m pip install -e . pytest
python -m pytest tests/decision -q --confcutdir=tests/decision
pre-commit run --all-files
```

An Ascend validation environment uses the base image below (no image export required), Python 3.12,
Torch 2.10.0, torch-npu 2.10.0.post4, CANN 9.1.0, Transformers 5.14.1 and PEFT 0.19.0.
Use a new container, mount only assigned devices and a task directory, and install Twinkle into an
isolated virtual environment. The image also contains unrelated inference packages; installing
training dependencies does not validate those packages' inference environments.

```text
quay.io/ascend/vllm-ascend@sha256:8a397e82a22f55de25ae355a957b91e843a16ce704a03badc0a8e9915f68d84f
```

If the container exposes one physical NPU, its runtime-visible ordinal is `0`, even when its host
card number differs. Record both device identities in private deployment notes.

## 4. Train, resume and score

No model or dataset downloads are needed for this synthetic smoke test; dependencies must already
be installed. Use a fresh output directory for every run.

```bash
python cookbook/transformers/decision/make_tiny.py --output /tmp/decision-tiny
python cookbook/transformers/decision/decision.py train \
  --model /tmp/decision-tiny --data /tmp/decision-tiny/mini.jsonl \
  --output /tmp/decision-train --steps 3 --dtype float32
python cookbook/transformers/decision/decision.py infer \
  --model /tmp/decision-tiny --adapter /tmp/decision-train/final \
  --data /tmp/decision-tiny/mini.jsonl --output /tmp/decision-infer --dtype float32
python cookbook/transformers/decision/decision.py train \
  --model /tmp/decision-tiny --resume /tmp/decision-train/final \
  --data /tmp/decision-tiny/mini.jsonl --output /tmp/decision-resume --steps 6 --dtype float32
```

`--adapter` initializes weights; `--resume` restores training state. They are mutually exclusive.
`steps` is the total target number of updates. The recipe records consumed examples with the
checkpoint; preserve data ordering and augmentation settings when resuming.

| Released family | Arguments to add to the same train/infer command |
| --- | --- |
| JEV | `--family jev --model <BASE_CHECKPOINT> --head-dir <JEV_RELEASE> --adapter <JEV_RELEASE>/adapter_vllm` |
| Clef | `--family clef --model <MERGED_CLEF_CHECKPOINT> --head-dir <CLEF_HEAD_DIRECTORY>` |
| OmniJev | `--family omnijev --model <BASE_CHECKPOINT> --head-dir <OMNIJEV_RELEASE> --adapter <OMNIJEV_ADAPTER>` |
| StartLux | `--family startlux --model <STARTLUX_CHECKPOINT>` |

For JEV/OmniJev, keep the released adapter: omitting it discards the released decision capability.
Use the matching tokenizer, head artifacts and calibration. StartLux also requires the official
`startlux_decision` Python package on PYTHONPATH. The validated source revision is
`StartLuxLabs/StartLux-Decision@0e7a2e81b9c92756e26d8edd843a44d50e362669`.
For StartLux 4B/35B, select language attention targets and exclude the vision tower and vocabulary
head; do not claim training benefit from a smoke test.

Training accepts hard labels or finite nonnegative distributions summing to one. Targets are never
rendered into model inputs. `--shuffle-fields`, `--drop-instructions` and `--subsample-options`
are explicit training-only augmentation options for generic/JEV/Clef/OmniJev; StartLux rejects these flags. Disable augmentation for evaluation.
Inference writes `predictions.jsonl` and `metrics.json` with per-question `acc` and per-encoded-record
`record_acc`. A JEV/OmniJev source record is expanded before this accounting.

Open-Jev is an explicit online download, separate from the offline smoke dataset:

```bash
python cookbook/transformers/decision/fetch_open_jev.py \
  --dataset ZefanCai/Open-Jev --revision <DATASET_REVISION> \
  --subset release-v2-redistributable --split train --output /tmp/open-jev-train.jsonl
# Use --open-jev-format --data /tmp/open-jev-train.jsonl with train or infer.
```

Respect the dataset's license and preserve separate training, validation, calibration and test splits.
Image rows accept local images. OmniJev videos must first be converted to a frame mosaic with
`twinkle.model.decision.omnijev_media.video_state`; this needs ffmpeg/ffprobe. Passing raw video paths
into the training encoder raises an error.

## 5. Acceptance before publication

| Gate | Required evidence |
| --- | --- |
| Input | Exact prompt IDs, option order, target exclusion, image grids and branch spans |
| Numeric | Head logits/probabilities and loss/gradient comparisons against each released reference |
| Training | Finite updates, adapter/head changes, frozen-backbone audit, checkpoint recovery |
| Quality | Independent held-out scoring; report each suite and its denominator |
| Performance | Same device, precision, input and warmup; keep performance artifacts local |

The scoring head is saved through PEFT `modules_to_save`; vocabulary embeddings are weak references
and are not duplicated into head state. FP32 activation must happen after adapter creation and before
the optimizer. Source attribution is retained in reused numerical modules.

## 6. Summary

Use the common Twinkle training API while retaining each model's decision format and head semantics.
Passing the included contract tests is an engineering milestone; it does not replace full released
model accuracy, multimodal, distributed or performance acceptance.
