# Public speech regression fixture

Utterance: LibriSpeech dev-clean **1272-128104-0000**.
Prepared by Vassil Panayotov with assistance from Daniel Povey; derived from LibriVox audiobooks.
Corpus: https://www.openslr.org/12 — **CC BY 4.0**: https://creativecommons.org/licenses/by/4.0/

Obtained from the Hugging Face internal testing mirror, clean/validation row 0:
https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy
Mirror revision: 5be91486e11a2d616f4ec5db8d3fd248585ac07a.
The checked-in FLAC is the viewer's audio export, with no local trimming or editing.
The transcript is preserved verbatim in sample.json. The test resamples to 24 kHz at runtime.

This public **dev-clean** example is exclusively a regression fixture. Do not include it
in customer training/evaluation datasets or claim held-out evaluation after using it to update a model.
A matching corpus transcript is not a new manual transcription or an audio quality rating.

## Additional process-resume fixture (2026-10-26)

`second/1272-128104-0001.flac`: row 1 of the same dataset/revision/split, same attribution and CC BY 4.0 license. SHA256 `46a9d58622b4675b29564da2d9ba73e702241c5fa969f12c387cad4aa984276a`. No local trimming or text edits; runtime resamples 16 kHz to 24 kHz. Full transcript and provenance are in `second/sample.json`. This is a regression fixture, not held-out evaluation data.
