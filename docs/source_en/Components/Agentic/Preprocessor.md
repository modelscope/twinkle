# Agentic Preprocessor

The agentic preprocessor module provides a pipeline-based data quality filtering framework for multi-turn conversation datasets. It is designed for cleaning and filtering training data before RLHF / agentic fine-tuning.

## QualityPreprocessor

`QualityPreprocessor` is a thin pipeline runner that accepts a list of filter callables and runs them in sequence. Each step receives a list of rows, returns `(kept, dropped)`, and the pipeline logs per-step statistics.

```python
from twinkle_agentic.preprocessor import QualityPreprocessor, HardFilter, DeadLoopFilter

pipeline = [
    HardFilter(min_user_chars=10),
    DeadLoopFilter(),
]
preprocessor = QualityPreprocessor(pipeline, dropped_log_path='dropped.jsonl')

# rows is a dict of columns (Dataset.map format)
cleaned = preprocessor(rows)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `pipeline` | `List[Callable]` | Ordered list of filter steps. Each step takes `List[Dict]` and returns `(kept, dropped)`. |
| `dropped_log_path` | `str` | Optional JSONL file path for logging dropped rows with step name and reason. |

## Built-in Filters

### HardFilter

Rule-based filter that removes trivially bad rows using deterministic rules. Supports multi-language detection (EN/ZH/JA/KO).

```python
from twinkle_agentic.preprocessor import HardFilter

f = HardFilter(
    min_user_chars=10,           # Min chars for non-CJK user query
    min_user_chars_cjk=6,        # Min chars for CJK user query
    min_assistant_chars_2turn=80, # Min assistant reply length (2-turn)
    min_thinking_chars=200,      # Min thinking chain length to exempt
    system_deny_keywords=['hack', 'exploit'],
    max_chars_per_round=50000,
    max_total_chars=200000,
    max_rounds=50,
)
```

**Drop reasons:** `trivial_single_turn`, `shallow_reply`, `all_empty_assistant`, `system_deny_keyword`, `round_too_long`, `total_too_long`, `too_many_rounds`

### DeadLoopFilter

Detects assistant messages exhibiting hesitation or dead-loop patterns — repetitive self-corrections, cascading corrections, and high n-gram repetition.

```python
from twinkle_agentic.preprocessor import DeadLoopFilter

f = DeadLoopFilter(
    hesitation_density_threshold=7.0,   # Markers per 1000 chars (response)
    cascade_threshold=5,                 # Cascade markers in window
    cascade_window=800,                  # Window size in chars
    repetition_threshold=0.45,           # N-gram repetition ratio
    think_hesitation_density_threshold=15.0,  # Laxer for <think> blocks
    think_repetition_threshold=0.65,
)
```

Uses separate threshold profiles for `<think>` reasoning blocks (laxer, free to ramble) and visible response (stricter).

### DedupFilter

Global longest-wins deduplication. The signature is derived from the first real user turn (head+tail) and the first assistant reply.

```python
from twinkle_agentic.preprocessor import DedupFilter

f = DedupFilter(prefix_chars=100, asst_chars=100)
kept, dropped = f(all_rows)  # Must see entire dataset in one call
```

> **Note:** `DedupFilter` requires the full dataset in a single call. Do **not** place it inside `QualityPreprocessor` (which processes per-batch). Run it separately before or after the pipeline.

### RefuseFilter

Detects self-referential refusals in the first assistant reply (e.g., "I cannot help with that"). Multi-language pattern matching (EN/ZH/JA/KO).

```python
from twinkle_agentic.preprocessor import RefuseFilter

f = RefuseFilter(check_window=600)  # Only check first N chars
```

### TokenSoupFilter

Detects garbled / token-soup output by checking for replacement characters, control characters, private-use Unicode, leaked special tokens, single-character repetition, and script chaos.

```python
from twinkle_agentic.preprocessor import TokenSoupFilter

f = TokenSoupFilter(
    replacement_char_ratio=0.02,
    special_token_count=20,
    script_chaos_threshold=0.55,
)
```

### PIIPresidioFilter

Multi-language PII detection and rewriting using Microsoft Presidio + spaCy NER + Faker. Detects and replaces personal identifiable information (names, emails, phone numbers, addresses, etc.).

```python
from twinkle_agentic.preprocessor import PIIPresidioFilter

f = PIIPresidioFilter(languages=['en', 'zh'])
```

### IntentClassifier

Heuristic intent classifier that tags each row with detected intents. Pluggable detector pipeline.

```python
from twinkle_agentic.preprocessor import IntentClassifier

classifier = IntentClassifier()
```

**Intent categories:** `tool_call`, `code`, `math`, `complex_logic`, `reasoning`, `user_dissatisfaction`, `other`

### ScoreFilter

Pluggable scorer-based filter with built-in scorers for character-level metrics, semantic similarity, and code execution.

```python
from twinkle_agentic.preprocessor import ScoreFilter

f = ScoreFilter()
```

**Built-in scorers:** `ChrMinScorer`, `SIFDScorer`, `PassNScorer`, `ParaphraseScorer`

### ModelFilter

Filters rows by model ID whitelist.

```python
from twinkle_agentic.preprocessor import ModelFilter

f = ModelFilter(allowed_models=['qwen3.5-4b', 'qwen3.5-32b'])
```

### MessageNormalizer

Three-pass message normalization: heartbeat stripping, tool-call rewriting, and consecutive same-role message merging.

```python
from twinkle_agentic.preprocessor import MessageNormalizer

normalizer = MessageNormalizer()
```

## Complete Pipeline Example

```python
from twinkle_agentic.preprocessor import (
    QualityPreprocessor,
    HardFilter,
    DeadLoopFilter,
    RefuseFilter,
    TokenSoupFilter,
    MessageNormalizer,
    DedupFilter,
)

# Step 1: Global dedup (must run on full dataset)
dedup = DedupFilter()
rows, _ = dedup(all_rows)

# Step 2: Per-batch pipeline
pipeline = [
    HardFilter(min_user_chars=10, max_rounds=30),
    DeadLoopFilter(),
    RefuseFilter(),
    TokenSoupFilter(),
    MessageNormalizer(),
]
preprocessor = QualityPreprocessor(pipeline, dropped_log_path='dropped.jsonl')
cleaned = preprocessor(rows)
```
