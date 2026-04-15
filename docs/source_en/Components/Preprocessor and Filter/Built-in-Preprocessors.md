# Built-in Preprocessors

Twinkle provides a collection of built-in preprocessors for common dataset formats. Each converts raw data into standardized `Trajectory` objects.

## LLM Preprocessors

### CompetitionMathProcessor

Converts competition math datasets with `problem` and `solution` fields.

```python
dataset.map('CompetitionMathProcessor')
# Input: {'problem': '...', 'solution': '...'}
# Output: Trajectory with user message (problem) and assistant message (solution)
```

### CompetitionMathGRPOProcessor

Similar to CompetitionMathProcessor but stores the solution in `user_data` for use as ground truth in GRPO reward computation.

```python
dataset.map('CompetitionMathGRPOProcessor')
```

### SelfCognitionProcessor

Replaces template placeholders with model identity information for self-cognition training.

```python
dataset.map('SelfCognitionProcessor', model_name='MyModel', model_author='MyOrg')
```

### AlpacaProcessor

Converts Alpaca-format datasets with `instruction`, `input`, and `output` fields.

```python
dataset.map('AlpacaProcessor')
# Input: {'instruction': '...', 'input': '...', 'output': '...'}
```

### CountdownProcessor

Generates countdown arithmetic problems for reasoning training.

```python
dataset.map('CountdownProcessor')
```

### GSM8KProcessor

Preprocesses GSM8K math datasets, extracting ground truth answers from the `#### answer` format.

```python
dataset.map('GSM8KProcessor')
# Extracts answer from '#### 42' format and stores in user_data
```

## DPO Preprocessor

### EmojiDPOProcessor

Converts emoji-based preference datasets into positive/negative trajectory pairs for DPO training.

```python
dataset.map('EmojiDPOProcessor')
# Input: {'prompt': '...', 'chosen': '...', 'rejected': '...'}
# Output: Interleaved chosen and rejected Trajectory pairs
```

## Multimodal Preprocessors

### CLEVRProcessor

Preprocesses CLEVR visual reasoning datasets with image handling.

```python
dataset.map('CLEVRProcessor')
# Input: {'question': '...', 'answer': '...', 'image': PIL.Image}
# Output: Trajectory with multimodal content (image + text)
```

### OlympiadBenchProcessor

Preprocesses OlympiadBench multimodal math/physics problems with image collection and metadata storage.

```python
dataset.map('OlympiadBenchProcessor')
# Handles multiple images per problem, stores ground truth and metadata in user_data
```

> All preprocessors follow the same interface: `__call__(rows) -> List[Trajectory]`. You can register custom preprocessors following the same pattern (see [Preprocessor](Preprocessor.md)).
