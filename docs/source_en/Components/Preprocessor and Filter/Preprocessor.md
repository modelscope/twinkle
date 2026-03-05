# Preprocessor

The preprocessor is a script for data ETL. Its role is to convert messy, uncleaned data into standardized, cleaned data. The preprocessing method supported by Twinkle runs on the dataset.map method.

The base class of Preprocessor:

```python
class Preprocessor:

    def __call__(self, rows: List[Dict]) -> List[Trajectory]:
        ...
```

The format is to pass in a list of samples and output a list of `Trajectory`. If a sample cannot be used, you can directly ignore it.

We provide some basic Preprocessors, such as `SelfCognitionProcessor`:

```python
dataset.map('SelfCognitionProcessor', model_name='some-model', model_author='some-author')
```

Preprocessor contains the __call__ method, which means you can use a function to replace the class:

```python
def self_cognition_preprocessor(rows):
    ...
    return [Trajectory(...), ...]
```
