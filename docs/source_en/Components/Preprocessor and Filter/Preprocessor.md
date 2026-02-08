# Preprocessor

The preprocessor is a script for data ETL. Its role is to convert messy, uncleaned data into standardized, cleaned data. The preprocessing method supported by Twinkle runs on the dataset.map method.

The base class of Preprocessor:

```python
class Preprocessor:

    def __call__(self, row) -> Trajectory:
        ...
```

The format is to pass in a raw sample and output a `Trajectory`. If the sample cannot be used, you can directly return None.

We provide some basic Preprocessors, such as `SelfCognitionProcessor`:

```python
dataset.map('SelfCognitionProcessor', model_name='some-model', model_author='some-author')
```

Preprocessor contains the __call__ method, which means you can use a function to replace the class:

```python
def self_cognition_preprocessor(row):
    ...
    return Trajectory(...)
```
