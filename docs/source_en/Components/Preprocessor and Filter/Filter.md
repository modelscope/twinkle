# Filter

The preprocessor is a script for data ETL. Its role is to convert messy, uncleaned data into standardized, cleaned data. The preprocessing method supported by Twinkle runs on the dataset.map method.

The base class of Filter:

```python
class DataFilter:

    def __call__(self, row) -> bool:
        ...
```

The format is to pass in a raw sample and output a `boolean`. Filter can occur before or after Preprocessor, used in combination:
```python
dataset.filter(...)
dataset.map(...)
dataset.filter(...)
```

Filter contains the __call__ method, which means you can use a function to replace the class:

```python
def my_custom_filter(row):
    ...
    return True
```
