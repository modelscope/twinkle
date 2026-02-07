# Preprocessor

预处理器是用于数据ETL的脚本。它的作用是将杂乱、未清洗的数据转换为标准化、清洗过的数据。twinkle支持的预处理方式是运行在dataset.map方法上。

Preprocessor的基类：

```python
class Preprocessor:

    def __call__(self, row) -> Trajectory:
        ...
```

格式为传入一个原始样本，输出一个`Trajectory`。如果样本无法使用，可以直接返回None。

我们提供了一些基本的Preprocessor，例如`SelfCognitionProcessor`:

```python
dataset.map('SelfCognitionProcessor', model_name='some-model', model_author='some-author')
```

Preprocessor包含__call__方法，这意味着你可以使用function来代替类：

```python
def self_cognition_preprocessor(row):
    ...
    return Trajectory(...)
```
