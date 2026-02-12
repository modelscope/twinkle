# Filter

预处理器是用于数据 ETL 的脚本。它的作用是将杂乱、未清洗的数据转换为标准化、清洗过的数据。Twinkle 支持的预处理方式是运行在 dataset.map 方法上。

Filter 的基类：

```python
class DataFilter:

    def __call__(self, row) -> bool:
        ...
```

格式为传入一个原始样本，输出一个`boolean`。Filter可以发生在Preprocessor的之前或之后，组合使用：
```python
dataset.filter(...)
dataset.map(...)
dataset.filter(...)
```

Filter 包含 __call__ 方法，这意味着你可以使用 function 来代替类：

```python
def my_custom_filter(row):
    ...
    return True
```
