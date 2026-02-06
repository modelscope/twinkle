# Filter

预处理器是用于数据ETL的脚本。它的作用是将杂乱、未清洗的数据转换为标准化、清洗过的数据。twinkle支持的预处理方式是运行在dataset.map方法上。

Filter的基类：

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

Filter包含__call__方法，这意味着你可以使用function来代替类：

```python
def my_custom_filter(row):
    ...
    return True
```
