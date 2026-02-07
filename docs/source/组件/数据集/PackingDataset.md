# 固定长度装箱数据集

装箱数据集用于将不定长的数据拼接到指定长度。例如：

数据集中包含4条长度为5的数据，而Template的组件max_length可接受长度为10，则装箱数据集会将数据预取出来，并拼接成为2条长度为10的样本。

```text
ABCDE
FGHIJ
KLMNO
PQRST
```

会被转换为
```text
ABCDEFGHIJ
KLMNOPQRST
```
注意这种拼接是在`encode`之后的，即实际的模型输入长度上。在流程中，数据集会进行如下操作：

1. 取出`buffer length`个样本
2. 对这些样本进行encode
3. 根据每个样本的长度进行自动装箱算法计算，寻找一个最优解，使批数量最小，每个样本的长度最接近`max_length`
4. 增加`position_ids`字段以区分不同样本。

最后形成的数据格式类似：

```json
{
  "input_ids": [1,2,3,4,5,6,7,8,9,10],
  "position_ids": [0,1,2,3,4,0,1,2,3,4],
  ...
}
```

数据集的使用上和`Dataset`有以下区别：

1. 必须设置`Template`
2. 调用`encode`之后需要调用`pack_dataset`方法来进行最后的装箱

```python
dataset.pack_dataset()
```

本数据集也有`@remote_class`装饰器，可以在ray的worker中运行。


