# Plugin

Twinkle中大部分组件均可以从外部传入使用。部分组件支持从ModelScope或Hugging Face社区下载使用。

| 组件名称                  | 支持的传入方式            | 是否支持函数 |
|-----------------------|--------------------|--------|
| InputProcessor        | modelhub下载/类/实例/类名 | 是      |
| Metric                | modelhub下载/类/实例/类名 | 否      |
| Loss                  | modelhub下载/类/实例/类名 | 是      |
| Preprocessor          | modelhub下载/类/实例/类名 | 是      |
| Filter                | modelhub下载/类/实例/类名 | 是      |
| Template              | modelhub下载/类/实例/类名 | 否      |
| Patch                 | modelhub下载/类/实例/类名 | 是      |
| Optimizer/LrScheduler | modelhub下载/类/实例/类名 | 否      |

## 编写插件

在上表中支持函数的组件都可以使用一个单独的函数传入调用它的类，例如：

```python
def my_custom_preprocessor(row):
    return ...

dataset.map(my_custom_preprocessor)
```

如果需要将插件上传到modelhub中并后续下载使用，则不能使用函数的方式，一定要继承对应的基类。

我们以Preprocessor为例，给出一个基本的插件编写方式：

```python
# __init__.py
from twinkle.preprocessor import Preprocessor

class CustomPreprocessor(Preprocessor):

    def __call__(self, row):
        # You custom code here
        return ...
```

注意，在插件的__init__.py中需要编写/引用你对应的插件类，之后给出一个符合插件作用的README.md之后，就可以使用这个插件了。

```python
# 假设model-id为MyGroup/CustomPreprocessor
dataset.map('ms://MyGroup/CustomPreprocessor')
# 或者hf
dataset.map('hf://MyGroup/CustomPreprocessor')
```

# 服务安全

Twinkle是一个支持服务化训练的框架。从客户端加载插件，或Callable代码对服务器存在一定的风险。此时可以使用`TWINKLE_TRUST_REMOTE_CODE`来禁止它们：

```python
import os

os.environ['TWINKLE_TRUST_REMOTE_CODE'] = '0'
```

通过设置这个环境变量为0（默认为`1`），可以禁止外部传入的类、Callable或网络插件，防止服务被攻击的可能性。
