# Plugin

Twinkle 中大部分组件均可以从外部传入使用。部分组件支持从 ModelScope 或 Hugging Face 社区下载使用。

| 组件名称                  | 支持的传入方式            | 是否支持函数 |
|-----------------------|--------------------|--------|
| InputProcessor        | modelhub 下载/类/实例/类名 | 是      |
| Metric                | modelhub 下载/类/实例/类名 | 否      |
| Loss                  | modelhub 下载/类/实例/类名 | 是      |
| Preprocessor          | modelhub 下载/类/实例/类名 | 是      |
| Filter                | modelhub 下载/类/实例/类名 | 是      |
| Template              | modelhub 下载/类/实例/类名 | 否      |
| Patch                 | modelhub 下载/类/实例/类名 | 是      |
| Optimizer/LrScheduler | modelhub 下载/类/实例/类名 | 否      |

## 编写插件

在上表中支持函数的组件都可以使用一个单独的函数传入调用它的类，例如：

```python
def my_custom_preprocessor(row):
    return ...

dataset.map(my_custom_preprocessor)
```

如果需要将插件上传到 modelhub 中并后续下载使用，则不能使用函数的方式，一定要继承对应的基类。

我们以 Preprocessor 为例，给出一个基本的插件编写方式：

```python
# __init__.py
from twinkle.preprocessor import Preprocessor

class CustomPreprocessor(Preprocessor):

    def __call__(self, row):
        # You custom code here
        return ...
```

注意，在插件的 __init__.py 中需要编写/引用你对应的插件类，之后给出一个符合插件作用的 README.md 之后，就可以使用这个插件了。

```python
# 假设 model-id 为 MyGroup/CustomPreprocessor
dataset.map('ms://MyGroup/CustomPreprocessor')
# 或者 hf
dataset.map('hf://MyGroup/CustomPreprocessor')
```

# 服务安全

Twinkle 是一个支持服务化训练的框架。从客户端加载插件，或 Callable 代码对服务器存在一定的风险。此时可以使用 `TWINKLE_TRUST_REMOTE_CODE` 来禁止它们：

```python
import os

os.environ['TWINKLE_TRUST_REMOTE_CODE'] = '0'
```

通过设置这个环境变量为 0（默认为 `1`），可以禁止外部传入的类、Callable 或网络插件，防止服务被攻击的可能性。
