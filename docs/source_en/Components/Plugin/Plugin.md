# Plugin

Most components in Twinkle can be passed in externally. Some components support downloading from the ModelScope or Hugging Face community.

| Component Name        | Supported Input Methods | Supports Functions |
|-----------------------|--------------------|--------|
| InputProcessor        | modelhub download/class/instance/class name | Yes      |
| Metric                | modelhub download/class/instance/class name | No      |
| Loss                  | modelhub download/class/instance/class name | Yes      |
| Preprocessor          | modelhub download/class/instance/class name | Yes      |
| Filter                | modelhub download/class/instance/class name | Yes      |
| Template              | modelhub download/class/instance/class name | No      |
| Patch                 | modelhub download/class/instance/class name | Yes      |
| Optimizer/LrScheduler | modelhub download/class/instance/class name | No      |

## Writing Plugins

Components that support functions in the above table can use a single function to pass into the class that calls it, for example:

```python
def my_custom_preprocessor(row):
    return ...

dataset.map(my_custom_preprocessor)
```

If you need to upload the plugin to modelhub and download it for subsequent use, you cannot use the function method and must inherit the corresponding base class.

Let's take Preprocessor as an example to give a basic plugin writing method:

```python
# __init__.py
from twinkle.preprocessor import Preprocessor

class CustomPreprocessor(Preprocessor):

    def __call__(self, row):
        # Your custom code here
        return ...
```

Note that in the plugin's __init__.py, you need to write/reference your corresponding plugin class, and then provide a README.md that matches the plugin's function, and you can use this plugin.

```python
# Assuming model-id is MyGroup/CustomPreprocessor
dataset.map('ms://MyGroup/CustomPreprocessor')
# Or hf
dataset.map('hf://MyGroup/CustomPreprocessor')
```

# Service Security

Twinkle is a framework that supports service-oriented training. Loading plugins from the client or Callable code poses certain risks to the server. You can use `TWINKLE_TRUST_REMOTE_CODE` to prohibit them:

```python
import os

os.environ['TWINKLE_TRUST_REMOTE_CODE'] = '0'
```

By setting this environment variable to 0 (default is `1`), you can prohibit externally passed classes, Callable or network plugins to prevent the possibility of server attacks.
