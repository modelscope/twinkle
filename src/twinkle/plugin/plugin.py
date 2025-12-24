import importlib
import inspect
import os
from typing import Type, TypeVar

from .. import remote_class, remote_function
from ..hub import MSHub, HFHub
import sys

T = TypeVar('T')


@remote_class()
class Plugin:

    def __init__(self, plugin_id: str, **kwargs):
        if plugin_id.startswith('hf://'):
            self.plugin_dir = HFHub.download_model(plugin_id[len('hf://'):], **kwargs)
        elif plugin_id.startswith('ms://'):
            self.plugin_dir = MSHub.download_model(plugin_id[len('ms://'):], **kwargs)
        else:
            raise ValueError(f'Unknown plugin id {plugin_id}, please use hf:// or ms://')

    @remote_function()
    def load_plugin(self, plugin_base: Type[T]) -> Type[T]:
        if self.plugin_dir not in sys.path:
            sys.path.insert(0, self.plugin_dir)
        plugin_file = os.path.join(self.plugin_dir, '__init__.py')
        assert os.path.isfile(plugin_file), f'Plugin file {plugin_file} does not exist.'
        plugin_module = importlib.import_module(plugin_file)
        module_classes = {
            name: plugin_cls
            for name, plugin_cls in inspect.getmembers(plugin_module,
                                                      inspect.isclass)
        }

        for name, plugin_cls in module_classes.items():
            if plugin_base in plugin_cls.__mro__[
                        1:] and plugin_cls.__module__ == plugin_file:
                return plugin_cls
        raise ValueError(f'Cannot find any subclass of {plugin_base.__name__}.')
