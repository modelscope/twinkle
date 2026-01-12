# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib
import inspect
import os
import sys
from typing import Type, TypeVar

from ..hub import MSHub, HFHub
from .unsafe import trust_remote_code

T = TypeVar('T')


class Plugin:
    """A plugin class for loading plugins from hub."""

    @staticmethod
    def load_plugin(plugin_id: str, plugin_base: Type[T], **kwargs) -> Type[T]:
        if plugin_id.startswith('hf://'):
            plugin_dir = HFHub.download_model(plugin_id[len('hf://'):], **kwargs)
        elif plugin_id.startswith('ms://'):
            plugin_dir = MSHub.download_model(plugin_id[len('ms://'):], **kwargs)
        else:
            raise ValueError(f'Unknown plugin id {plugin_id}, please use hf:// or ms://')

        if not trust_remote_code():
            raise ValueError(
                "Twinkle does not support plugin in safe mode."
            )

        if plugin_dir not in sys.path:
            sys.path.insert(0, plugin_dir)
        plugin_file = os.path.join(plugin_dir, '__init__.py')
        assert os.path.isfile(plugin_file), f'Plugin file {plugin_file} does not exist.'
        plugin_module = importlib.import_module(plugin_file)
        module_classes = {
            name: plugin_cls
            for name, plugin_cls in inspect.getmembers(plugin_module,
                                                      inspect.isclass)
        }
        sys.path.remove(plugin_dir)
        for name, plugin_cls in module_classes.items():
            if plugin_base in plugin_cls.__mro__[
                        1:] and plugin_cls.__module__ == plugin_file:
                return plugin_cls
        raise ValueError(f'Cannot find any subclass of {plugin_base.__name__}.')
