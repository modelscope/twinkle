# Copyright (c) ModelScope Contributors. All rights reserved.
import hashlib
import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Type, TypeVar, Union

from ..hub import HFHub, MSHub
from .unsafe import trust_remote_code

T = TypeVar('T')

#: Resolved plugin path (or hub id) -> imported module, so loading the same plugin twice is a no-op
#: rather than a re-import, and two different plugins never collide on a shared module name.
_LOADED_MODULES: Dict[str, ModuleType] = {}


def _unique_module_name(path: Path) -> str:
    """A unique, path-derived module name for a plugin file.

    twinkle used to import every plugin under the fixed name ``__init__``, so a second plugin hit
    ``sys.modules`` and silently handed back the first one's classes -- a collision that reads as "my
    plugin was ignored". Deriving the name from the absolute path keeps distinct plugins distinct and
    makes reloading the same path idempotent.
    """
    return f'twinkle_plugin_{hashlib.sha1(str(path).encode()).hexdigest()[:8]}_{path.stem}'


def _load_local(path: Path) -> ModuleType:
    """Import a local ``.py`` file, or a local folder via its ``__init__.py``, under a unique name.

    The file's directory joins ``sys.path`` so a plugin may import its own neighbours; it is left in
    place, since a folder plugin's siblings must stay importable for the process lifetime.
    """
    path = path.expanduser().resolve()
    if path.is_dir():
        folder, target = path, path / '__init__.py'
        if not target.is_file():
            raise FileNotFoundError(f'plugin folder {folder} has no __init__.py.')
    elif path.is_file():
        folder, target = path.parent, path
    else:
        raise FileNotFoundError(f'plugin path {path} is neither a file nor a folder.')

    key = str(target)
    if key in _LOADED_MODULES:
        return _LOADED_MODULES[key]
    module_name = _unique_module_name(target)
    if module_name in sys.modules:
        module = sys.modules[module_name]
        _LOADED_MODULES[key] = module
        return module

    parent = str(folder)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(module_name, key)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    _LOADED_MODULES[key] = module
    return module


def _load_hub(plugin_id: str, **kwargs) -> ModuleType:
    """Download a ``hf://`` / ``ms://`` plugin and import it. Refused in safe mode."""
    if not trust_remote_code():
        raise ValueError('Twinkle does not support plugin in safe mode.')
    if plugin_id.startswith('hf://'):
        plugin_dir = HFHub.download_model(plugin_id[len('hf://'):], **kwargs)
    elif plugin_id.startswith('ms://'):
        plugin_dir = MSHub.download_model(plugin_id[len('ms://'):], **kwargs)
    else:
        raise ValueError(f'Unknown plugin id {plugin_id}, please use hf:// or ms://')
    return _load_local(Path(plugin_dir))


def load_module(source: str, **kwargs) -> ModuleType:
    """The single plugin-loading entry point: a local file, a local folder, or a hub id -> a module.

    swift and twinkle share this one loader, so a plugin source is written once and resolved the same
    way everywhere:
      - a local ``.py`` file, or a folder holding ``__init__.py``, is imported directly. A local path
        is something the user handed us on their own machine, so it loads WITHOUT the hub's
        trust_remote_code gate;
      - ``hf://`` / ``ms://`` is downloaded first (and refused in safe mode), then imported.

    Idempotent: each resolved path is imported once under a unique, path-derived module name and
    cached in ``_LOADED_MODULES``, so loading the same plugin twice -- or two different plugins --
    never collides. ``construct_class`` falls back here for a string it cannot find in the kernel
    namespace; swift's ``PluginRegistry.load_external`` calls it to run a plugin file's ``@register``.
    """
    if source.startswith('hf://') or source.startswith('ms://'):
        return _load_hub(source, **kwargs)
    return _load_local(Path(source))


class Plugin:
    """Resolve a plugin source to a concrete subclass of a kernel base class."""

    @staticmethod
    def load_plugin(plugin_id: str, plugin_base: Type[T], **kwargs) -> Type[T]:
        """``plugin_id`` (local file/folder or ``hf://``/``ms://`` id) -> the ``plugin_base`` subclass it defines.

        Only classes DEFINED in the loaded module count (``__module__`` match), so a plugin that merely
        imports a base or a sibling subclass does not have that re-exported class mistaken for its own.
        """
        plugin_module = load_module(plugin_id, **kwargs)
        for _, plugin_cls in inspect.getmembers(plugin_module, inspect.isclass):
            if plugin_base in plugin_cls.__mro__[1:] and plugin_cls.__module__ == plugin_module.__name__:
                return plugin_cls
        raise ValueError(f'Cannot find any subclass of {plugin_base.__name__} in {plugin_id}.')


def construct_class(func: Union[str, Type[T], T], class_T: Type[T], module_T: Union[List[ModuleType], ModuleType],
                    **init_args) -> T:
    """Try to load a class.

    Args:
        func: The input class or class name/plugin name to load instance from
        class_T: The base class of the instance
        module_T: The module of the class_T
        **init_args: The args to construct the instruct
    Returns:
        The instance
    """
    if not isinstance(module_T, list):
        module_T = [module_T]
    if isinstance(func, class_T):
        # Already an instance
        return func
    elif isinstance(func, type) and issubclass(func, class_T):
        # Is a subclass type
        return func(**init_args)
    elif isinstance(func, str):
        # Is a subclass name, or a plugin name
        for module in module_T:
            if hasattr(module, func):
                cls = getattr(module, func)
                break
        else:
            cls = Plugin.load_plugin(func, class_T)
        return cls(**init_args)
    else:
        # Do nothing by default
        return func
