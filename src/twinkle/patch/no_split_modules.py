from typing import Set, Union

from twinkle.patch import Patch


class NoSplitModulesPatch(Patch):
    """Set _no_split_modules on a model so FSDP2 respects layer boundaries."""

    def __init__(self, module_names: Union[Set[str], str] = frozenset({'Qwen3_5DecoderLayer'})):
        if isinstance(module_names, str):
            module_names = {module_names}
        self._names = set(module_names)

    def __call__(self, module, *args, **kwargs):
        module._no_split_modules = self._names
        return module

    def unpatch(self, module, *args, **kwargs):
        if hasattr(module, '_no_split_modules'):
            del module._no_split_modules
        return module
