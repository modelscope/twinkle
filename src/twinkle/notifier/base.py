from typing import Dict, Optional


class Notifier:
    """Base class for notification sinks.

    Subclasses implement ``to_dict`` / ``_from_dict_impl`` for serialization.
    The registry auto-populates via ``__init_subclass__`` so ``from_dict``
    can dispatch by class name at restore time.
    """

    _registry: Dict[str, type] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Notifier._registry[cls.__name__] = cls

    def __call__(self, message: str):
        raise NotImplementedError

    def to_dict(self) -> Dict[str, str]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict) -> Optional['Notifier']:
        class_name = data.get('class')
        if not class_name:
            return None
        target = cls._registry.get(class_name)
        if target is None:
            return None
        return target._from_dict_impl(data)

    @classmethod
    def _from_dict_impl(cls, data: dict) -> Optional['Notifier']:
        raise NotImplementedError
