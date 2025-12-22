import functools
from typing import Literal, List, Optional
from typing import Type, TypeVar, Tuple, overload

from .device_group import DeviceGroup
from ..utils import requires

_mode: Optional[Literal['local', 'ray']] = 'local'

_nproc_per_node: Optional[int] = 8

_device_group: Optional[List[DeviceGroup]] = None

_remote_components: dict = {}


def initialize(mode: Literal['local', 'ray'],
               nproc_per_node: Optional[int] = 8,
               groups: Optional[List[DeviceGroup]] = None,):
    global _mode, _device_group, _nproc_per_node
    assert mode in ('local', 'ray')
    _mode = mode
    if _mode == 'ray':
        requires('ray')
        _device_group = groups
        _nproc_per_node = nproc_per_node


def _get_remote_component(component):
    if component not in _remote_components:
        import ray
        _remote_components[component] = ray.remote(component)
    return _remote_components[component]


T1 = TypeVar('T1', bound=object)
T2 = TypeVar('T2', bound=object)
T3 = TypeVar('T3', bound=object)
T4 = TypeVar('T4', bound=object)
T5 = TypeVar('T5', bound=object)

@overload
def prepare(__c1: Type[T1], /) -> Tuple[Type[T1]]: ...

@overload
def prepare(__c1: Type[T1], __c2: Type[T2], /) -> Tuple[Type[T1], Type[T2]]: ...

@overload
def prepare(__c1: Type[T1], __c2: Type[T2], __c3: Type[T3], /) -> Tuple[Type[T1], Type[T2], Type[T3]]: ...

@overload
def prepare(__c1: Type[T1], __c2: Type[T2], __c3: Type[T3], __c4: Type[T4], /) -> Tuple[Type[T1], Type[T2], Type[T3], Type[T4]]: ...

@overload
def prepare(__c1: Type[T1], __c2: Type[T2], __c3: Type[T3], __c4: Type[T4], __c5: Type[T5], /) -> Tuple[Type[T1], Type[T2], Type[T3], Type[T4], Type[T5]]: ...

def prepare(*components, group_name: Optional[str]=None):
    _output = []
    for component in components:
        _output.append(prepare_one(component, group_name=group_name))
    return tuple(_output)


def prepare_one(component: Type[T1], group_name: Optional[str]=None) -> Type[T1]:

    class WrappedComponent:

        def __init__(self, *args, **kwargs):
            if _mode == 'local':
                self._actor = component(*args, **kwargs)
            elif _mode == 'ray':
                import ray
                from .ray import RayHelper
                RayHelper.initialize(_nproc_per_node, device_groups=_device_group)
                self._actor = RayHelper.create_workers(_get_remote_component(self._actor), group_name, *args, **kwargs)
            else:
                raise NotImplementedError(f'Unsupported mode {_mode}')

        def __getattr__(self, name):
            if _mode == 'local':
                return getattr(self._actor, name)
            elif _mode == 'ray':
                attr = getattr(self._actor[0], name)
                if callable(attr):
                    @functools.wraps(attr)
                    def wrapper(*args, **kwargs):
                        import ray
                        from .ray import RayHelper
                        return RayHelper.execute_all_sync(self._actor, dispatch=, execute=, method_name=name, *args, **kwargs)
                    return wrapper
                return attr
            else:
                raise NotImplementedError(f'Unsupported mode {_mode}')

        @property
        def actor(self):
            return self._actor

    return WrappedComponent
