# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING, Any

from ._lazy_module import _LazyModule  # noqa


def init_tinker_client(**kwargs) -> None:
    """Compatibility entry point; prefer ``twinkle_client.init_tinker_client``."""
    from twinkle_client import init_tinker_client as _init_tinker_client
    return _init_tinker_client(**kwargs)


def init_twinkle_client(
    base_url: str | None = None,
    api_key: str | None = None,
    session_heartbeat_interval: int = 10,
    **kwargs,
) -> Any:
    """Compatibility entry point; prefer ``twinkle_client.init_twinkle_client``."""
    from twinkle_client import init_twinkle_client as _init_twinkle_client
    return _init_twinkle_client(
        base_url=base_url,
        api_key=api_key,
        session_heartbeat_interval=session_heartbeat_interval,
        **kwargs,
    )


if TYPE_CHECKING:
    from .infra import get_device_placement, initialize, is_master, remote_class, remote_function
    from .utils import (GPU, NPU, DeviceGroup, DeviceMesh, Platform, Plugin, check_unsafe, exists, find_free_port,
                        find_node_ip, framework_util, get_logger, requires, torch_util, trust_remote_code)
    from .version import __release_datetime__, __version__
else:
    _import_structure = {
        'version': ['__release_datetime__', '__version__'],
        'utils': [
            'framework_util', 'torch_util', 'exists', 'requires', 'Platform', 'GPU', 'NPU', 'find_node_ip',
            'find_free_port', 'trust_remote_code', 'check_unsafe', 'DeviceMesh', 'Plugin', 'DeviceGroup', 'get_logger'
        ],
        'infra': ['initialize', 'remote_class', 'remote_function', 'get_device_placement', 'is_master'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,  # noqa
        extra_objects={
            'init_tinker_client': init_tinker_client,
            'init_twinkle_client': init_twinkle_client,
        },
    )
