# Copyright (c) ModelScope Contributors. All rights reserved.
import sys

from twinkle._lazy_module import _LazyModule

sys.modules[__name__] = _LazyModule(
    __name__, __file__, {'launcher.server_launcher': ['ServerLauncher', 'launch_server']}, module_spec=__spec__)
sys.modules[__name__].__all__ = ['ServerLauncher', 'launch_server']
