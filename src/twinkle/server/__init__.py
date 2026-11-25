# Copyright (c) ModelScope Contributors. All rights reserved.
import sys

# Pickle/unpickling recursion guards and the tinker-type pydantic compatibility must run at
# import time -- before any FastAPI body validation sees a request. Applying them inside the
# lazy module would defer them past the first request and reintroduce the failures.
try:
    from . import _compat_pickle  # noqa: F401  (applies both patches on import)
except Exception:
    pass

from twinkle._lazy_module import _LazyModule

sys.modules[__name__] = _LazyModule(
    __name__, __file__, {'launcher.server_launcher': ['ServerLauncher', 'launch_server']}, module_spec=__spec__)
sys.modules[__name__].__all__ = ['ServerLauncher', 'launch_server']
