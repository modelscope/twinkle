# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
from typing import TYPE_CHECKING

from twinkle.utils.import_utils import _LazyModule

_import_structure = {
    'model': ['build_model_app'],
    'sampler': ['build_sampler_app'],
    'server': ['build_server_app'],
}

if TYPE_CHECKING:
    from .model import build_model_app
    from .sampler import build_sampler_app
    from .server import build_server_app
else:
    sys.modules[__name__] = _LazyModule(__name__, __file__, _import_structure, module_spec=__spec__)
