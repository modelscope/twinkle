# Copyright (c) ModelScope Contributors. All rights reserved.
import sys

from twinkle._lazy_module import _LazyModule

sys.modules[__name__] = _LazyModule(
    __name__,
    __file__, {
        'hub': ['HFHub', 'HubOperation', 'MSHub'],
        'model_alias':
        ['MODEL_ID_ALIASES_ENV', 'build_model_alias_map', 'load_model_alias_map', 'resolve_model_id_alias'],
    },
    module_spec=__spec__)
