from typing import TYPE_CHECKING
from .utils.import_utils import _LazyModule # noqa

if TYPE_CHECKING:
    from .version import __version__, __release_datetime__
    from .utils import framework_util, torch_util, requires, exists
    from .infra import initialize, remote_class, remote_function
    from .dataset import Dataset

else:
    _import_structure = {
        'version': ['__release_datetime__', '__version__'],
        'utils': ['framework_util', 'torch_util', 'requires'],
        'infra': ['initialize', 'remote_class', 'remote_function'],
        'dataset': ['Dataset'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__, # noqa
        extra_objects={},
    )
