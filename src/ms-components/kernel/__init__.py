from typing import Union, Callable, Any, List


def apply_kernel(module: Any,
                 kernel: Union[str, 'torch.nn.Module', Callable[[*Any], Any]],
                 target_modules: Union[str, List[str]]) -> None:

    if module.__class__