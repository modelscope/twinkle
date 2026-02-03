from typing import List
from twinkle.patch import Patch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch.nn as nn


class MegatronPeft(Patch):
    _peft_patched = False

    def patch(self, *args, **kwargs):
        from peft.tuners.tuners_utils import BaseTuner

        if MegatronPeft._peft_patched:
            return

        _origin_get_tied_target_modules = BaseTuner._get_tied_target_modules

        def _get_tied_target_modules(self, model: 'nn.Module') -> List[str]:
            try:
                return _origin_get_tied_target_modules(self, model)
            except AttributeError:
                # Megatron's TransformerConfig doesn't have .get() method
                # Check share_embeddings_and_output_weights instead
                tied_target_modules = []
                if getattr(model, 'share_embeddings_and_output_weights',
                           False):
                    for target_module in self.targeted_module_names:
                        module_name = target_module.split('.')[-1]
                        if module_name in [
                            'output_layer', 'embedding', 'word_embeddings'
                        ]:
                            tied_target_modules.append(target_module)
                return tied_target_modules

        BaseTuner._get_tied_target_modules = _get_tied_target_modules
        MegatronPeft._peft_patched = True