# Copyright (c) ModelScope Contributors. All rights reserved.
"""DeepSpeed ZeRO-3 compatibility patches.

Four patches that make PEFT and multimodal hooks work correctly under
DeepSpeed ZeRO-3 parameter partitioning. All inherit from twinkle's
``Patch`` base class and are applied via ``apply_patch`` / ``apply_context``.

1. **DeepSpeedLeafModulesPatch** — marks MoE blocks as ZeRO-3 leaf modules
   so that expert parameters are partitioned as a unit, not individually.

2. **DeepSpeedParamWrapperPatch** — patches ``ParamWrapper.get_param`` so
   that NOT_AVAILABLE ZeRO-3 params return a stride-0 placeholder with
   correct metadata (shape/ndim/dtype) using O(1) memory instead of
   gathering the full parameter.

3. **DeepSpeedModulesToSavePatch** — patches ``ModulesToSaveWrapper.__setattr__``
   so that ``ds_grads_remaining`` is propagated to all ``modules_to_save``
   sub-modules, keeping ZeRO-3 gradient tracking consistent.

4. **DeepSpeedHookReorderPatch** — wraps ``deepspeed.initialize`` so that
   application-level forward pre-hooks (e.g. multimodal input preprocessing)
   are moved to the end of ``_forward_pre_hooks`` after DeepSpeed adds its
   own hooks, ensuring they execute last.
"""
from functools import wraps
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type

from twinkle.patch import Patch

if TYPE_CHECKING:
    import torch.nn as nn

# model_type -> (import_path, class_name) for MoE leaf module resolution.
_MOE_LEAF_MAP: Dict[str, Tuple[str, str]] = {
    'qwen3_vl_moe': ('transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe', 'Qwen3VLMoeTextSparseMoeBlock'),
    'qwen3_omni_moe':
    ('transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe', 'Qwen3OmniMoeThinkerTextSparseMoeBlock'),
    'qwen2_moe': ('transformers.models.qwen2_moe.modeling_qwen2_moe', 'Qwen2MoeSparseMoeBlock'),
    'qwen3_moe': ('transformers.models.qwen3_moe.modeling_qwen3_moe', 'Qwen3MoeSparseMoeBlock'),
    'gemma4': ('transformers.models.gemma4.modeling_gemma4', 'Gemma4TextExperts'),
    'glm4_moe': ('transformers.models.glm4_moe.modeling_glm4_moe', 'Glm4MoeMoE'),
    'glm4_moe_lite': ('transformers.models.glm4_moe_lite.modeling_glm4_moe_lite', 'Glm4MoeLiteMoE'),
    'glm4v_moe': ('transformers.models.glm4v_moe.modeling_glm4v_moe', 'Glm4vMoeTextMoE'),
    'gpt_oss': ('transformers.models.gpt_oss.modeling_gpt_oss', 'GptOssMLP'),
    'llama4': ('transformers.models.llama4.modeling_llama4', 'Llama4TextMoe'),
    'qwen3_next': ('transformers.models.qwen3_next.modeling_qwen3_next', 'Qwen3NextSparseMoeBlock'),
    'olmoe': ('transformers.models.olmoe.modeling_olmoe', 'OlomoeSparseMoeBlock'),
    'qwen3_5_moe': ('transformers.models.qwen3_5_moe.modeling_qwen3_5_moe', 'Qwen3_5MoeSparseMoeBlock'),
    'glm_moe_dsa': ('transformers.models.glm_moe_dsa.modeling_glm_moe_dsa', 'GlmMoeDsaMoE'),
}


class DeepSpeedLeafModulesPatch(Patch):
    """Mark MoE blocks as ZeRO-3 leaf modules.

    DeepSpeed ZeRO-3 partitions parameters at the leaf-module level. For MoE
    models, the sparse MoE block must be a leaf module so that expert
    parameters are partitioned as a group, not individually (which would
    break expert routing).

    Resolves the MoE block class by ``model.config.model_type`` via a static
    map. For trust_remote_code models not in the map, scans ``model.modules()``
    for classes whose name ends with ``MoE`` or ``SparseMoeBlock``.

    Permanent patch: ``unpatch`` is a no-op.
    """

    def __call__(self, module: 'nn.Module', *args, **kwargs):
        try:
            model_type = module.config.model_type
        except Exception:
            return module

        leaf_modules = self._resolve_leaf_modules(module, model_type)
        if leaf_modules:
            from deepspeed.utils import set_z3_leaf_modules
            set_z3_leaf_modules(module, leaf_modules)
        return module

    @staticmethod
    def _resolve_leaf_modules(
        model: 'nn.Module',
        model_type: str,
    ) -> Optional[List[Type]]:
        entry = _MOE_LEAF_MAP.get(model_type)
        if entry is not None:
            import importlib
            module_path, class_name = entry
            try:
                mod = importlib.import_module(module_path)
                return [getattr(mod, class_name)]
            except (ImportError, AttributeError):
                pass

        # trust_remote_code fallback: scan for MoE block by class name
        for sub in model.modules():
            cn = type(sub).__name__
            if cn.endswith('MoE') or cn.endswith('SparseMoeBlock'):
                return [type(sub)]
        return None

    def unpatch(self, module: 'nn.Module', *args, **kwargs):
        pass


class DeepSpeedParamWrapperPatch(Patch):
    """Patch ``ParamWrapper.get_param`` for ZeRO-3 compatibility.

    When a parameter is ``NOT_AVAILABLE`` in ZeRO-3, ``param.data`` is a
    placeholder with wrong shape/ndim. Callers of ``get_param()`` only need
    metadata (shape, ndim, dtype, device, requires_grad), so we use
    ``ds_shape`` + ``expand`` to create a stride-0 tensor with correct
    metadata using O(1) memory instead of gathering the full parameter.

    Temporary patch: ``unpatch`` restores the original method.
    """

    def __call__(self, module: 'nn.Module', *args, **kwargs):
        try:
            from peft.tuners.lora.layer import ParamWrapper
        except ImportError:
            return module

        self._origin = ParamWrapper.get_param
        origin = self._origin

        def _get_param_patched(wrapper_self):
            param = origin(wrapper_self)
            if hasattr(param, 'ds_id'):
                from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
                if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                    ds_shape = param.ds_shape
                    ones = tuple(1 for _ in ds_shape)
                    import torch
                    fake = torch.empty(ones, dtype=param.dtype, device=param.device)
                    if param.requires_grad and param.dtype.is_floating_point:
                        fake.requires_grad_(True)
                    return fake.expand(ds_shape)
            return param

        ParamWrapper.get_param = _get_param_patched
        return module

    def unpatch(self, module: 'nn.Module', *args, **kwargs):
        try:
            from peft.tuners.lora.layer import ParamWrapper
            ParamWrapper.get_param = self._origin
        except (ImportError, AttributeError):
            pass


class DeepSpeedModulesToSavePatch(Patch):
    """Patch ``ModulesToSaveWrapper.__setattr__`` for ZeRO-3 compatibility.

    Propagates ``ds_grads_remaining`` from the wrapper to all
    ``modules_to_save`` sub-modules, so ZeRO-3 gradient tracking stays
    consistent when modules are saved outside the main model.

    Permanent patch: ``_patched`` class flag prevents double-application.
    ``unpatch`` is a no-op.
    """

    def __call__(self, module: 'nn.Module', *args, **kwargs):
        from peft.utils import ModulesToSaveWrapper

        if getattr(ModulesToSaveWrapper, '_patched', False):
            return module

        ModulesToSaveWrapper._patched = True
        old_setattr = ModulesToSaveWrapper.__setattr__
        self._old_setattr = old_setattr

        def _patched_setattr(wrapper_self, name, value):
            old_setattr(wrapper_self, name, value)
            if name == 'ds_grads_remaining':
                for sub in wrapper_self.modules_to_save.values():
                    sub.ds_grads_remaining = value

        ModulesToSaveWrapper.__setattr__ = _patched_setattr
        return module

    def unpatch(self, module: 'nn.Module', *args, **kwargs):
        pass


class DeepSpeedHookReorderPatch(Patch):
    """Reorder forward pre-hooks after ``deepspeed.initialize``.

    ``deepspeed.initialize`` adds its own forward pre-hooks to the model.
    Application-level hooks (e.g. multimodal input preprocessing) registered
    before initialization must run *after* DeepSpeed's hooks, so we record
    pre-existing hook IDs, call the original ``deepspeed.initialize``, then
    move those hooks to the end of ``_forward_pre_hooks``.

    Temporary patch: ``unpatch`` restores ``deepspeed.initialize``.
    """

    def __call__(self, module: 'nn.Module', *args, **kwargs):
        import deepspeed

        self._model = module
        self._origin_init = deepspeed.initialize

        model = module
        origin_init = self._origin_init

        @wraps(origin_init)
        def _initialize(*args, **kwargs):
            pre_hook_ids = list(model._forward_pre_hooks.keys())
            res = origin_init(*args, **kwargs)
            for hook_id in pre_hook_ids:
                model._forward_pre_hooks.move_to_end(hook_id)
            return res

        deepspeed.initialize = _initialize
        return module

    def unpatch(self, module: 'nn.Module', *args, **kwargs):
        import deepspeed
        if hasattr(self, '_origin_init'):
            deepspeed.initialize = self._origin_init
