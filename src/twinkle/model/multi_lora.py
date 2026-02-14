import re
import torch
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import Embedding, Linear, LoraLayer
from types import MethodType
from typing import Any, Dict, List, Optional, Union

from twinkle import torch_util
from twinkle.data_format import InputFeature
from twinkle.utils import get_logger

logger = get_logger()


@dataclass
class LoraTenant:

    index: int
    adapter_name: str
    config: LoraConfig
    tenant_adapter_name: Optional[str] = None
    tenant_config: Optional[LoraConfig] = None
    lora_A_weights: Dict[str, torch.Tensor] = field(default_factory=lambda: {})


class MultiLora:

    def __init__(self, max_loras=5, max_r=32, max_length: int = 8192):
        self.max_loras = max_loras
        self.max_r = max_r
        self.loras: List[LoraTenant] = []
        self.module: PeftModel
        self._active_adapters = []
        self.max_length = max_length

    def _get_available_lora(self) -> Optional[LoraTenant]:
        for _lora in self.loras:
            if _lora.tenant_adapter_name is None:
                return _lora
        return None

    def _count_available_loras(self):
        return len([_lora for _lora in self.loras if _lora.tenant_adapter_name is None])

    def activate_adapter(self, tenant_adapter_name: str):
        if not self.has_lora(tenant_adapter_name):
            raise ValueError(f'Adapter {tenant_adapter_name} does not exist')
        adapter_name = self.find_lora_by_tenant(tenant_adapter_name).adapter_name
        if isinstance(self.module, list):
            for _module in self.module:
                # _module.enable_adapter_layers()
                if _module.active_adapter != adapter_name:
                    _module.set_adapter(adapter_name)
        else:
            # self.module.enable_adapter_layers()
            if self.module.active_adapter != adapter_name:
                self.module.set_adapter(adapter_name)

    def deactivate_adapter(self):
        if isinstance(self.module, list):
            for _module in self.module:
                _module.disable_adapter_layers()
        else:
            self.module.disable_adapter_layers()

    @contextmanager
    def adapter(self, tenant_adapter_name: str):
        self.activate_adapter(tenant_adapter_name)
        yield self.find_lora_by_tenant(tenant_adapter_name).adapter_name
        # self.deactivate_adapter()

    @contextmanager
    def save_context(self, tenant_adapter_name: str):
        _lora = self.find_lora_by_tenant(tenant_adapter_name)
        adapter_name = _lora.adapter_name

        def _before(_module):
            peft_config = _module.peft_config
            config_dict = {
                tenant_adapter_name if not isinstance(self.module, list) else adapter_name: _lora.tenant_config
            }
            _module.peft_config = config_dict
            _module._peft_config_origin = peft_config
            active_adapter = _module.active_adapter
            _module._active_adapter_origin = active_adapter
            _module.active_adapter = tenant_adapter_name

        def _after(_module):
            _module.peft_config = _module._peft_config_origin
            _module.active_adapter = _module._active_adapter_origin

        if isinstance(self.module, list):
            for _module in self.module:
                _before(_module)
        else:
            _before(self.module)
        yield adapter_name
        if isinstance(self.module, list):
            for _module in self.module:
                _after(_module)
        else:
            _after(self.module)
        # self.deactivate_adapter()

    def check_length(self, inputs: InputFeature):
        total_length = sum(len(_input['input_ids']) for _input in inputs)
        if total_length > self.max_length:
            raise ValueError(f'Max length exceeds {self.max_length}')

    def acquire_lora(self, tenant_adapter_name: str, config: LoraConfig) -> str:
        if self.has_lora(tenant_adapter_name):
            raise ValueError(f'Lora {tenant_adapter_name} already exists')
        _available_lora = self._get_available_lora()
        if _available_lora is None:
            raise RuntimeError(f'No lora available for tenant {tenant_adapter_name}')
        if config.r > self.max_r:
            raise RuntimeError(f'Too big rank for lora: {config.r}')
        _available_lora.tenant_config = config
        _available_lora.tenant_adapter_name = tenant_adapter_name
        logger.info(f'Lora count: {len(self.loras)}, available lora: {self._count_available_loras()}')
        return _available_lora.adapter_name

    def release_lora(self, tenant_adapter_name: str) -> Optional[str]:
        try:
            _lora = self.find_lora_by_tenant(tenant_adapter_name)
            _lora.tenant_config = None
            _lora.tenant_adapter_name = None
            self._load_initial_weights(_lora.adapter_name)
            logger.info(f'Lora count: {len(self.loras)}, available lora: {self._count_available_loras()}')
        except ValueError:
            return

    def has_lora(self, adapter_name: str) -> bool:
        return len([_lora for _lora in self.loras if _lora.tenant_adapter_name == adapter_name]) > 0

    def find_lora_by_tenant(self, tenant_adapter_name):
        _loras = [_lora for _lora in self.loras if _lora.tenant_adapter_name == tenant_adapter_name]
        if len(_loras) > 0:
            return _loras[0]
        else:
            raise ValueError(f'No lora found for tenant {tenant_adapter_name}')

    def find_lora(self, adapter_name):
        _loras = [_lora for _lora in self.loras if _lora.adapter_name == adapter_name]
        if len(_loras) > 0:
            return _loras[0]
        else:
            raise ValueError(f'No lora found for real adapter_name {adapter_name}')

    @staticmethod
    def match_target_modules(
        module_name: str,
        target_modules: Optional[Union[List[str], str]],
    ) -> bool:
        if target_modules is None:
            return False

        if isinstance(target_modules, list) and len(target_modules) == 0:
            return False

        if target_modules == 'all-linear':
            return True

        if isinstance(target_modules, str):
            return re.fullmatch(target_modules, module_name) is not None

        if isinstance(target_modules, list):
            return any(module_name.endswith(t) for t in target_modules)

        return False

    def _patch_lora_forward(_self, name, base_layer: LoraLayer):
        # Note: The Transformers backend also reaches this point to apply the LoRA forward patch.
        # Megatron is an optional dependency; if megatron-core/megatron is missing,
        # we must not crash the entire service just because we try to import megatron modules.
        try:
            from twinkle.model.megatron.tuners import LoraParallelLinear as _LoraParallelLinear
        except Exception:  # noqa: broad-except
            _LoraParallelLinear = ()

        if isinstance(base_layer, Linear):

            def _linear_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
                self._check_forward_args(x, *args, **kwargs)

                result = self.base_layer(x, *args, **kwargs)
                torch_result_dtype = result.dtype

                lora_A_keys = self.lora_A.keys()
                for active_adapter in self.active_adapters:
                    if active_adapter not in lora_A_keys:
                        continue
                    _lora = _self.find_lora(active_adapter)
                    target_modules = _lora.tenant_config.target_modules
                    if not _self.match_target_modules(self.layer_name, target_modules):
                        continue

                    lora_A = self.lora_A[active_adapter]
                    lora_B = self.lora_B[active_adapter]
                    dropout = self.lora_dropout[_lora.adapter_name]
                    scaling = _lora.tenant_config.lora_alpha / _lora.tenant_config.r
                    x = self._cast_input_dtype(x, lora_A.weight.dtype)
                    dropout_x = dropout(x)
                    lora_A_out = torch.nn.functional.linear(
                        dropout_x, lora_A.weight[:_lora.tenant_config.r, :], bias=None)
                    lora_B_out = torch.nn.functional.linear(
                        lora_A_out, lora_B.weight[:, :_lora.tenant_config.r], bias=None)
                    result = result + lora_B_out * scaling
                result = result.to(torch_result_dtype)
                return result

            base_layer.forward = MethodType(_linear_forward, base_layer)
            base_layer.layer_name = name
        elif isinstance(base_layer, Embedding):

            def _embedding_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
                self._check_forward_args(x, *args, **kwargs)

                result = self.base_layer(x, *args, **kwargs)
                torch_result_dtype = result.dtype

                lora_embedding_A_keys = self.lora_embedding_A.keys()
                for active_adapter in self.active_adapters:
                    if active_adapter not in lora_embedding_A_keys:
                        continue
                    _lora = self.find_lora(active_adapter)
                    target_modules = _lora.tenant_config.target_modules
                    if not self.match_target_modules(self.layer_name, target_modules):
                        continue

                    embedding_A = self.lora_embedding_A[active_adapter]
                    embedding_B = self.lora_embedding_B[active_adapter]
                    scaling = _lora.tenant_config.lora_alpha / _lora.tenant_config.r

                    embedding_A_T = embedding_A.T[:, :_lora.tenant_config.r]
                    embedding_B_T = embedding_B.T[:_lora.tenant_config.r, :]

                    after_A = self._embed(x, embedding_A_T.T)
                    lora_out = after_A @ embedding_B_T.T

                    result = result + lora_out * scaling

                result = result.to(torch_result_dtype)
                return result

            base_layer.forward = MethodType(_embedding_forward, base_layer)
            base_layer.layer_name = name

        elif isinstance(base_layer, _LoraParallelLinear):

            def _megatron_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
                from megatron.core.extensions.transformer_engine import (TEGroupedLinear,
                                                                         TELayerNormColumnParallelLinear, TELinear)
                from megatron.core.tensor_parallel import (gather_from_sequence_parallel_region,
                                                           scatter_to_sequence_parallel_region)
                from megatron.core.transformer.moe.router import TopKRouter

                previous_dtype = x.dtype
                if self.disable_adapters and self.merged:
                    self.unmerge()

                if isinstance(self.base_layer, TELayerNormColumnParallelLinear):
                    if self.disable_adapters or self.merged:
                        self.base_layer.return_layernorm_output = False
                        result, bias = self.base_layer(x, *args, **kwargs)
                    else:
                        self.base_layer.return_layernorm_output = True
                        if torch_util.is_torch_npu_available():
                            result, bias = self.base_layer(x, *args, **kwargs)
                        else:
                            (result, x), bias = self.base_layer(x, *args, **kwargs)
                elif isinstance(self.base_layer, (TELinear, TEGroupedLinear)):
                    result, bias = self.base_layer(x, *args, **kwargs)
                elif isinstance(self.base_layer, TopKRouter):
                    with self._patch_router_gating():
                        result, bias = self.base_layer(x, *args, **kwargs)
                else:
                    raise ValueError(f'Unsupported base layer type: {type(self.base_layer)}')

                if not isinstance(self.base_layer, TopKRouter) and not self.disable_adapters and not self.merged:
                    if self.sequence_parallel and self.base_layer.parallel_mode == 'column':
                        x = gather_from_sequence_parallel_region(x)

                    for active_adapter in self.active_adapters:
                        if active_adapter not in self.lora_A.keys():
                            continue

                        _lora = _self.find_lora(active_adapter)
                        target_modules = _lora.tenant_config.target_modules
                        if not _self.match_target_modules(self.layer_name, target_modules):
                            continue

                        lora_A = self.lora_A[active_adapter]
                        lora_B = self.lora_B[active_adapter]
                        dropout = self.lora_dropout[_lora.adapter_name]
                        scaling = _lora.tenant_config.lora_alpha / _lora.tenant_config.r

                        def _lora_A(x, *args, **kwargs):
                            if isinstance(lora_A, TEGroupedLinear):

                                def _get_weight_tensors(self):
                                    tensors = self._get_weight_tensors_origin()
                                    return [t[:_lora.tenant_config.r, :] for t in tensors]

                                lora_A._get_weight_tensors_origin = lora_A._get_weight_tensors
                                lora_A._get_weight_tensors = MethodType(_get_weight_tensors, lora_A)
                                output = lora_A(x, *args, **kwargs)
                                lora_A._get_weight_tensors = lora_A._get_weight_tensors_origin
                                delattr(lora_A, '_get_weight_tensors_origin')
                                return output
                            else:
                                return torch.nn.functional.linear(
                                    x, lora_A.weight[:_lora.tenant_config.r, :], bias=None)

                        def _lora_B(x, *args, **kwargs):
                            if isinstance(lora_B, TEGroupedLinear):

                                def _get_weight_tensors(self):
                                    tensors = self._get_weight_tensors_origin()
                                    return [t[:, :_lora.tenant_config.r] for t in tensors]

                                lora_B._get_weight_tensors_origin = lora_B._get_weight_tensors
                                lora_B._get_weight_tensors = MethodType(_get_weight_tensors, lora_B)
                                output = lora_B(x, *args, **kwargs)
                                lora_B._get_weight_tensors = lora_B._get_weight_tensors_origin
                                delattr(lora_B, '_get_weight_tensors_origin')
                                return output
                            else:
                                return torch.nn.functional.linear(
                                    x, lora_B.weight[:, :_lora.tenant_config.r], bias=None)

                        dtype = lora_A.weight0.dtype if isinstance(lora_A, TEGroupedLinear) else lora_A.weight.dtype
                        x = x.to(dtype)

                        lora_result = _lora_A(dropout(x), *args, **kwargs)
                        if isinstance(lora_result, tuple):
                            lora_result = lora_result[0]

                        lora_result = _lora_B(lora_result, *args, **kwargs)
                        if isinstance(lora_result, tuple):
                            lora_result = lora_result[0]

                        lora_result = lora_result * scaling

                        if self.sequence_parallel and self.base_layer.parallel_mode == 'row':
                            lora_result = scatter_to_sequence_parallel_region(lora_result)

                        result = result + lora_result

                result = result.to(previous_dtype)
                return result, bias

            base_layer.forward = MethodType(_megatron_forward, base_layer)
            base_layer.layer_name = name

    def patch(self, module: Union[torch.nn.Module, List[torch.nn.Module]], *args, **kwargs):
        for i in range(self.max_loras):
            config = LoraConfig(
                r=self.max_r,
                target_modules='all-linear',
                lora_alpha=32,
            )
            lora_tenant = LoraTenant(index=i, adapter_name=f'lora_{i}', config=config)
            self.loras.append(lora_tenant)

            def _patch_peft(_module):
                if isinstance(_module, PeftModel):
                    _module.add_adapter(lora_tenant.adapter_name, config)
                else:
                    _module = get_peft_model(_module, config, lora_tenant.adapter_name)

                for name, submodule in _module.named_modules():
                    if isinstance(submodule, LoraLayer):
                        self._patch_lora_forward(name, submodule)
                return _module

            def _patch_megatron(_module):
                # Mark expert layers for MoE models
                from .megatron.tuners.utils import set_linear_is_expert
                set_linear_is_expert(_module)

                # Expand target_modules (e.g., 'all-linear' -> actual module names)
                _config = deepcopy(config)

                from .megatron.tuners.utils import patch_deepcopy
                with patch_deepcopy():
                    if isinstance(_module, PeftModel):
                        _module.add_adapter(lora_tenant.adapter_name, _config)
                    else:
                        # TODO first wrap needs parse target_modules, need to fix later
                        if _config.target_modules:
                            if isinstance(_config.target_modules, str):
                                target_modules = [_config.target_modules]
                            else:
                                target_modules = list(_config.target_modules)

                            from .megatron.tuners.utils import get_target_modules
                            _config.target_modules = get_target_modules(_module, target_modules)
                        _module = get_peft_model(_module, _config, lora_tenant.adapter_name)

                    for name, submodule in _module.named_modules():
                        if isinstance(submodule, LoraLayer):
                            self._patch_lora_forward(name, submodule)
                return _module

            if isinstance(module, list):
                module = [_patch_megatron(_m) for _m in module]
            else:
                module = _patch_peft(module)

        self.module = module
        return module

    def save_initial_weights(self):
        for i in range(self.max_loras):
            lora_tenant = self.loras[i]
            pattern = re.compile(rf'\.lora_(?:A|embedding_A)\.{re.escape(lora_tenant.adapter_name)}\.')

            def _store_weights(_module):
                for name, parameter in _module.named_parameters():
                    if pattern.search(name):
                        lora_tenant.lora_A_weights[name] = parameter.data.clone().to('cpu')

            if isinstance(self.module, list):
                for _module in self.module:
                    _store_weights(_module)
            else:
                _store_weights(self.module)

    def load_lora_converter(self, name, parameter):

        def convert_param(name, parameter):
            if 'embedding_A' in name:
                r_saved = parameter.shape[1]
                parameter = torch.cat(
                    (parameter, torch.zeros(parameter.shape[0], self.max_r - r_saved).to(parameter.dtype)), dim=1)
            elif 'embedding_B' in name:
                r_saved = parameter.shape[0]
                parameter = torch.cat(
                    (parameter, torch.zeros(self.max_r - r_saved, parameter.shape[1]).to(parameter.dtype)), dim=0)
            elif '_A' in name:
                r_saved = parameter.shape[0]
                parameter = torch.cat(
                    (parameter, torch.zeros(self.max_r - r_saved, parameter.shape[1]).to(parameter.dtype)), dim=0)
            elif '_B' in name:
                r_saved = parameter.shape[1]
                parameter = torch.cat(
                    (parameter, torch.zeros(parameter.shape[0], self.max_r - r_saved).to(parameter.dtype)), dim=1)
            return name, parameter

        if isinstance(parameter, torch.Tensor):
            return convert_param(name, parameter)
        elif 'lazytensor' in parameter.__class__.__name__.lower():

            def _loader(self):
                tensor = self.loader_origin()
                return convert_param(name, tensor)[1]

            parameter.loader_origin = parameter.loader
            parameter.loader = MethodType(_loader, parameter)
            return name, parameter

    def save_lora_converter(self, name, parameter, adapter_name):
        _lora = self.find_lora(adapter_name)
        pattern = re.compile(rf'\.lora_\w+\.{adapter_name}\.')
        pattern_no_adapter = re.compile(r'\.lora_\w+\.weight')
        if (pattern.search(name) or pattern_no_adapter.search(name)) and self.match_target_modules(
                name, _lora.tenant_config.target_modules):
            _param = torch_util.to_local_tensor(parameter)
            if 'embedding_A' in name:
                _param = _param[:, :_lora.tenant_config.r]
            elif 'embedding_B' in name:
                _param = _param[:_lora.tenant_config.r, :]
            elif '_A' in name:
                _param = _param[:_lora.tenant_config.r, :]
            elif '_B' in name:
                _param = _param[:, :_lora.tenant_config.r]
            name = name.replace(f'.{_lora.adapter_name}.', '.')
            return name, _param
        else:
            return None, None

    def set_state_dict(self, tenant_adapter_name, state_dict):
        _lora = self.find_lora_by_tenant(tenant_adapter_name)
        pattern = re.compile(rf'\.lora_\w+\.{re.escape(_lora.adapter_name)}\.')

        def _load_weights(_module):
            for name, parameter in _module.named_parameters():
                if pattern.search(name) and self.match_target_modules(name, _lora.tenant_config.target_modules):
                    name = name.replace(f'.{_lora.adapter_name}.', '.')
                    src_tensor = state_dict[name]
                    if 'embedding_A' in name:
                        r_saved = src_tensor.shape[1]
                        parameter.data[:, :r_saved].copy_(src_tensor)
                    elif 'embedding_B' in name:
                        r_saved = src_tensor.shape[0]
                        parameter.data[:r_saved, :].copy_(src_tensor)
                    elif '_A' in name:
                        r_saved = src_tensor.shape[0]
                        parameter.data[:r_saved, :].copy_(src_tensor)
                    elif '_B' in name:
                        r_saved = src_tensor.shape[1]
                        parameter.data[:, :r_saved].copy_(src_tensor)

        if isinstance(self.module, list):
            for _module in self.module:
                _load_weights(_module)
        else:
            _load_weights(self.module)

    def get_state_dict(self, tenant_adapter_name):
        state_dict = {}
        _lora = self.find_lora_by_tenant(tenant_adapter_name)
        pattern = re.compile(rf'\.lora_\w+\.{re.escape(_lora.adapter_name)}\.')

        def _get_weights(_module):
            state_dict = {}
            for name, parameter in _module.named_parameters():
                if pattern.search(name) and self.match_target_modules(name, _lora.tenant_config.target_modules):
                    _param = torch_util.to_local_tensor(parameter)
                    if 'embedding_A' in name:
                        _param = _param[:, :_lora.tenant_config.r]
                    elif 'embedding_B' in name:
                        _param = _param[:_lora.tenant_config.r, :]
                    elif '_A' in name:
                        _param = _param[:_lora.tenant_config.r, :]
                    elif '_B' in name:
                        _param = _param[:, :_lora.tenant_config.r]
                    name = name.replace(f'.{_lora.adapter_name}.', '.')
                    state_dict[name] = _param
            return state_dict

        if isinstance(self.module, list):
            for _module in self.module:
                state_dict.update(_get_weights(_module))
        else:
            state_dict = _get_weights(self.module)
        return state_dict

    def _load_initial_weights(self, origin_adapter_name):
        _lora = self.find_lora(origin_adapter_name)
        pattern_A = re.compile(rf'\.lora_(?:A|embedding_A)\.{origin_adapter_name}\.')
        pattern_B = re.compile(rf'\.lora_(?:B|embedding_B)\.{origin_adapter_name}\.')

        def _load_initial_weights(_module):
            for name, parameter in _module.named_parameters():
                if pattern_A.search(name):
                    parameter.data.copy_(_lora.lora_A_weights[name])
                if pattern_B.search(name):
                    parameter.data.copy_(torch.zeros_like(parameter.data).to(parameter.data.dtype))

        if isinstance(self.module, list):
            for _module in self.module:
                _load_initial_weights(_module)
        else:
            _load_initial_weights(self.module)

    def get_nb_trainable_parameters(self, tenant_adapter_name) -> tuple[int, int]:
        r"""
        Returns the number of trainable parameters and the number of all parameters in the model.
        """
        _lora = self.find_lora_by_tenant(tenant_adapter_name)
        adapter_name = _lora.adapter_name
        pattern = re.compile(rf'\.lora_\w+\.{re.escape(adapter_name)}\.')

        def _count_trainable_parameters(_module):
            trainable_params = 0
            all_param = 0
            for name, param in _module.named_parameters():
                if not pattern.search(name) and 'lora_' in name:
                    # Other lora
                    continue
                if pattern.search(name) and not self.match_target_modules(name, _lora.tenant_config.target_modules):
                    # lora not match target_modules
                    continue

                if pattern.search(name):
                    if 'embedding_A' in name:
                        param = param[:, :_lora.tenant_config.r]
                    elif 'embedding_B' in name:
                        param = param[:_lora.tenant_config.r, :]
                    elif '_A' in name:
                        param = param[:_lora.tenant_config.r, :]
                    elif '_B' in name:
                        param = param[:, :_lora.tenant_config.r]

                num_params = param.numel()
                if num_params == 0 and hasattr(param, 'ds_numel'):
                    num_params = param.ds_numel

                if param.__class__.__name__ == 'Params4bit':
                    if hasattr(param, 'element_size'):
                        num_bytes = param.element_size()
                    elif not hasattr(param, 'quant_storage'):
                        num_bytes = 1
                    else:
                        num_bytes = param.quant_storage.itemsize
                    num_params = num_params * 2 * num_bytes

                all_param += num_params
                if param.requires_grad:
                    trainable_params += num_params
            return trainable_params, all_param

        trainable_params = 0
        all_param = 0
        if isinstance(self.module, list):
            for _module in self.module:
                _trainable, _all = _count_trainable_parameters(_module)
                trainable_params += _trainable
                all_param += _all
        else:
            trainable_params, all_param = _count_trainable_parameters(self.module)

        return trainable_params, all_param

    def get_trainable_parameters_example(self, tenant_adapter_name):
        trainable_param_names = []
        _lora = self.find_lora_by_tenant(tenant_adapter_name)
        adapter_name = _lora.adapter_name
        pattern = re.compile(rf'\.lora_\w+\.{re.escape(adapter_name)}\.')

        def _get_parameters(_module):
            for name, parameter in _module.named_parameters():
                if parameter.requires_grad and pattern.search(name) and self.match_target_modules(
                        name, _lora.tenant_config.target_modules):
                    name = name.replace(f'A.{adapter_name}', f'A.{tenant_adapter_name}')
                    name = name.replace(f'B.{adapter_name}', f'B.{tenant_adapter_name}')
                    trainable_param_names.append(name)

        if isinstance(self.module, list):
            for _module in self.module:
                _get_parameters(_module)
        else:
            _get_parameters(self.module)

        trainable_param_names = trainable_param_names[:5] + ['...'] + trainable_param_names[-5:]
        trainable_param_names = '\n'.join(trainable_param_names)
        return trainable_param_names
