from dataclasses import dataclass, field
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MethodType
from typing import Dict
from typing import Optional, Union, List

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer, Linear, Embedding

from twinkle import torch_util
from twinkle.data_format import InputFeature
from twinkle.patch.base import Patch


@dataclass
class LoraTenant:

    index: int
    adapter_name: str
    config: LoraConfig
    tenant_adapter_name: Optional[str] = None
    tenant_config: Optional[LoraConfig] = None
    lora_A_weights: Dict[str, torch.Tensor] = field(default_factory=lambda: {})


class MultiLora(Patch):

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

    def activate_adapter(self, tenant_adapter_name: str):
        if not self.has_lora(tenant_adapter_name):
            raise ValueError(f"Adapter {tenant_adapter_name} does not exist")
        adapter_name = self.find_lora_by_tenant(tenant_adapter_name).adapter_name
        self.module.set_adapter(adapter_name)

    def deactivate_adapter(self):
        self.module.disable_adapter_layers()

    @contextmanager
    def adapter(self, tenant_adapter_name: str):
        self.activate_adapter(tenant_adapter_name)
        yield self.find_lora_by_tenant(tenant_adapter_name).adapter_name
        self.deactivate_adapter()
    
    @contextmanager
    def save_context(self, tenant_adapter_name: str):
        adapter_name = self.find_lora_by_tenant(tenant_adapter_name).adapter_name
        peft_config = self.module.peft_config
        config_dict = {tenant_adapter_name: self.module.peft_config[adapter_name]}
        self.module.peft_config = config_dict
        active_adapter = self.module.active_adapter
        self.module.active_adapter = tenant_adapter_name
        yield
        self.module.peft_config = peft_config
        self.module.active_adapter = active_adapter
        self.deactivate_adapter()

    def check_length(self, inputs: InputFeature):
        total_length = sum(len(_input['input_ids']) for _input in inputs)
        if total_length > self.max_length:
             raise ValueError(f'Max length exceeds {self.max_length}')

    def acquire_lora(self, tenant_adapter_name: str, config: LoraConfig) -> str:
        if self.has_lora(tenant_adapter_name):
            raise ValueError(f'Lora {tenant_adapter_name} already exists')
        _available_lora = self._get_available_lora()
        if _available_lora is None:
            raise RuntimeError(f"No lora available for tenant {tenant_adapter_name}")
        if config.r > self.max_r:
            raise RuntimeError(f"Too big rank for lora: {config.r}")
        _available_lora.tenant_config = config
        _available_lora.tenant_adapter_name = tenant_adapter_name
        return _available_lora.adapter_name

    def release_lora(self, tenant_adapter_name: str) -> Optional[str]:
        _lora = self.find_lora_by_tenant(tenant_adapter_name)
        if _lora is not None:
            _lora.tenant_config = None
            _lora.tenant_adapter_name = None
            self._load_initial_weights(_lora.adapter_name)
        else:
            raise ValueError(f'No lora found for tenant {tenant_adapter_name}')

    def has_lora(self, adapter_name: str) -> bool:
        return len([_lora for _lora in self.loras if _lora.tenant_adapter_name == adapter_name]) > 0

    def find_lora_by_tenant(self, tenant_adapter_name):
        return [_lora for _lora in self.loras if _lora.tenant_adapter_name == tenant_adapter_name][0]

    def find_lora(self, adapter_name):
        return [_lora for _lora in self.loras if _lora.adapter_name == adapter_name][0]

    @staticmethod
    def match_target_modules(
            module_name: str,
            target_modules: Optional[Union[List[str], str]],
    ) -> bool:
        if target_modules is None:
            return False

        if isinstance(target_modules, list) and len(target_modules) == 0:
            return False

        if target_modules == "all-linear":
            return True

        if isinstance(target_modules, str):
            return re.fullmatch(target_modules, module_name) is not None

        if isinstance(target_modules, list):
            return any(module_name.endswith(t) for t in target_modules)

        return False

    def _patch_lora_forward(_self, name, base_layer: LoraLayer):

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
                    lora_A_out = torch.nn.functional.linear(dropout_x, lora_A.weight[:_lora.tenant_config.r, :], bias=None)
                    lora_B_out = torch.nn.functional.linear(lora_A_out, lora_B.weight[:, :_lora.tenant_config.r], bias=None)
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

    def patch(self, module: torch.nn.Module, *args, **kwargs):
        self.module = module
        for i in range(self.max_loras):
            config = LoraConfig(
                r=self.max_r,
                target_modules='all-linear',
                lora_alpha=32,
            )
            lora_tenant = LoraTenant(index=i, adapter_name=f'lora_{i}', config=config)
            self.loras.append(lora_tenant)
            if isinstance(module, PeftModel):
                module.add_adapter(lora_tenant.adapter_name, config)
            else:
                module = get_peft_model(module, config, lora_tenant.adapter_name)

        for name, submodule in module.named_modules():
            if isinstance(submodule, LoraLayer):
                self._patch_lora_forward(name, submodule)

        self.module = module
        return module

    def save_initial_weights(self):
        for i in range(self.max_loras):
            lora_tenant = self.loras[i]
            pattern = re.compile(rf'\.lora_(?:A|embedding_A)\.{re.escape(lora_tenant.adapter_name)}\.')
            for name, parameter in self.module.named_parameters():
                if pattern.search(name):
                    lora_tenant.lora_A_weights[name] = parameter.data.clone().to('cpu')

    def set_state_dict(self, tenant_adapter_name, state_dict):
        _lora = self.find_lora_by_tenant(tenant_adapter_name)
        pattern = re.compile(rf'\.lora_\w+\.{re.escape(_lora.adapter_name)}\.')
        for name, parameter in self.module.named_parameters():
            if pattern.search(name) and self.match_target_modules(name, _lora.tenant_config.target_modules):
                name = name.replace(f'.{_lora.adapter_name}.', f'.{_lora.tenant_adapter_name}.')
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

    def get_state_dict(self, tenant_adapter_name):
        state_dict = {}
        _lora = self.find_lora_by_tenant(tenant_adapter_name)
        pattern = re.compile(rf'\.lora_\w+\.{re.escape(_lora.adapter_name)}\.')
        for name, parameter in self.module.named_parameters():
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
                name = name.replace(f'.{_lora.adapter_name}.', f'.{_lora.tenant_adapter_name}.')
                state_dict[name] = _param
        return state_dict

    def _load_initial_weights(self, origin_adapter_name):
        _lora = self.find_lora(origin_adapter_name)
        pattern_A = re.compile(rf'\.lora_(?:A|embedding_A)\.{origin_adapter_name}\.')
        pattern_B = re.compile(rf'\.lora_(?:B|embedding_B)\.{origin_adapter_name}\.')
        for name, parameter in self.module.named_parameters():
            if pattern_A.search(name):
                parameter.data.copy_(_lora.lora_A_weights[name])
            if pattern_B.search(name):
                parameter.data.copy_(torch.zeros_like(parameter.data).to(parameter.data.dtype))

    def get_nb_trainable_parameters(self, tenant_adapter_name) -> tuple[int, int]:
        r"""
        Returns the number of trainable parameters and the number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        _lora = self.find_lora_by_tenant(tenant_adapter_name)
        adapter_name = _lora.adapter_name
        pattern = re.compile(rf'\.lora_\w+\.{re.escape(adapter_name)}\.')
        for name, param in self.module.named_parameters():
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
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            if param.__class__.__name__ == "Params4bit":
                if hasattr(param, "element_size"):
                    num_bytes = param.element_size()
                elif not hasattr(param, "quant_storage"):
                    num_bytes = 1
                else:
                    num_bytes = param.quant_storage.itemsize
                num_params = num_params * 2 * num_bytes

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    def get_trainable_parameters_example(self, tenant_adapter_name):
        trainable_param_names = []
        _lora = self.find_lora_by_tenant(tenant_adapter_name)
        adapter_name = _lora.adapter_name
        pattern = re.compile(rf'\.lora_\w+\.{re.escape(adapter_name)}\.')
        for name, parameter in self.module.named_parameters():
            if parameter.requires_grad and pattern.search(name) and self.match_target_modules(name, _lora.tenant_config.target_modules):
                name = name.replace(f'A.{adapter_name}', f'A.{tenant_adapter_name}')
                name = name.replace(f'B.{adapter_name}', f'B.{tenant_adapter_name}')
                trainable_param_names.append(name)
        trainable_param_names = trainable_param_names[:5] + ['...'] + trainable_param_names[-5:]
        trainable_param_names = '\n'.join(trainable_param_names)
        return trainable_param_names
