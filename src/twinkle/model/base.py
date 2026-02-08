# Copyright (c) ModelScope Contributors. All rights reserved.
from abc import abstractmethod, ABC
from typing import Dict, Any, Union, Type, Optional, Callable, TYPE_CHECKING

from twinkle.data_format import InputFeature, ModelOutput
from twinkle.loss.base import Loss
from twinkle.metric import Metric
from twinkle.patch import Patch
from twinkle.processor import InputProcessor
from twinkle.template import Template
from twinkle import torch_util, Platform
from twinkle.hub import HubOperation

if TYPE_CHECKING:
    import torch
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler


class TwinkleModel(ABC):

    _checkpoint_engine = None

    @abstractmethod
    def forward(self, *, inputs: Dict[str, Any], **kwargs):
        ...

    @abstractmethod
    def forward_only(self, *, inputs: Dict[str, Any], **kwargs):
        ...

    @abstractmethod
    def calculate_loss(self, **kwargs):
        ...

    @abstractmethod
    def backward(self, **kwargs):
        ...

    @abstractmethod
    def forward_backward(self, *, inputs: Dict[str, Any], **kwargs):
        ...

    @abstractmethod
    def clip_grad_norm(self, max_grad_norm: float = 1.0, norm_type=2, **kwargs):
        ...

    @abstractmethod
    def step(self, **kwargs):
        ...

    @abstractmethod
    def zero_grad(self, **kwargs):
        ...

    @abstractmethod
    def lr_step(self, **kwargs):
        ...

    @abstractmethod
    def clip_grad_and_step(self, max_grad_norm: float=1.0, norm_type=2, **kwargs):
        ...

    @abstractmethod
    def set_loss(self, loss_cls: Union[Loss, Type[Loss], str, Callable[[InputFeature, ModelOutput, ...], 'torch.Tensor']], **kwargs):
        ...

    @abstractmethod
    def set_optimizer(self, optimizer_cls: Union['Optimizer', Type['Optimizer'], str], **kwargs):
        ...

    @abstractmethod
    def set_lr_scheduler(self, scheduler_cls: Union['LRScheduler', Type['LRScheduler'], str], **kwargs):
        ...

    @abstractmethod
    def save(self, name: str, output_dir: Optional[str] = None, **kwargs):
        ...

    @abstractmethod
    def load(self, name: str, output_dir: Optional[str] = None, **kwargs):
        ...

    @abstractmethod
    def get_state_dict(self, **kwargs):
        ...

    @abstractmethod
    def apply_patch(self, patch_cls: Union[Patch, Type[Patch], str], **kwargs):
        ...

    @abstractmethod
    def add_metric(self, metric_cls: Union[Metric, str], is_training: Optional[bool] = None, **kwargs):
        ...

    @abstractmethod
    def calculate_metric(self, is_training: bool, **kwargs):
        ...

    @abstractmethod
    def add_adapter_to_model(self, adapter_name: str, config_or_dir, **kwargs):
        ...

    @abstractmethod
    def set_template(self, template_cls: Union[Template, Type[Template], str], **kwargs):
        ...

    @abstractmethod
    def set_processor(self, processor_cls: Union[InputProcessor, Type[InputProcessor], str], **kwargs):
        ...

    @abstractmethod
    def get_train_configs(self, **kwargs) -> str:
        ...

    def upload_to_hub(self, checkpoint_dir: str, hub_model_id: str, hub_token: Optional[str] = None, async_upload: bool = True):
        """Upload model checkpoint to hub.
        
        Args:
            checkpoint_dir: The directory path of the checkpoint to upload.
            hub_model_id: The hub model id.
            hub_token: The hub token (optional).
            async_upload: Whether to use async upload (default: True).
        """
        if async_upload:
            HubOperation.async_push_to_hub(
                repo_id=hub_model_id,
                folder_path=checkpoint_dir,
                token=hub_token,
                private=True
            )
        else:
            HubOperation.push_to_hub(
                repo_id=hub_model_id,
                folder_path=checkpoint_dir,
                token=hub_token,
                private=True
            )

    def _try_init_process_group(self):
        import torch
        import torch.distributed as dist
        if not dist.is_initialized() and Platform.get_world_size() > 1:
            torch_util.set_device()
            backend = Platform.device_backend()
            init_kwargs = {
                "backend": backend,
                "init_method": "env://",
                "rank": Platform.get_rank(),
                "world_size": Platform.get_world_size(),
            }
            if backend in ("nccl", "hccl"):
                init_kwargs["device_id"] = torch.device(Platform.get_local_device())
            dist.init_process_group(**init_kwargs)
