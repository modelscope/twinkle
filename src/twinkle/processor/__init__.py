from .base import InputProcessor
from typing import Union, Type, Optional

from twinkle import template, Plugin, DeviceMesh


class DataProcessorMixin:

    def __init__(self, device_mesh: Optional[DeviceMesh] = None):
        self.model_id: Optional[str] = None
        self.template: Optional[template.Template] = None
        self.processor: Optional[InputProcessor] = None
        self.device_mesh = device_mesh

    def set_template(self, template_cls: Union[Type[template.Template], str], model_id: str, **template_params):
        if isinstance(template_cls, str):
            if hasattr(template, template_cls):
                template_cls = getattr(template, template_cls)
            else:
                template_cls = Plugin.load_plugin(template_cls, template.Template)
        self.model_id = model_id
        self.template = template_cls(model_id, **template_params)

    def set_processor(self, processor_cls: Union[Type[InputProcessor], str], **processor_params):
        if isinstance(processor_cls, str):
            if hasattr(__file__.__module__, processor_cls):
                processor_cls = getattr(__file__.__module__, processor_cls)
            else:
                processor_cls = Plugin.load_plugin(processor_cls, InputProcessor)
        self.processor = processor_cls(device_mesh=self.device_mesh, **processor_params)