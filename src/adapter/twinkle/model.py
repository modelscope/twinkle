import threading
from typing import Dict, Any, Union, Type, List, Optional

from fastapi import FastAPI, Request
from peft import LoraConfig
from ray import serve

import twinkle
from twinkle import DeviceGroup, DeviceMesh
from twinkle.data_format import InputFeature, Trajectory
from twinkle.loss import Loss
from twinkle.model import TransformersModel
from twinkle.model.base import TwinkleModel
from .validation import verify_request_token


def build_model_app(model_id: str,
                    device_group: Dict[str, Any],
                    device_mesh: Dict[str, Any],
                    **kwargs):
    app = FastAPI()
    device_group = DeviceGroup(**device_group)
    twinkle.initialize(mode='ray', groups=[device_group], lazy_collect=False)

    device_mesh = DeviceMesh(**device_mesh)

    @app.middleware("http")
    async def verify_token(request: Request, call_next):
        return await verify_request_token(request=request, call_next=call_next)

    @serve.deployment(name="ModelManagement")
    @serve.ingress(app)
    class ModelManagement(TwinkleModel):

        COUNT_DOWN = 60 * 30

        def __init__(self):
            self.model = TransformersModel(model_id=model_id, device_mesh=device_mesh,
                                           remote_group=device_group.name, **kwargs)
            self.adapter_records: Dict[str, int] = {}
            self.hb_thread = threading.Thread(target=self.countdown)
            self.hb_thread.start()

        def countdown(self):
            for key in self.adapter_records.copy():
                self.adapter_records[key] += 1
                if self.adapter_records[key] > self.COUNT_DOWN:
                    self.model.remove_adapter(key)
                    self.adapter_records.pop(key)

        @app.post("/create")
        def create(self, *args, **kwargs):
            return ''

        def assert_adapter_exists(self, adapter_name):
            assert adapter_name and adapter_name in self.adapter_records

        def assert_adapter_valid(self, adapter_name):
            assert adapter_name == '' or adapter_name in self.adapter_records

        @staticmethod
        def get_adapter_name(request, adapter_name):
            return request.state.request_id + '-' + adapter_name

        @app.post("/forward")
        def forward(self,
                    request,
                    *,
                    inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
                    adapter_name: str,
                    **kwargs):
            self.assert_adapter_exists(adapter_name=adapter_name)
            adapter_name = self.get_adapter_name(request, adapter_name=adapter_name)
            return self.model.forward(inputs=inputs, adapter_name=adapter_name, **kwargs)

        @app.post("/forward_only")
        def forward_only(self,
                         request,
                         *,
                         inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
                         adapter_name: Optional[str] = None,
                         **kwargs):
            self.assert_adapter_valid(adapter_name=adapter_name)
            adapter_name = self.get_adapter_name(request, adapter_name=adapter_name)
            return self.model.forward_only(inputs=inputs, adapter_name=adapter_name, **kwargs)

        @app.post("/calculate_loss")
        def calculate_loss(self, request, *, adapter_name: str, **kwargs):
            self.assert_adapter_exists(adapter_name=adapter_name)
            adapter_name = self.get_adapter_name(request, adapter_name=adapter_name)
            return self.model.calculate_loss(adapter_name=adapter_name, **kwargs)

        @app.post("/backward")
        def backward(self, request, *, adapter_name: str, **kwargs):
            self.assert_adapter_exists(adapter_name=adapter_name)
            adapter_name = self.get_adapter_name(request, adapter_name=adapter_name)
            return self.model.backward(adapter_name=adapter_name, **kwargs)

        @app.post("/forward_backward")
        def forward_backward(self,
                             request,
                             *,
                             inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
                             adapter_name: str,
                             **kwargs):
            self.assert_adapter_exists(adapter_name=adapter_name)
            adapter_name = self.get_adapter_name(request, adapter_name=adapter_name)
            return self.model.forward_backward(inputs=inputs, adapter_name=adapter_name, **kwargs)

        @app.post("/step")
        def step(self, request, *, adapter_name: str, **kwargs):
            self.assert_adapter_exists(adapter_name=adapter_name)
            adapter_name = self.get_adapter_name(request, adapter_name=adapter_name)
            return self.model.step(adapter_name=adapter_name, **kwargs)

        @app.post("/zero_grad")
        def zero_grad(self, request, *, adapter_name: str, **kwargs):
            self.assert_adapter_exists(adapter_name=adapter_name)
            adapter_name = self.get_adapter_name(request, adapter_name=adapter_name)
            return self.model.zero_grad(adapter_name=adapter_name, **kwargs)

        @app.post("/lr_step")
        def lr_step(self, request, *, adapter_name: str, **kwargs):
            self.assert_adapter_exists(adapter_name=adapter_name)
            adapter_name = self.get_adapter_name(request, adapter_name=adapter_name)
            return self.model.lr_step(adapter_name=adapter_name, **kwargs)

        @app.post("/set_loss")
        def set_loss(self, request, *, loss_cls: Union[Type[Loss], str], adapter_name: str, **kwargs):
            self.assert_adapter_exists(adapter_name=adapter_name)
            adapter_name = self.get_adapter_name(request, adapter_name=adapter_name)
            return self.model.set_loss(loss_cls, adapter_name=adapter_name, **kwargs)

        @app.post("/set_optimizer")
        def set_optimizer(self, request, *, optimizer_cls: str, adapter_name: str, **kwargs):
            self.assert_adapter_exists(adapter_name=adapter_name)
            adapter_name = self.get_adapter_name(request, adapter_name=adapter_name)
            return self.model.set_optimizer(optimizer_cls, adapter_name=adapter_name, **kwargs)

        @app.post("/set_lr_scheduler")
        def set_lr_scheduler(self, request, *, scheduler_cls: str, adapter_name: str, **kwargs):
            self.assert_adapter_exists(adapter_name=adapter_name)
            adapter_name = self.get_adapter_name(request, adapter_name=adapter_name)
            return self.model.set_lr_scheduler(scheduler_cls, adapter_name=adapter_name, **kwargs)

        @app.post("/save")
        def save(self, request, *, output_dir: str, adapter_name: str, **kwargs):
            self.assert_adapter_exists(adapter_name=adapter_name)
            adapter_name = self.get_adapter_name(request, adapter_name=adapter_name)
            return self.model.save(output_dir, adapter_name=adapter_name, **kwargs)

        @app.post("/add_adapter")
        def add_adapter_to_model(self, adapter_name: str, config: Dict[str, Any]):
            assert adapter_name, 'You need to specify a valid `adapter_name`'
            config = LoraConfig(**config)
            self.adapter_records[adapter_name] = 0
            return self.model.add_adapter_to_model(adapter_name, config)

        @app.post("/set_template")
        def set_template(self, request, *, template_cls: str, adapter_name: str, **kwargs):
            self.assert_adapter_exists(adapter_name=adapter_name)
            adapter_name = self.get_adapter_name(request, adapter_name=adapter_name)
            return self.model.set_template(template_cls, adapter_name=adapter_name, **kwargs)

        @app.post("/set_processor")
        def set_processor(self, request, *, processor_cls: str, adapter_name: str, **kwargs):
            self.assert_adapter_exists(adapter_name=adapter_name)
            adapter_name = self.get_adapter_name(request, adapter_name=adapter_name)
            return self.model.set_processor(processor_cls, adapter_name=adapter_name, **kwargs)

        @app.post("/heartbeat")
        def heartbeat(self, request, *, adapter_name: str):
            self.assert_adapter_exists(adapter_name=adapter_name)
            adapter_name = self.get_adapter_name(request, adapter_name=adapter_name)
            self.adapter_records[adapter_name] = 0

    return ModelManagement.bind()