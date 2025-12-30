import threading
from typing import Dict, Any, Union, Type, List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from peft import LoraConfig
from ray import serve

import twinkle
from adapter.twinkle.validation import is_token_valid
from twinkle import DeviceGroup, DeviceMesh
from twinkle.data_format import InputFeature, Trajectory
from twinkle.loss import Loss
from twinkle.model import TransformersModel
from twinkle.model.base import TwinkleModel


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
        authorization = request.headers.get("Authorization")
        token = authorization[7:] if authorization and authorization.startswith("Bearer ") else authorization
        if not is_token_valid(token):
            return JSONResponse(status_code=403, content={"detail": "Invalid token"})

        request_id = request.headers.get("X-Ray-Serve-Request-Id")
        if not request_id:
            return JSONResponse(
                status_code=400,
                content={"detail": "Missing X-Ray-Serve-Request-Id header, required for sticky session"}
            )
        request.state.request_id = request_id
        request.state.token = token
        response = await call_next(request)
        return response

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
            return self.model.__class__.__name__

        def assert_adapter_exists(self, adapter_name):
            assert adapter_name and adapter_name in self.adapter_records

        def assert_adapter_valid(self, adapter_name):
            assert not adapter_name or adapter_name in self.adapter_records

        @app.post("/forward")
        def forward(self, *, inputs: Union[InputFeature, List[InputFeature], List[Trajectory]], **kwargs):
            self.assert_adapter_exists(adapter_name=kwargs.get("adapter_name"))
            return self.model.forward(inputs=inputs, **kwargs)

        @app.post("/forward_only")
        def forward_only(self, *, inputs: Union[InputFeature, List[InputFeature], List[Trajectory]], **kwargs):
            self.assert_adapter_valid(adapter_name=kwargs.get("adapter_name"))
            return self.model.forward_only(inputs=inputs, **kwargs)

        @app.post("/calculate_loss")
        def calculate_loss(self, **kwargs):
            self.assert_adapter_exists(adapter_name=kwargs.get("adapter_name"))
            return self.model.calculate_loss(**kwargs)

        @app.post("/backward")
        def backward(self, **kwargs):
            self.assert_adapter_exists(adapter_name=kwargs.get("adapter_name"))
            return self.model.backward(**kwargs)

        @app.post("/forward_backward")
        def forward_backward(self, *, inputs: Union[InputFeature, List[InputFeature], List[Trajectory]], **kwargs):
            self.assert_adapter_exists(adapter_name=kwargs.get("adapter_name"))
            return self.model.forward_backward(inputs=inputs, **kwargs)

        @app.post("/step")
        def step(self, **kwargs):
            self.assert_adapter_exists(adapter_name=kwargs.get("adapter_name"))
            return self.model.step(**kwargs)

        @app.post("/zero_grad")
        def zero_grad(self, **kwargs):
            self.assert_adapter_exists(adapter_name=kwargs.get("adapter_name"))
            return self.model.zero_grad(**kwargs)

        @app.post("/lr_step")
        def lr_step(self, **kwargs):
            self.assert_adapter_exists(adapter_name=kwargs.get("adapter_name"))
            return self.model.lr_step(**kwargs)

        @app.post("/set_loss")
        def set_loss(self, loss_cls: Union[Type[Loss], str], **kwargs):
            self.assert_adapter_exists(adapter_name=kwargs.get("adapter_name"))
            return self.model.set_loss(loss_cls, **kwargs)

        @app.post("/set_optimizer")
        def set_optimizer(self, optimizer_cls: str, **kwargs):
            self.assert_adapter_exists(adapter_name=kwargs.get("adapter_name"))
            return self.model.set_optimizer(optimizer_cls, **kwargs)

        @app.post("/set_lr_scheduler")
        def set_lr_scheduler(self, scheduler_cls: str, **kwargs):
            self.assert_adapter_exists(adapter_name=kwargs.get("adapter_name"))
            return self.model.set_lr_scheduler(scheduler_cls, **kwargs)

        @app.post("/save")
        def save(self, output_dir: str, **kwargs):
            self.assert_adapter_exists(adapter_name=kwargs.get("adapter_name"))
            return self.model.save(output_dir, **kwargs)

        @app.post("/add_adapter")
        def add_adapter_to_model(self, adapter_name: str, config: Dict[str, Any]):
            assert adapter_name, 'You need to specify `adapter_name`'
            config = LoraConfig(**config)
            self.adapter_records[adapter_name] = 0
            return self.model.add_adapter_to_model(adapter_name, config)

        @app.post("/set_template")
        def set_template(self, template_cls: str, **kwargs):
            self.assert_adapter_exists(adapter_name=kwargs.get("adapter_name"))
            return self.model.set_template(template_cls, **kwargs)

        @app.post("/set_processor")
        def set_processor(self, processor_cls: str, **kwargs):
            self.assert_adapter_exists(adapter_name=kwargs.get("adapter_name"))
            return self.model.set_processor(processor_cls, **kwargs)

        @app.post("/heartbeat")
        def heartbeat(self, adapter_name: str):
            self.assert_adapter_exists(adapter_name=kwargs.get("adapter_name"))
            self.adapter_records[adapter_name] = 0

    return ModelManagement.bind()