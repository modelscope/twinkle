import threading
from typing import Dict, Any, List

from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse
from peft import LoraConfig
from ray import serve

import twinkle
from adapter.twinkle.validation import is_token_valid
from twinkle import DeviceGroup, DeviceMesh
from twinkle.data_format import Trajectory
from twinkle.sampler import VLLMSampler, Sampler


def build_sampler_app(model_id: str,
                      device_group: Dict[str, Any],
                      device_mesh: Dict[str, Any],
                      **kwargs):
    app = FastAPI()
    device_group = DeviceGroup(**device_group)
    twinkle.initialize(mode='ray',
                       groups=[device_group],
                       lazy_collect=False)

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

    @serve.deployment(name="SamplerManagement")
    @serve.ingress(app)
    class SamplerManagement(Sampler):

        COUNT_DOWN = 60 * 30

        def __init__(self):
            self.sampler = VLLMSampler(model_id=model_id,
                                       device_mesh=device_mesh,
                                       remote_group=device_group.name,
                                       **kwargs)
            self.adapter_records: Dict[str, int] = {}
            self.hb_thread = threading.Thread(target=self.countdown)
            self.hb_thread.start()

        def countdown(self):
            for key in self.adapter_records.copy():
                self.adapter_records[key] += 1
                if self.adapter_records[key] > self.COUNT_DOWN:
                    self.sampler.remove_adapter(key)
                    self.adapter_records.pop(key)

        def assert_adapter_exists(self, adapter_name):
            assert adapter_name and adapter_name in self.adapter_records

        def assert_adapter_valid(self, adapter_name):
            assert not adapter_name or adapter_name in self.adapter_records

        @app.post("/create")
        def create(self, *args, **kwargs):
            return self.sampler.__class__.__name__

        @app.post("/sample")
        def sample(self, trajectories: List[Trajectory], adapter_name = '')-> List[Trajectory]:
            self.assert_adapter_valid(adapter_name)
            return self.sampler.sample(trajectories, adapter_name)

        @app.post("/add_adapter_to_sampler")
        def add_adapter_to_sampler(self, adapter_name: str, config):
            assert adapter_name, 'You need to specify `adapter_name`'
            config = LoraConfig(**config)
            return self.sampler.add_adapter_to_sampler(adapter_name, config)

        @app.post("/sync_weights")
        def sync_weights(self, state_dict: Dict[str, Any], adapter_name=''):
            return self.sampler.sync_weights(state_dict, adapter_name)

        @app.post("/heartbeat")
        def heartbeat(self, adapter_name: str):
            self.assert_adapter_exists(adapter_name=adapter_name)
            self.adapter_records[adapter_name] = 0

    return SamplerManagement.bind()
