# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import threading
from typing import Dict, Any, List

from fastapi import FastAPI
from fastapi import Request
from peft import LoraConfig
from ray import serve

import twinkle
from twinkle.server.twinkle.validation import verify_request_token, ConfigRegistryProxy, init_config_registry
from twinkle import DeviceGroup, DeviceMesh
from twinkle.data_format import Trajectory
from twinkle.sampler import VLLMSampler, Sampler


def build_sampler_app(model_id: str,
                      device_group: Dict[str, Any],
                      device_mesh: Dict[str, Any],
                      deploy_options: Dict[str, Any],
                      **kwargs):
    app = FastAPI()
    device_group = DeviceGroup(**device_group)
    twinkle.initialize(mode='ray',
                       groups=[device_group],
                       lazy_collect=False)

    device_mesh = DeviceMesh(**device_mesh)

    @app.middleware("http")
    async def verify_token(request: Request, call_next):
        return await verify_request_token(request=request, call_next=call_next)

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
            self.config_registry: ConfigRegistryProxy = init_config_registry()
            self.per_token_sampler_limit = os.environ.get("TWINKLE_PER_USER_SAMPLER_LIMIT", 3)
            self.key_token_dict = {}

        def countdown(self):
            for key in self.adapter_records.copy():
                self.adapter_records[key] += 1
                if self.adapter_records[key] > self.COUNT_DOWN:
                    self.sampler.remove_adapter(key)
                    self.adapter_records.pop(key)
                    if key in self.key_token_dict:
                        self.handle_adapter_count(self.key_token_dict[key], False)

        def handle_adapter_count(self, token, add: bool):
            user_key = token + '_' + 'sampler_adapter'
            cur_count = self.config_registry.get_config(user_key) or 0
            if add:
                if cur_count < self.per_token_sampler_limit:
                    self.config_registry.add_config(user_key, cur_count + 1)
                else:
                    raise RuntimeError(f'Model adapter count limitation reached: {self.per_token_sampler_limit}')
            else:
                if cur_count > 0:
                    cur_count -= 1
                    self.config_registry.add_config(user_key, cur_count)
                if cur_count <= 0:
                    self.config_registry.pop(user_key)

        def assert_adapter_exists(self, adapter_name):
            assert adapter_name and adapter_name in self.adapter_records

        def assert_adapter_valid(self, adapter_name):
            assert adapter_name == '' or adapter_name in self.adapter_records

        @staticmethod
        def get_adapter_name(request, adapter_name):
            return request.state.request_id + '-' + adapter_name

        @app.post("/create")
        def create(self, *args, **kwargs):
            return ''

        @app.post("/sample")
        def sample(self, request, *, trajectories: List[Trajectory], adapter_name: str = '')-> List[Trajectory]:
            self.assert_adapter_valid(adapter_name)
            adapter_name = self.get_adapter_name(request, adapter_name=adapter_name)
            return self.sampler.sample(trajectories, adapter_name)

        @app.post("/add_adapter_to_sampler")
        def add_adapter_to_sampler(self, request, *, adapter_name: str, config):
            assert adapter_name, 'You need to specify a valid `adapter_name`'
            self.handle_adapter_count(request.state.token, True)
            adapter_name = self.get_adapter_name(request, adapter_name=adapter_name)
            config = LoraConfig(**config)
            self.sampler.add_adapter_to_sampler(adapter_name, config)
            self.adapter_records[adapter_name] = 0

        @app.post("/sync_weights")
        def sync_weights(self, request, *, state_dict: Dict[str, Any], adapter_name: str):
            # TODO this design is illness
            self.assert_adapter_valid(adapter_name)
            adapter_name = self.get_adapter_name(request, adapter_name=adapter_name)
            return self.sampler.sync_weights(state_dict, adapter_name)

        @app.post("/heartbeat")
        def heartbeat(self, request, *, adapter_name: str):
            self.assert_adapter_exists(adapter_name=adapter_name)
            adapter_name = self.get_adapter_name(request, adapter_name=adapter_name)
            self.adapter_records[adapter_name] = 0

    return SamplerManagement.options(**deploy_options).bind()
