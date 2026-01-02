import os
import threading
import uuid
from typing import Dict, Any

from fastapi import FastAPI
from fastapi import Request
from ray import serve

import twinkle
from adapter.twinkle.validation import verify_request_token, ConfigRegistry, init_config_registry
from twinkle import DeviceGroup, DeviceMesh


def build_processor_app(device_group: Dict[str, Any],
                      device_mesh: Dict[str, Any]):
    app = FastAPI()
    device_group = DeviceGroup(**device_group)
    twinkle.initialize(mode='ray',
                       groups=[device_group],
                       lazy_collect=False)

    device_mesh = DeviceMesh(**device_mesh)

    @app.middleware("http")
    async def verify_token(request: Request, call_next):
        return await verify_request_token(request=request, call_next=call_next)

    processors = ['dataset', 'gym', 'hub', 'preprocessor', 'processor',
                  'reward', 'template', 'weight_loader']

    @serve.deployment(name="ProcessorManagement")
    @serve.ingress(app)
    class ProcessorManagement:

        COUNT_DOWN = 60 * 30

        def __init__(self):
            self.resource_dict = {}
            self.resource_records: Dict[str, int] = {}
            self.hb_thread = threading.Thread(target=self.countdown)
            self.hb_thread.start()
            self.config_registry: ConfigRegistry = init_config_registry()
            self.per_token_processor_limit = os.environ.get("TWINKLE_PER_USER_PROCESSOR_LIMIT", 20)
            self.key_token_dict = {}

        def countdown(self):
            for key in self.resource_records.copy():
                self.resource_records[key] += 1
                if self.resource_records[key] > self.COUNT_DOWN:
                    self.resource_records.pop(key)
                    self.resource_dict.pop(key, None)
                    if key in self.key_token_dict:
                        self.handle_processor_count(self.key_token_dict[key], False)

        def assert_processor_exists(self, processor_id):
            assert processor_id and processor_id in self.resource_dict

        def handle_processor_count(self, token, add: bool):
            user_key = token + '_' + 'processor'
            cur_count = self.config_registry.get_config(user_key) or 0
            if add:
                if cur_count < self.per_token_processor_limit:
                    self.config_registry.add_config(user_key, cur_count + 1)
                else:
                    raise RuntimeError(f'Processor count limitation reached: {self.per_token_processor_limit}')
            else:
                if cur_count > 0:
                    cur_count -= 1
                    self.config_registry.add_config(user_key, cur_count)
                if cur_count <= 0:
                    self.config_registry.pop(user_key)

        @app.post("/create")
        def create(self, request, *, processor_type, class_type, **kwargs):
            assert processor_type in processors
            processor_type = getattr(twinkle, processor_type)
            assert hasattr(processor_type, class_type)
            self.handle_processor_count(request.state.token, True)
            processor_id = str(uuid.uuid4().hex)
            self.key_token_dict[processor_id] = request.state.token
            kwargs.pop('remote_group', None)
            kwargs.pop('device_mesh', None)
            processor = getattr(processor_type, class_type)(remote_group=device_group.name,
                                                            device_mesh=device_mesh,
                                                            **kwargs)
            self.resource_dict.update({processor_id: processor})
            return processor_id

        @app.post("/heartbeat")
        def heartbeat(self, processor_id: str):
            processor_id = processor_id.split(',')
            for _id in processor_id:
                if _id and _id in self.resource_dict:
                    self.resource_records[_id] = 0

        @app.post("/call")
        def call(self, processor_id: str, function: str, **kwargs):
            self.assert_processor_exists(processor_id=processor_id)
            processor = self.resource_dict.get(processor_id)
            function = getattr(processor, function, None)
            assert function is not None, f'`function` not found in {processor.__class__}'
            assert hasattr(function, '_execute'), f'Cannot call inner method of {processor.__class__}'
            return function(**kwargs)

    return ProcessorManagement.bind()
