# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Processor management application (moved from twinkle/processor.py).

Provides a Ray Serve deployment for managing distributed processors
(datasets, dataloaders, preprocessors, rewards, templates, weight loaders, etc.).
"""
import importlib
import os
import threading
import uuid
from fastapi import FastAPI, HTTPException, Request
from ray import serve
from typing import Any, Dict

import twinkle
import twinkle_client.types as types
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.server.common.serialize import deserialize_object
from twinkle.server.utils.state import ServerStateProxy, get_server_state
from twinkle.server.utils.validation import verify_request_token

logger = get_logger()


def build_processor_app(nproc_per_node: int, ncpu_proc_per_node: int, device_group: Dict[str, Any],
                        device_mesh: Dict[str, Any], deploy_options: Dict[str, Any], **kwargs):
    """Build the processor management application.

    Args:
        nproc_per_node: Number of GPU processes per node
        ncpu_proc_per_node: Number of CPU processes per node
        device_group: Device group configuration dict
        device_mesh: Device mesh configuration dict
        deploy_options: Ray Serve deployment options
        **kwargs: Additional arguments

    Returns:
        Ray Serve deployment bound with configuration
    """
    app = FastAPI()

    @app.middleware('http')
    async def verify_token(request: Request, call_next):
        return await verify_request_token(request=request, call_next=call_next)

    processors = ['dataset', 'dataloader', 'preprocessor', 'processor', 'reward', 'template', 'weight_loader']

    @serve.deployment(name='ProcessorManagement')
    @serve.ingress(app)
    class ProcessorManagement:
        """Processor management service.

        Manages lifecycle and invocation of distributed processor objects
        (datasets, dataloaders, rewards, templates, etc.).
        """

        COUNT_DOWN = 60 * 30

        def __init__(self, nproc_per_node: int, ncpu_proc_per_node: int, device_group: Dict[str, Any],
                     device_mesh: Dict[str, Any]):
            self.device_group = DeviceGroup(**device_group)
            twinkle.initialize(
                mode='ray',
                nproc_per_node=nproc_per_node,
                groups=[self.device_group],
                lazy_collect=False,
                ncpu_proc_per_node=ncpu_proc_per_node)
            if 'mesh_dim_names' in device_mesh:
                self.device_mesh = DeviceMesh(**device_mesh)
            else:
                self.device_mesh = DeviceMesh.from_sizes(**device_mesh)
            self.resource_dict = {}
            self.resource_records: Dict[str, int] = {}
            self.hb_thread = threading.Thread(target=self.countdown, daemon=True)
            self.hb_thread.start()
            self.state: ServerStateProxy = get_server_state()
            self.per_token_processor_limit = int(os.environ.get('TWINKLE_PER_USER_PROCESSOR_LIMIT', 20))
            self.key_token_dict = {}

        def countdown(self):
            import time
            while True:
                time.sleep(1)
                for key in list(self.resource_records.keys()):
                    self.resource_records[key] += 1
                    if self.resource_records[key] > self.COUNT_DOWN:
                        self.resource_records.pop(key, None)
                        self.resource_dict.pop(key, None)
                        if key in self.key_token_dict:
                            self.handle_processor_count(self.key_token_dict.pop(key), False)

        def assert_processor_exists(self, processor_id: str):
            assert processor_id and processor_id in self.resource_dict, f'Processor {processor_id} not found'

        def handle_processor_count(self, token: str, add: bool):
            user_key = token + '_' + 'processor'
            cur_count = self.state.get_config(user_key) or 0
            if add:
                if cur_count < self.per_token_processor_limit:
                    self.state.add_config(user_key, cur_count + 1)
                else:
                    raise RuntimeError(f'Processor count limitation reached: {self.per_token_processor_limit}')
            else:
                if cur_count > 0:
                    cur_count -= 1
                    self.state.add_config(user_key, cur_count)
                if cur_count <= 0:
                    self.state.pop_config(user_key)

        @app.post('/twinkle/create', response_model=types.ProcessorCreateResponse)
        def create(self, request: Request, body: types.ProcessorCreateRequest) -> types.ProcessorCreateResponse:
            processor_type_name = body.processor_type
            class_type = body.class_type
            _kwargs = body.model_extra or {}

            assert processor_type_name in processors, f'Invalid processor type: {processor_type_name}'
            processor_module = importlib.import_module(f'twinkle.{processor_type_name}')
            assert hasattr(processor_module, class_type), f'Class {class_type} not found in {processor_type_name}'
            self.handle_processor_count(request.state.token, True)
            processor_id = str(uuid.uuid4().hex)
            self.key_token_dict[processor_id] = request.state.token

            _kwargs.pop('remote_group', None)
            _kwargs.pop('device_mesh', None)

            resolved_kwargs = {}
            for key, value in _kwargs.items():
                if isinstance(value, str) and value.startswith('pid:'):
                    ref_id = value[4:]
                    resolved_kwargs[key] = self.resource_dict[ref_id]
                else:
                    value = deserialize_object(value)
                    resolved_kwargs[key] = value

            processor = getattr(processor_module, class_type)(
                remote_group=self.device_group.name,
                device_mesh=self.device_mesh,
                instance_id=processor_id,
                **resolved_kwargs)
            self.resource_dict[processor_id] = processor
            self.resource_records[processor_id] = 0
            return types.ProcessorCreateResponse(processor_id='pid:' + processor_id)

        @app.post('/twinkle/heartbeat', response_model=types.ProcessorHeartbeatResponse)
        def heartbeat(self, body: types.ProcessorHeartbeatRequest) -> types.ProcessorHeartbeatResponse:
            processor_ids = body.processor_id.split(',')
            for _id in processor_ids:
                if _id and _id in self.resource_dict:
                    self.resource_records[_id] = 0
            return types.ProcessorHeartbeatResponse()

        @app.post('/twinkle/call', response_model=types.ProcessorCallResponse)
        def call(self, body: types.ProcessorCallRequest) -> types.ProcessorCallResponse:
            processor_id = body.processor_id
            function_name = body.function
            _kwargs = body.model_extra or {}
            processor_id = processor_id[4:]
            self.assert_processor_exists(processor_id=processor_id)
            processor = self.resource_dict.get(processor_id)
            function = getattr(processor, function_name, None)

            assert function is not None, f'`{function_name}` not found in {processor.__class__}'
            assert hasattr(function, '_execute'), f'Cannot call inner method of {processor.__class__}'

            resolved_kwargs = {}
            for key, value in _kwargs.items():
                if isinstance(value, str) and value.startswith('pid:'):
                    ref_id = value[4:]
                    resolved_kwargs[key] = self.resource_dict[ref_id]
                else:
                    value = deserialize_object(value)
                    resolved_kwargs[key] = value

            # Special handling for __next__ to catch StopIteration
            if function_name == '__next__':
                try:
                    result = function(**resolved_kwargs)
                    return types.ProcessorCallResponse(result=result)
                except StopIteration:
                    # HTTP 410 Gone signals iterator exhausted
                    raise HTTPException(status_code=410, detail='Iterator exhausted')

            result = function(**resolved_kwargs)
            if function_name == '__iter__':
                return types.ProcessorCallResponse(result='ok')
            else:
                return types.ProcessorCallResponse(result=result)

    return ProcessorManagement.options(**deploy_options).bind(nproc_per_node, ncpu_proc_per_node, device_group,
                                                              device_mesh)
