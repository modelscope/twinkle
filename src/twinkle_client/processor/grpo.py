from typing import Optional
from twinkle_client.http import TWINKLE_SERVER_URL
from twinkle_client.http import http_post, heartbeat_manager
from twinkle import DeviceMesh
from twinkle.data_format import InputFeature
from .base import InputProcessor

class GRPOLossProcessor(InputProcessor):
    """Client wrapper for GRPOLossProcessor that calls server HTTP endpoints."""

    def __init__(self, device_mesh: Optional[DeviceMesh] = None, ignore_index: int = -100, **kwargs):
        from twinkle_client.http import get_base_url
        self.server_url = get_base_url()

        response = http_post(
            url=f'{self.server_url}/processors/create',
            json_data={
                'processor_type': 'processor',
                'class_type': 'GRPOLossProcessor',
                **{'device_mesh': device_mesh, 'ignore_index': ignore_index}, **kwargs
            }
        )
        response.raise_for_status()
        self.processor_id = response.json()['processor_id']
        heartbeat_manager.register_processor(self.processor_id)

    def __del__(self):
        try:
            heartbeat_manager.unregister_processor(self.processor_id)
        except:
            pass

        
    def prepare_inputs(self, inputs: InputFeature):
        response = http_post(
            url=f'{self.server_url}/processors/call',
            json_data={
                'processor_id': self.processor_id,
                'function': 'prepare_inputs',
                **{'inputs': inputs},
            }
        )
        response.raise_for_status()
        return response.json()["result"]
    