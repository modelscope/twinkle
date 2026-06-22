"""Boot the e2e variant of the transformer server (model + sampler + processor)."""
import os

os.environ['TWINKLE_TRUST_REMOTE_CODE'] = '0'

from twinkle.server import launch_server

file_dir = os.path.abspath(os.path.dirname(__file__))
config_path = os.path.join(file_dir, 'server_config_e2e.yaml')

launch_server(config_path=config_path)
