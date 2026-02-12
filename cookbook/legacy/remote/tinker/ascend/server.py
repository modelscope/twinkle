import os
os.environ['RAY_DEBUG'] = '1'

from twinkle.server import launch_server

file_dir = os.path.abspath(os.path.dirname(__file__))
config_path = os.path.join(file_dir, 'server_config.yaml')

launch_server(config_path=config_path)
