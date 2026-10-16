# Twinkle Server Launcher - swift model family (gemma-4-12B-it) on transformers.
#
# Starts the Twinkle server from server_config.yaml in this directory. The model
# app there names a swift `model_loader` factory, so ms-swift must be importable
# in this process's env (its Ray workers inherit it) before launching.
# Run this BEFORE the client script (self_cognition.py).

import os

os.environ['TWINKLE_TRUST_REMOTE_CODE'] = '1'

from twinkle.server import launch_server

# Resolve the path to server_config.yaml relative to this script's location.
file_dir = os.path.abspath(os.path.dirname(__file__))
config_path = os.path.join(file_dir, 'server_config.yaml')

# Launch the Twinkle server -- this call blocks until the server is shut down.
launch_server(config_path=config_path)
