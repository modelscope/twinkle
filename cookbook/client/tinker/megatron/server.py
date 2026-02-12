# Twinkle Server Launcher - Tinker-Compatible Megatron Backend
#
# This script starts the Twinkle server with Tinker-compatible API support
# using the Megatron model backend.
# It reads the server_config.yaml in the same directory for all
# configuration (model, deployment settings, etc.).
# Run this script BEFORE running the client training script (lora.py).

import os

# Enable Ray debug mode for verbose logging during development
os.environ['TWINKLE_TRUST_REMOTE_CODE'] = '1'

from twinkle.server import launch_server

# Resolve the path to server_config.yaml relative to this script's location
file_dir = os.path.abspath(os.path.dirname(__file__))
config_path = os.path.join(file_dir, 'server_config.yaml')

# Launch the Twinkle server — this call blocks until the server is shut down
launch_server(config_path=config_path)