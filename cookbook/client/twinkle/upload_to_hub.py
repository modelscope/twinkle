# Twinkle Client - Upload Checkpoint to Hub Example
#
# This script demonstrates how to upload a saved checkpoint to ModelScope Hub
# using the Twinkle client.  No training is required: any existing checkpoint
# (obtained from a previous run via model.save()) can be uploaded directly.
#
# Prerequisites:
#   - A running Twinkle server with the `server` installation extra
#   - A ModelScope API token with write access to the target repository

import dotenv
import os

from twinkle import get_logger
from twinkle_client import init_twinkle_client

dotenv.load_dotenv('.env')
logger = get_logger()

# ── Configuration ─────────────────────────────────────────────────────────────
base_model = os.environ.get('TWINKLE_MODEL_ID', 'Qwen/Qwen3.5-4B')
base_url = os.environ.get('TWINKLE_SERVER_URL', 'http://localhost:8000')
api_key = os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY_TOKEN')

# Use the server checkpoint path returned by model.save().
# Example twinkle:// path (from model.save()):
#   'twinkle://20260410_131831-Qwen_Qwen3_5-4B-85279a20/weights/my-checkpoint'
twinkle_path = 'twinkle://REPLACE_ME/weights/REPLACE_ME'

# ModelScope destination (must belong to your account)
hub_model_id = 'your_username/your-model-name'
hub_token = os.environ.get('MODELSCOPE_API_TOKEN')
# ── End of configuration ──────────────────────────────────────────────────────


def upload():
    client = init_twinkle_client(base_url=base_url, api_key=api_key)
    try:
        model = client.model(f'ms://{base_model}')
        # The SDK logs transferred and committed bytes while waiting.
        result = model.upload_to_hub(twinkle_path, hub_model_id, hub_token)
        if result is not None:
            logger.info(f'Uploaded {result.repo_id} at revision {result.revision}')
    finally:
        client.close()


if __name__ == '__main__':
    upload()
