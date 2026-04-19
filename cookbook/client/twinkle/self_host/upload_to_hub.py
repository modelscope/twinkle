# Twinkle Client - Upload Checkpoint to Hub Example
#
# This script demonstrates how to upload a saved checkpoint to ModelScope Hub
# using the Twinkle client.  No training is required: any existing checkpoint
# (obtained from a previous run via model.save()) can be uploaded directly.
#
# How it works:
#   1. The server submits the upload as a background task and returns a
#      request_id immediately, so the HTTP call never times out.
#   2. The client polls /upload_status/{request_id} every few seconds and
#      blocks until the upload completes or raises on failure.
#
# Prerequisites:
#   - Server must be running (see server.py / server_config.yaml)
#   - A ModelScope API token with write access to the target repository

import dotenv

dotenv.load_dotenv('.env')

from twinkle import get_logger, init_twinkle_client
from twinkle_client.model import MultiLoraTransformersModel

logger = get_logger()

# ── Configuration ─────────────────────────────────────────────────────────────
base_model = 'Qwen/Qwen3.5-4B'
base_url = 'http://localhost:8000'
api_key = 'EMPTY_TOKEN'  # token used for model training / server access

# Checkpoint to upload: either a twinkle:// path or a local directory path.
# Example twinkle:// path (from model.save()):
#   'twinkle://20260410_131831-Qwen_Qwen3_5-4B-85279a20/weights/my-checkpoint'
twinkle_path = 'twinkle://REPLACE_ME/weights/REPLACE_ME'

# ModelScope destination (must belong to your account)
hub_model_id = 'your_username/your-model-name'
hub_token = None  # Set to your ModelScope API token, or None to use server default
# ── End of configuration ──────────────────────────────────────────────────────


def upload():
    # Step 1: Initialize the Twinkle client
    init_twinkle_client(base_url=base_url, api_key=api_key)

    # Step 2: Create the model client (no training state needed for upload)
    model = MultiLoraTransformersModel(model_id=f'ms://{base_model}')

    # Step 3: Upload checkpoint to ModelScope Hub.
    # The client polls for completion automatically; progress is printed to stdout.
    logger.info(f'Uploading {twinkle_path!r} → {hub_model_id!r} ...')
    model.upload_to_hub(
        checkpoint_dir=twinkle_path,
        hub_model_id=hub_model_id,
        hub_token=hub_token,
    )
    logger.info(f'Upload complete: https://modelscope.cn/models/{hub_model_id}')


if __name__ == '__main__':
    upload()
