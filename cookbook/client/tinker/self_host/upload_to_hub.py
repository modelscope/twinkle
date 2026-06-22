# Tinker-Compatible Client - Upload Checkpoint to Hub Example
#
# This script demonstrates how to upload a Tinker checkpoint to ModelScope Hub.
# Tinker checkpoints use the same twinkle:// path format as Twinkle checkpoints,
# so the upload is handled identically via the Twinkle upload interface.
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

# Checkpoint to upload: the twinkle:// path returned by training_client.save_state(),
# e.g. 'twinkle://20260301_142318-Qwen_Qwen3-4B-199d2cdb/weights/my-lora-epoch-0'
tinker_path = 'twinkle://REPLACE_ME/weights/REPLACE_ME'

# ModelScope destination (must belong to your account)
hub_model_id = 'your_username/your-model-name'
hub_token = None  # Set to your ModelScope API token, or None to use server default
# ── End of configuration ──────────────────────────────────────────────────────


def upload():
    # Step 1: Initialize the Twinkle client.
    # Tinker checkpoints (twinkle:// paths) are resolved by the same checkpoint
    # manager on the server, so init_twinkle_client is sufficient for upload.
    init_twinkle_client(base_url=base_url, api_key=api_key)

    # Step 2: Create the model client (no training state needed for upload)
    model = MultiLoraTransformersModel(model_id=f'ms://{base_model}')

    # Step 3: Upload checkpoint to ModelScope Hub.
    # The client polls for completion automatically; progress is printed to stdout.
    logger.info(f'Uploading {tinker_path!r} → {hub_model_id!r} ...')
    model.upload_to_hub(
        checkpoint_dir=tinker_path,
        hub_model_id=hub_model_id,
        hub_token=hub_token,
    )
    logger.info(f'Upload complete: https://modelscope.cn/models/{hub_model_id}')


if __name__ == '__main__':
    upload()
