import os
import uuid

TWINKLE_SERVER_URL = os.environ.get("TWINKLE_SERVER_URL")
TWINKLE_SERVER_TOKEN = os.environ.get("TWINKLE_SERVER_TOKEN", "tml-EMPTY_TOKEN") # Must start with tml-
TWINKLE_REQUEST_ID = str(uuid.uuid4().hex)
