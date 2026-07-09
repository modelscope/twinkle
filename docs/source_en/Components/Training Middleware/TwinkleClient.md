# TwinkleClient

`TwinkleClient` is the Python client for interacting with the Twinkle REST API. It manages sessions, training runs, and checkpoints.

## Initialization

```python
from twinkle_client.manager import TwinkleClient

client = TwinkleClient(
    base_url='http://localhost:8000',   # Or TWINKLE_SERVER_URL env var
    api_key='your-api-key',             # Or TWINKLE_SERVER_TOKEN env var
    route_prefix='/twinkle',            # API route prefix
    session_heartbeat_interval=10,      # Heartbeat interval in seconds
    session_metadata={'user': 'alice'}, # Optional session metadata
)
```

On init, the client:
1. Sets `base_url` and `api_key` into shared context (used by all client objects)
2. Creates a server-side session
3. Starts a background heartbeat thread to keep the session alive

## Health Check

```python
is_healthy = client.health_check()  # Returns True/False
capabilities = client.get_server_capabilities()  # Supported models
```

## Training Runs

```python
# List runs
runs = client.list_training_runs(limit=20, offset=0)

# List with pagination cursor
runs, cursor = client.list_training_runs_with_cursor(limit=20)

# Get specific run
run = client.get_training_run(run_id='run_abc123')

# Find by base model
qwen_runs = client.find_training_run_by_model('Qwen/Qwen3.5-4B')
```

## Checkpoints

```python
# List checkpoints for a run
checkpoints = client.list_checkpoints(run_id='run_abc123')

# Get checkpoint path
parsed = client.get_checkpoint_path(run_id, checkpoint_id)
# parsed.path         → filesystem path
# parsed.twinkle_path → twinkle:// URI

# Get latest checkpoint (useful for resume training)
latest_path = client.get_latest_checkpoint_path(run_id)

# Delete checkpoint
client.delete_checkpoint(run_id, checkpoint_id)
```

## Capacity & Weights Info

```python
# LoRA capacity
capacity = client.get_capacity_info()
# capacity.max_loras, capacity.used_loras, capacity.free_loras

# Weights metadata
info = client.get_weights_info('twinkle://run_id/weights/checkpoint')
# info.base_model, info.is_lora, info.lora_rank
```

## Cleanup

```python
client.close()  # Stops heartbeat thread (also registered via atexit)
```
