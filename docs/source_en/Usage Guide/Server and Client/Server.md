# Server

## Startup Methods

The Server is uniformly launched through the `launch_server` function or CLI command, with YAML configuration files.

### Method 1: Python Script Startup

```python
# server.py
import os
from twinkle.server import launch_server

# Get configuration file path (server_config.yaml in the same directory as the script)
file_dir = os.path.abspath(os.path.dirname(__file__))
config_path = os.path.join(file_dir, 'server_config.yaml')

# Launch service, this call will block until the service is shut down
launch_server(config_path=config_path)
```

### Method 2: Command Line Startup

```bash
# Start Twinkle native Server
python -m twinkle.server --config server_config.yaml

# Start Tinker compatible Server
python -m twinkle.server --config server_config.yaml --server-type tinker
```

CLI supported parameters:

| Parameter | Description | Default Value |
|------|------|-------|
| `-c, --config` | YAML configuration file path (required) | — |
| `-t, --server-type` | Server mode: `twinkle` or `tinker` | `twinkle` |
| `--namespace` | Ray namespace | tinker mode defaults to `twinkle_cluster` |
| `--no-wait` | Do not block and wait (daemon mode) | `False` |
| `--log-level` | Log level | `INFO` |

## YAML Configuration Details

The configuration file defines the complete deployment plan for the Server, including HTTP listening, application components, and resource allocation.

### Twinkle Server + Transformers Backend

```yaml
# server_config.yaml — Twinkle native protocol + Transformers backend

# Protocol type: twinkle native protocol
server_type: twinkle

# HTTP proxy location: EveryNode means running one proxy per Ray node (recommended for multi-node scenarios)
proxy_location: EveryNode

# HTTP listening configuration
http_options:
  host: 0.0.0.0        # Listen on all network interfaces
  port: 8000            # Service port number

# Application list: Each entry defines a service component deployed on the Server
applications:

  # 1. TwinkleServer: Central management service
  # Responsible for handling client connections, training run tracking, checkpoint management, etc.
  - name: server
    route_prefix: /server          # API path prefix
    import_path: server            # Built-in component identifier
    args:                          # No additional parameters
    deployments:
      - name: TwinkleServer
        autoscaling_config:
          min_replicas: 1                # Minimum number of replicas
          max_replicas: 1                # Maximum number of replicas
          target_ongoing_requests: 128   # Target concurrent requests per replica
        ray_actor_options:
          num_cpus: 0.1                  # CPU resources allocated to this Actor

  # 2. Model service: Hosts the base model
  # Executes forward propagation, backward propagation and other training computations
  - name: models-Qwen2.5-7B-Instruct
    route_prefix: /models/Qwen/Qwen2.5-7B-Instruct   # REST path for the model
    import_path: model
    args:
      use_megatron: false                              # Use Transformers backend
      model_id: "ms://Qwen/Qwen2.5-7B-Instruct"      # ModelScope model identifier
      adapter_config:                                  # LoRA adapter configuration
        per_token_adapter_limit: 30   # Maximum number of LoRAs that can be activated simultaneously
        adapter_timeout: 1800         # Idle adapter timeout unload time (seconds)
      nproc_per_node: 2               # Number of GPU processes per node
      device_group:                   # Logical device group
        name: model
        ranks: [0, 1]                 # GPU card numbers to use
        device_type: cuda
      device_mesh:                    # Distributed training mesh
        device_type: cuda
        mesh: [0, 1]                  # Device indices in the mesh
        mesh_dim_names: ['dp']        # Mesh dimensions: dp=data parallel
    deployments:
      - name: ModelManagement
        autoscaling_config:
          min_replicas: 1
          max_replicas: 1
          target_ongoing_requests: 16
        ray_actor_options:
          num_cpus: 0.1

  # 3. Processor service: Data preprocessing
  # Executes preprocessing tasks such as tokenization, template conversion, etc. on CPU
  - name: processor
    route_prefix: /processors
    import_path: processor
    args:
      nproc_per_node: 2               # Number of processor workers per node
      ncpu_proc_per_node: 2           # Number of CPU processes per node
      device_group:
        name: model
        ranks: 2
        device_type: CPU
      device_mesh:
        device_type: CPU
        mesh: [0, 1]
        mesh_dim_names: ['dp']
    deployments:
      - name: ProcessorManagement
        autoscaling_config:
          min_replicas: 1
          max_replicas: 1
          target_ongoing_requests: 128
        ray_actor_options:
          num_cpus: 0.1
```

### Twinkle Server + Megatron Backend

The difference from the Transformers backend is only in the `use_megatron` parameter of the Model service:

```yaml
  # Model service — Megatron backend
  - name: models-Qwen2.5-7B-Instruct
    route_prefix: /models/Qwen/Qwen2.5-7B-Instruct
    import_path: model
    args:
      use_megatron: true                               # Use Megatron-LM backend
      model_id: "ms://Qwen/Qwen2.5-7B-Instruct"
      nproc_per_node: 2
      device_group:
        name: model
        ranks: [0, 1]
        device_type: cuda
      device_mesh:
        device_type: cuda
        mesh: [0, 1]
        mesh_dim_names: ['dp']
```

> **Note**: The Megatron backend does not need `adapter_config` (LoRA adapter management is handled internally by Megatron).

### Tinker Compatible Server Configuration

Main differences in Tinker compatible mode:
- `server_type` set to `tinker`
- `route_prefix` uses `/api/v1` prefix (Tinker protocol specification)
- Can additionally configure Sampler service (for inference sampling)

```yaml
# server_config.yaml — Tinker compatible protocol

server_type: tinker

proxy_location: EveryNode

http_options:
  host: 0.0.0.0
  port: 8000

applications:

  # 1. TinkerCompatServer: Central API service
  - name: server
    route_prefix: /api/v1              # Tinker protocol API prefix
    import_path: server
    args:
    deployments:
      - name: TinkerCompatServer
        autoscaling_config:
          min_replicas: 1
          max_replicas: 1
          target_ongoing_requests: 128
        ray_actor_options:
          num_cpus: 0.1

  # 2. Model service (Megatron backend example)
  - name: models-Qwen2.5-0.5B-Instruct
    route_prefix: /api/v1/model/Qwen/Qwen2.5-0.5B-Instruct
    import_path: model
    args:
      use_megatron: true
      model_id: "ms://Qwen/Qwen2.5-0.5B-Instruct"
      nproc_per_node: 2
      device_group:
        name: model
        ranks: [0, 1]
        device_type: cuda
      device_mesh:
        device_type: cuda
        mesh: [0, 1]
        mesh_dim_names: ['dp']
    deployments:
      - name: ModelManagement
        autoscaling_config:
          min_replicas: 1
          max_replicas: 1
          target_ongoing_requests: 16
        ray_actor_options:
          num_cpus: 0.1

  # 3. Sampler service (optional, for inference sampling)
  - name: sampler-Qwen2.5-0.5B-Instruct
    route_prefix: /api/v1/sampler/Qwen/Qwen2.5-0.5B-Instruct
    import_path: sampler
    args:
      model_id: "ms://Qwen/Qwen2.5-0.5B-Instruct"
      nproc_per_node: 1
      sampler_type: vllm              # Inference engine: vllm (high performance) or torch
      engine_args:                    # vLLM engine parameters
        max_model_len: 4096           # Maximum sequence length
        gpu_memory_utilization: 0.5   # GPU memory usage ratio
        enable_lora: true             # Support loading LoRA during inference
      device_group:
        name: sampler
        ranks: [0]
        device_type: cuda
      device_mesh:
        device_type: cuda
        mesh: [0]
        mesh_dim_names: ['dp']
    deployments:
      - name: SamplerManagement
        autoscaling_config:
          min_replicas: 1
          max_replicas: 1
          target_ongoing_requests: 16
        ray_actor_options:
          num_cpus: 0.1
          num_gpus: 1                 # Sampler needs independent GPU
```

## Configuration Item Description

### Application Components (import_path)

| import_path | Twinkle Mode | Tinker Mode | Description |
|-------------|-------------|------------|------|
| `server` | ✅ | ✅ | Central management service, handles training runs and checkpoints |
| `model` | ✅ | ✅ | Model service, hosts base model for training |
| `processor` | ✅ | ❌ | Data preprocessing service (Twinkle mode only, Tinker mode needs local processing) |
| `sampler` | ✅ | ✅ | Inference sampling service |

### device_group and device_mesh

- **device_group**: Defines logical device groups, specifying which GPU cards to use
- **device_mesh**: Defines distributed training mesh, controls parallelization strategy

```yaml
device_group:
  name: model          # Device group name
  ranks: [0, 1]        # GPU card number list
  device_type: cuda     # Device type: cuda / CPU

device_mesh:
  device_type: cuda
  mesh: [0, 1]                    # Device indices in the mesh
  mesh_dim_names: ['dp']          # Dimension names, commonly used: dp (data parallel), tp (tensor parallel), pp (pipeline parallel)
```
