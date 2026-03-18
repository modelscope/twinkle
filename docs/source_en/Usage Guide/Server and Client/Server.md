# Server

## Ray Cluster Configuration

Before starting the Server, **you must first start and configure the Ray nodes**. Only after the Ray nodes are properly configured can the Server correctly allocate and occupy resources (GPU, CPU, etc.).

### Starting Ray Nodes

A Ray cluster consists of multiple nodes, each of which can be configured with different resources. The startup steps are as follows:

#### 1. Start the Head Node (First GPU Node)

```bash
# Stop existing Ray cluster (if any)
ray stop

# Start the Head node with GPU 0-3, 4 GPUs in total
CUDA_VISIBLE_DEVICES=0,1,2,3 ray start --head --num-gpus=4 --port=6379
```

#### 2. Start Worker Nodes

```bash
# Second GPU node, using GPU 4-7, 4 GPUs in total
CUDA_VISIBLE_DEVICES=4,5,6,7 ray start --address=10.28.252.9:6379 --num-gpus=4

# CPU node (for running Processor and other CPU tasks)
ray start --address=10.28.252.9:6379 --num-gpus=0
```

**Notes:**
- `--head`: Marks this node as the Head node (the primary node of the cluster)
- `--port=6379`: The port the Head node listens on
- `--address=<IP>:<PORT>`: The address for Worker nodes to connect to the Head node
- `--num-gpus=N`: The number of GPUs available on this node
- `CUDA_VISIBLE_DEVICES`: Restricts the GPU devices visible to this node

#### 3. Complete Example: 3-Node Cluster

```bash
# Stop the old cluster and start a new one
ray stop && \
CUDA_VISIBLE_DEVICES=0,1,2,3 ray start --head --num-gpus=4 --port=6379 && \
CUDA_VISIBLE_DEVICES=4,5,6,7 ray start --address=10.28.252.9:6379 --num-gpus=4 && \
ray start --address=10.28.252.9:6379 --num-gpus=0
```

This configuration starts 3 nodes:
- **Node 0** (Head): 4 GPUs (cards 0-3)
- **Node 1** (Worker): 4 GPUs (cards 4-7)
- **Node 2** (Worker): CPU-only node

#### 4. Set Environment Variables

Before starting the Server, you need to set the following environment variables:

```bash
export TWINKLE_TRUST_REMOTE_CODE=0       # Whether to trust remote code (security consideration)
```

### Node Rank in YAML Configuration

In the YAML configuration file, **each component needs to occupy a separate Node**.

**Example configuration:**

```yaml
applications:
  # Model service occupies GPU 0-3 (physical card numbers)
  - name: models-Qwen2.5-7B-Instruct
    route_prefix: /models/Qwen/Qwen2.5-7B-Instruct
    import_path: model
    args:
      nproc_per_node: 4
      device_group:
        name: model
        ranks: 4               # Number of GPUs to use
        device_type: cuda
      device_mesh:
        device_type: cuda
        dp_size: 4             # Data parallel size
        # tp_size: 1           # Tensor parallel size (optional)
        # pp_size: 1           # Pipeline parallel size (optional)
        # ep_size: 1           # Expert parallel size (optional)

  # Sampler service occupies GPU 4-5 (physical card numbers)
  - name: sampler-Qwen2.5-7B-Instruct
    route_prefix: /sampler/Qwen/Qwen2.5-7B-Instruct
    import_path: sampler
    args:
      nproc_per_node: 2
      device_group:
        name: sampler
        ranks: 2               # Number of GPUs to use
        device_type: cuda
      device_mesh:
        device_type: cuda
        dp_size: 2             # Data parallel size

  # Processor service occupies CPU
  - name: processor
    route_prefix: /processors
    import_path: processor
    args:
      ncpu_proc_per_node: 4
      device_group:
        name: processor
        ranks: 0               # CPU index
        device_type: CPU
      device_mesh:
        device_type: CPU
        dp_size: 4             # Data parallel size
```
**Important notes:**
- The `ranks` configuration specifies the **number of GPUs** to allocate for the component
- The `device_mesh` configuration uses parameters like `dp_size`, `tp_size`, `pp_size`, `ep_size` to define the parallelization strategy
- Different components will be automatically assigned to different Nodes
- Ray will automatically schedule to the appropriate Node based on resource requirements (`num_gpus`, `num_cpus` in `ray_actor_options`)

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
python -m twinkle.server --config server_config.yaml
```

CLI supported parameters:

| Parameter | Description | Default Value |
|------|------|-------|
| `-c, --config` | YAML configuration file path (required) | — |
| `--namespace` | Ray namespace | `twinkle_cluster` |
| `--log-level` | Log level | `INFO` |

## YAML Configuration Details

The configuration file defines the complete deployment plan for the Server, including HTTP listening, application components, and resource allocation. The Server simultaneously supports both Twinkle and Tinker clients through a unified configuration file.

### Complete Configuration Example (Megatron Backend)

```yaml
# HTTP proxy location: EveryNode means running one proxy per Ray node (recommended for multi-node scenarios)
proxy_location: EveryNode

# HTTP listening configuration
http_options:
  host: 0.0.0.0        # Listen on all network interfaces
  port: 8000            # Service port number

# Application list: Each entry defines a service component deployed on the Server
applications:

  # 1. TinkerCompatServer: Central API service
  # Handles client connections, training run tracking, checkpoint management, etc.
  # route_prefix uses /api/v1, compatible with both Tinker and Twinkle clients
  - name: server
    route_prefix: /api/v1
    import_path: server
    args:
      server_config:
        per_token_model_limit: 3      # Maximum number of models (adapters) per token (server-globally enforced)
      supported_models:
        - Qwen/Qwen3.5-4B
    deployments:
      - name: TinkerCompatServer
        max_ongoing_requests: 50
        autoscaling_config:
          min_replicas: 1
          max_replicas: 1
          target_ongoing_requests: 128
        ray_actor_options:
          num_cpus: 0.1

  # 2. Model service: Hosts the base model
  # Executes forward propagation, backward propagation and other training computations
  - name: models-Qwen3.5-4B
    route_prefix: /api/v1/model/Qwen/Qwen3.5-4B
    import_path: model
    args:
      use_megatron: true                               # Use Megatron-LM backend
      model_id: "ms://Qwen/Qwen3.5-4B"               # ModelScope model identifier
      max_length: 10240
      nproc_per_node: 2                                # Number of GPU processes per node
      device_group:                                    # Logical device group
        name: model
        ranks: 2                                       # Number of GPUs to use
        device_type: cuda
      device_mesh:                                     # Distributed training mesh
        device_type: cuda
        dp_size: 2                                     # Data parallel size
      queue_config:
        rps_limit: 100                                 # Max requests per second
        tps_limit: 10000                               # Max tokens per second per user
        max_input_tokens: 10000                        # Maximum input tokens per request
      adapter_config:
        adapter_timeout: 30                            # Idle adapter timeout unload time (seconds)
        adapter_max_lifetime: 36000                    # Maximum adapter lifetime (seconds)
      max_loras: 1                                     # Maximum number of LoRA adapters per model
    deployments:
      - name: ModelManagement
        autoscaling_config:
          min_replicas: 1
          max_replicas: 1
          target_ongoing_requests: 16
        ray_actor_options:
          num_cpus: 0.1
          runtime_env:
            env_vars:
              TWINKLE_TRUST_REMOTE_CODE: "0"

  # 3. Sampler service: Inference sampling
  # Uses vLLM engine for inference, supports LoRA adapters
  - name: sampler-Qwen3.5-4B
    route_prefix: /api/v1/sampler/Qwen/Qwen3.5-4B
    import_path: sampler
    args:
      model_id: "ms://Qwen/Qwen3.5-4B"               # ModelScope model identifier
      nproc_per_node: 2                                # Number of GPU processes per node
      sampler_type: vllm                               # Inference engine: vllm (high performance) or torch
      engine_args:                                     # vLLM engine parameters
        max_model_len: 4096                            # Maximum sequence length
        gpu_memory_utilization: 0.5                    # GPU memory usage ratio (0.0-1.0)
        enable_lora: true                              # Support loading LoRA during inference
        logprobs_mode: processed_logprobs              # Logprobs output mode
      device_group:                                    # Logical device group
        name: sampler
        ranks: 1                                       # Number of GPUs to use
        device_type: cuda
      device_mesh:
        device_type: cuda
        dp_size: 1
      queue_config:
        rps_limit: 100                                 # Max requests per second
        tps_limit: 100000                              # Max tokens per second
    deployments:
      - name: SamplerManagement
        autoscaling_config:
          min_replicas: 1
          max_replicas: 1
          target_ongoing_requests: 16
        ray_actor_options:
          num_cpus: 0.1
          runtime_env:
            env_vars:
              TWINKLE_TRUST_REMOTE_CODE: "0"

  # 4. Processor service: Data preprocessing
  # Executes tokenization, template conversion, and other preprocessing tasks on CPU
  - name: processor
    route_prefix: /api/v1/processor
    import_path: processor
    args:
      ncpu_proc_per_node: 2
      device_group:
        name: model
        ranks: 2
        device_type: CPU
      device_mesh:
        device_type: CPU
        dp_size: 2
    deployments:
      - name: ProcessorManagement
        autoscaling_config:
          min_replicas: 1
          max_replicas: 1
          target_ongoing_requests: 128
        ray_actor_options:
          num_cpus: 0.1
```

### Transformers Backend

The difference from the Megatron backend is only in the `use_megatron` parameter of the Model service:

```yaml
  - name: models-Qwen2.5-7B-Instruct
    route_prefix: /api/v1/model/Qwen/Qwen2.5-7B-Instruct
    import_path: model
    args:
      use_megatron: false                              # Use Transformers backend
      model_id: "ms://Qwen/Qwen2.5-7B-Instruct"
      nproc_per_node: 2
      device_group:
        name: model
        ranks: 2
        device_type: cuda
      device_mesh:
        device_type: cuda
        dp_size: 2
      adapter_config:
        adapter_timeout: 1800                          # Idle adapter timeout unload time (seconds)
        adapter_max_lifetime: 36000
    deployments:
      - name: ModelManagement
        autoscaling_config:
          min_replicas: 1
          max_replicas: 1
          target_ongoing_requests: 16
        ray_actor_options:
          num_cpus: 0.1
```

## Configuration Item Description

### Application Components (import_path)

| import_path | Description |
|-------------|------|
| `server` | Central management service, handles training runs and checkpoints |
| `model` | Model service, hosts base model for training |
| `processor` | Data preprocessing service, executes tokenization and template conversion on CPU |
| `sampler` | Inference sampling service |

### device_group and device_mesh

- **device_group**: Defines logical device groups, specifying how many GPUs to use
- **device_mesh**: Defines distributed training mesh, controls parallelization strategy

```yaml
device_group:
  name: model          # Device group name
  ranks: 2             # Number of GPUs to use
  device_type: cuda     # Device type: cuda / CPU

device_mesh:
  device_type: cuda
  dp_size: 2           # Data parallel size
  # tp_size: 1         # Tensor parallel size (optional)
  # pp_size: 1         # Pipeline parallel size (optional)
  # ep_size: 1         # Expert parallel size (optional)
```

**Important configuration parameters:**

| Parameter | Type | Description |
|------|------|------|
| `ranks` | int | **Number of GPUs to use** for this component |
| `dp_size` | int | Data parallel size |
| `tp_size` | int (optional) | Tensor parallel size |
| `pp_size` | int (optional) | Pipeline parallel size |
| `ep_size` | int (optional) | Expert parallel size (for MoE models) |

**Environment variables:**

```bash
export TWINKLE_TRUST_REMOTE_CODE=0       # Whether to trust remote code
```
