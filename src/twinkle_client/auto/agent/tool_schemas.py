# Copyright (c) Twinkle Contributors. All rights reserved.
"""OpenAI function-calling schemas exposed by the auto agent."""
from __future__ import annotations

from typing import Any

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        'type': 'function',
        'function': {
            'name': 'list_training_runs',
            'description': 'List all active and historical training runs.',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': []
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_training_status',
            'description': 'Get detailed status and recent metrics for a training run.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'run_id': {
                        'type': 'string',
                        'description': 'Training run ID.'
                    },
                },
                'required': ['run_id'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name':
            'start_server',
            'description': ('Start Ray cluster and Twinkle Server. MUST be called before start_training. '
                            'Idempotent: skips if server is already reachable. '
                            'Supports multi-model deployments: one training model + N sampler/teacher models. '
                            'Automatically generates server_config.yaml from parameters.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'model_id': {
                        'type': 'string',
                        'description': 'Student/training model ID (e.g. "Qwen/Qwen3.5-4B").',
                    },
                    'train_gpus': {
                        'type': 'integer',
                        'description': 'GPUs for the training model. Default: auto-detect remaining GPUs.',
                    },
                    'backend': {
                        'type': 'string',
                        'enum': ['transformers', 'megatron'],
                        'description': 'Training model backend. Default: transformers.',
                    },
                    'samplers': {
                        'type':
                        'array',
                        'description': ('List of sampler/teacher models for RL/OPD. Each entry deploys '
                                        'an inference service (vLLM or torch). Omit for simple SFT.'),
                        'items': {
                            'type': 'object',
                            'properties': {
                                'model_id': {
                                    'type': 'string',
                                    'description': 'Teacher/reference model ID (e.g. "Qwen/Qwen3.5-72B").',
                                },
                                'gpus': {
                                    'type': 'integer',
                                    'description':
                                    'Total number of GPUs for this sampler. Default: 1. Must equal tp * dp.',
                                },
                                'tp': {
                                    'type':
                                    'integer',
                                    'description':
                                    ('Tensor parallelism size (GPUs per vLLM worker process). '
                                     'Use tp>1 for large models that do not fit on a single GPU. Default: 1.'),
                                },
                                'dp': {
                                    'type':
                                    'integer',
                                    'description': ('Data parallelism size (number of independent inference replicas). '
                                                    'If not specified, computed as gpus // tp. Default: 1.'),
                                },
                                'engine': {
                                    'type': 'string',
                                    'enum': ['vllm', 'torch'],
                                    'description': 'Inference engine. Default: vllm.',
                                },
                                'max_model_len': {
                                    'type': 'integer',
                                    'description': 'Max sequence length for inference. Default: 16000.',
                                },
                            },
                            'required': ['model_id'],
                        },
                    },
                    'port': {
                        'type': 'integer',
                        'description': 'HTTP port for server. Default: 8000.',
                    },
                },
                'required': ['model_id'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name':
            'shutdown_server',
            'description': ('Shut down Twinkle Server and Ray cluster. WARNING: This releases all GPU resources '
                            'and DESTROYS model state held in server memory. Only call when training is truly '
                            'finished and you no longer need the server. Model weights/optimizer state in GPU '
                            'will be LOST unless a checkpoint was explicitly saved.'),
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': []
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name':
            'start_training',
            'description': ('Create a new training run: write the client script, launch it, and start monitoring. '
                            'REQUIRES: Twinkle Server must be running (call start_server first). '
                            'The client script connects to the server — server holds model state in GPU memory. '
                            'Kill client = pause (state preserved). Re-launch client = resume.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'run_id': {
                        'type': 'string',
                        'description': 'Unique run ID (e.g., "grpo-gsm8k").'
                    },
                    'script_content': {
                        'type': 'string',
                        'description': 'Full Python source code of the training script.'
                    },
                    'model_id': {
                        'type': 'string',
                        'description': 'Model identifier for metadata (e.g., "Qwen/Qwen3.5-4B").'
                    },
                },
                'required': ['run_id', 'script_content'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'select_run',
            'description': 'Switch to monitor a different training run. Updates connection context.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'run_id': {
                        'type': 'string',
                        'description': 'Training run ID to monitor.'
                    },
                },
                'required': ['run_id'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name':
            'pause_training',
            'description': ('Pause training by killing the client process (SIGKILL). '
                            'Server retains all state — call resume_training to continue.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'run_id': {
                        'type': 'string',
                        'description': 'Training run ID to pause.'
                    },
                },
                'required': ['run_id'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'resume_training',
            'description': 'Resume a paused training run by re-launching the client script. Server state is preserved.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'run_id': {
                        'type': 'string',
                        'description': 'Training run ID to resume.'
                    },
                },
                'required': ['run_id'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name':
            'stop_training',
            'description': ('Gracefully stop the training client (SIGTERM). The script saves a checkpoint '
                            'before exiting. Server retains model/optimizer state in GPU memory — '
                            'use resume_training to continue. Similar to pause_training but with checkpoint save. '
                            'To fully release GPU resources, use shutdown_server.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'run_id': {
                        'type': 'string',
                        'description': 'Training run ID to stop.'
                    },
                },
                'required': ['run_id'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name':
            'update_script',
            'description':
            ('Update the training script for a run. Archives the current train.py as train_v{N}.py '
             'and writes the new version. Use after diagnosing a script error, then call resume_training.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'run_id': {
                        'type': 'string',
                        'description': 'Training run ID.'
                    },
                    'script_content': {
                        'type': 'string',
                        'description': 'Full Python source code of the new training script.'
                    },
                },
                'required': ['run_id', 'script_content'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name':
            'list_supported_models',
            'description': ('Query the Twinkle server for its list of supported base models. '
                            'Always call this before writing a training script to verify model availability.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'base_url': {
                        'type':
                        'string',
                        'description':
                        'Server base URL. Default: http://localhost:8000. Cloud: http://www.modelscope.cn/twinkle',
                    },
                },
                'required': [],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'search_datasets',
            'description': 'Search ModelScope Hub for datasets matching a query.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'Search query for datasets.'
                    },
                    'limit': {
                        'type': 'integer',
                        'description': 'Max results (default 5).'
                    },
                },
                'required': ['query'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'search_models',
            'description': 'Search ModelScope Hub for models matching a query.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'Search query for models.'
                    },
                    'limit': {
                        'type': 'integer',
                        'description': 'Max results (default 5).'
                    },
                },
                'required': ['query'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name':
            'get_cluster_info',
            'description': ('Get cluster GPU resource info for planning training parallelism. '
                            'First attempts to query a running Ray cluster; if Ray is not available, '
                            'falls back to nvidia-smi for local GPU discovery. '
                            'The result indicates whether Ray is active — if not, the training script '
                            'should either start a local Ray cluster itself or the user should launch '
                            'Ray manually (see server mode run.sh).'),
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': []
            },
        },
    },
]
