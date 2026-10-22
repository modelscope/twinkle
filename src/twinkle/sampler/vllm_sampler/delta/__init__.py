# Copyright (c) ModelScope Contributors. All rights reserved.
"""vLLM-side adapters for delta weight sync (bound to vLLM internals, so kept out of ``checkpoint_engine.delta``).

    transfer_engine.py     vLLM ``WeightTransferEngine`` backends: ``twinkle_delta_ipc`` (GPU) and
                           ``twinkle_delta_hccl`` (NPU); decode each flush and apply it via vLLM's patch loaders
    cuda_ipc.py            bucketed zmq + CUDA-IPC channel between a CudaDeltaReceiver and its vLLM worker (GPU only)
    weight_update_utils.py refresh of derived weights (tied aliases, packed conv layouts) after a sparse patch
"""
