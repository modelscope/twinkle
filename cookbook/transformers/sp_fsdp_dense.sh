#!/bin/bash
# To enabele sequence parallelism, please set ulysses_size > 1
# device_mesh = DeviceMesh(
#     device_type="cuda",
#     mesh=np.arange(4).reshape(2, 2),
#     mesh_dim_names=("dp", "fsdp"),
#     ulysses_size=2,
# )
#
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 sp_fsdp_dense.py