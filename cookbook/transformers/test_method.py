#固定随机
import random
import numpy as np
def seed_all_own(seed=1234, mode=True, is_gpu=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['GLOBAL_SEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(mode)
    if is_gpu:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enable = False
        torch.backends.cudnn.benchmark = False
    else:
        import torch_npu
        os.environ['HCCL_DETERMINISTIC'] = 'true'
        os.environ['CLOSE_MATMUL_K_SHIFT'] = '1'
        torch_npu.npu.manual_seed_all(seed)
        torch_npu.npu.manual_seed(seed)
    print("====== seed all ========")
seed_all_own(is_gpu=False)
from msprobe.pytorch import seed_all
seed_all(mode=True)

def get_time():
    import time
    torch.npu.synchronize()
    return time.time()


def set_modules_to_forward_prefetch(block, num_to_forward_prefetch):
    for i, layer in enumerate(block.layers):
        if i < num_to_forward_prefetch:
            continue
        layers_to_prefetch = [layers[i + j] for j in range(1, num_to_forward_prefetch + 1)]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)


def set_modules_to_backward_prefetch(block, num_to_backward_prefetch):
    for i, layer in enumerate(block.layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [layers[i - j] for j in range(1, num_to_backward_prefetch + 1)]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)
