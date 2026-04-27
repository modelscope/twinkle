import functools
import torch
import torch_npu


class GmmFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, group_list: torch.Tensor, weight_ekn: torch.Tensor):
        assert x.dim() == 2, f'x must be [M, K], got {tuple(x.shape)}'
        assert group_list.dim() == 1, f'group_list must be [E], got {tuple(group_list.shape)}'
        assert weight_ekn.dim() == 3, f'weight_ekn must be [E, K, N], got {tuple(weight_ekn.shape)}'
        assert group_list.numel() == weight_ekn.size(0), (
            f'group_list len {group_list.numel()} != num_experts {weight_ekn.size(0)}')
        assert x.size(1) == weight_ekn.size(1), (
            f'input dim mismatch: x.shape={tuple(x.shape)}, weight_ekn.shape={tuple(weight_ekn.shape)}')

        group_list = group_list.to(torch.int64)

        ctx.save_for_backward(x, group_list, weight_ekn)

        outputs = torch_npu.npu_grouped_matmul(
            [x],
            [weight_ekn],
            group_list=group_list,
            group_type=0,
            split_item=2,
            group_list_type=1,
        )
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, group_list, weight_ekn = ctx.saved_tensors

        grad_input = torch_npu.npu_grouped_matmul(
            [grad_output],
            [weight_ekn.transpose(-2, -1).contiguous()],
            bias=None,
            group_list=group_list,
            group_type=0,
            split_item=2,
            group_list_type=1,
        )[0]

        grad_weight = torch_npu.npu_grouped_matmul(
            [x.transpose(0, 1)],
            [grad_output],
            bias=None,
            group_list=group_list,
            group_type=2,
            split_item=3,
            group_list_type=1,
        )[0]

        return grad_input, None, grad_weight.contiguous()


def _grouped_mm_npu(input: torch.Tensor, weight_ekn: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    assert input.dim() == 2, f'input must be [M, K], got {tuple(input.shape)}'
    assert weight_ekn.dim() == 3, f'weight_ekn must be [E, K, N], got {tuple(weight_ekn.shape)}'
    assert offs.dim() == 1, f'offs must be [E], got {tuple(offs.shape)}'
    assert weight_ekn.size(0) == offs.numel(), (
        f'weight_ekn.size(0)={weight_ekn.size(0)} != offs.numel()={offs.numel()}')

    counts = torch.empty_like(offs)
    counts[0] = offs[0]
    if offs.numel() > 1:
        counts[1:] = offs[1:] - offs[:-1]
    counts = counts.to(torch.int64)

    return GmmFunction.apply(input, counts, weight_ekn)


def apply_hf_moe_grouped_mm_patch():
    import transformers.integrations.moe as hf_moe

    hf_moe._grouped_mm = _grouped_mm_npu
    print('[PATCH] transformers.integrations.moe._grouped_mm -> _grouped_mm_npu')


def apply_npu_patch():
    import torch
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    apply_hf_moe_grouped_mm_patch()
