// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/extension.h>
#include "../_aclnn_common.h"

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_lightning_indexer_grad(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &dy,
    const at::Tensor &sparse_indices,
    const at::Tensor &weights,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_key,
    const c10::optional<std::string> layout,
    c10::optional<int64_t> sparse_mode,
    c10::optional<int64_t> pre_tokens,
    c10::optional<int64_t> next_tokens,
    c10::optional<int64_t> cmp_ratio,
    c10::optional<int64_t> head_num)
{
    at::Tensor d_query = at::zeros(query.sizes(), query.options());
    at::Tensor d_key = at::zeros(key.sizes(), key.options());
    at::Tensor d_weights = at::zeros(weights.sizes(), weights.options());

    std::string layout_str = layout.value_or("BSND");
    char *layout_ptr = const_cast<char *>(layout_str.c_str());
    const int64_t sparse_mode_val = sparse_mode.value_or(0);
    const int64_t pre_tokens_val = pre_tokens.value_or(9223372036854775807LL);
    const int64_t next_tokens_val = next_tokens.value_or(9223372036854775807LL);
    const int64_t cmp_ratio_val = cmp_ratio.value_or(1);
    const int64_t head_num_val = head_num.value_or(64);
    const bool deterministic = false;

    ACLNN_CMD(aclnnLightningIndexerGrad,
              query, key, dy, sparse_indices, weights,
              actual_seq_lengths_query, actual_seq_lengths_key,
              head_num_val, layout_ptr,
              sparse_mode_val, pre_tokens_val, next_tokens_val,
              deterministic, cmp_ratio_val,
              d_query, d_key, d_weights);

    return std::make_tuple(d_query, d_key, d_weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_lightning_indexer_grad", &npu_lightning_indexer_grad,
          "Lightning Indexer Backward");
}
