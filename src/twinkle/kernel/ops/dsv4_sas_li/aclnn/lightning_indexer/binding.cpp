// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/extension.h>
#include "../_aclnn_common.h"

std::tuple<at::Tensor, at::Tensor> npu_lightning_indexer(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &weights,
    const c10::optional<at::Tensor> &actual_seq_q,
    const c10::optional<at::Tensor> &actual_seq_k,
    const c10::optional<at::Tensor> &block_table,
    std::string layout_q,
    std::string layout_k,
    int64_t sparse_count,
    int64_t sparse_mode,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t cmp_ratio,
    bool return_values)
{
    TORCH_CHECK(query.numel() > 0, "query is empty");

    const char *layout_q_ptr = layout_q.c_str();
    const char *layout_k_ptr = layout_k.c_str();

    auto q_sizes = query.sizes();
    int64_t B  = q_sizes[0];
    int64_t S1 = q_sizes[1];
    int64_t N2 = key.size(2);

    auto opts_int = at::TensorOptions().dtype(at::kInt).device(query.device());
    at::Tensor sparse_indices = at::empty({B, S1, N2, sparse_count}, opts_int);
    at::Tensor sparse_values = at::empty({B, S1, N2, sparse_count}, query.options());

    ACLNN_CMD(aclnnLightningIndexer,
              query, key, weights,
              actual_seq_q, actual_seq_k, block_table,
              layout_q_ptr, layout_k_ptr,
              sparse_count, sparse_mode, pre_tokens, next_tokens,
              cmp_ratio, return_values,
              sparse_indices, sparse_values);

    return {sparse_indices, sparse_values};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_lightning_indexer", &npu_lightning_indexer, "Lightning Indexer");
}
