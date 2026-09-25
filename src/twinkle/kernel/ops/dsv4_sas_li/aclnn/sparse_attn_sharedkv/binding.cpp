// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/extension.h>
#include "../_aclnn_common.h"

at::Tensor npu_sparse_attn_sharedkv_metadata(
    const c10::optional<at::Tensor> &cuSeqLensQ,
    const c10::optional<at::Tensor> &sequsedOriKv,
    const c10::optional<at::Tensor> &sequsedCmpKv,
    const c10::optional<at::Tensor> &sequsedQ,
    const c10::optional<at::Tensor> &sequsedKv,
    int64_t numHeadsQ,
    int64_t numHeadsKv,
    int64_t headDim,
    int64_t batchSize,
    int64_t maxSeqLenQ,
    int64_t maxSeqLenKv,
    int64_t oriTopk,
    int64_t cmpTopk,
    int64_t cmpRatio,
    int64_t oriMaskMode,
    int64_t cmpMaskMode,
    int64_t oriWinLeft,
    int64_t oriWinRight,
    const c10::optional<std::string> layoutQ,
    const c10::optional<std::string> layoutKv,
    bool hasOriKv,
    bool hasCmpKv)
{
    std::string layoutQStr = layoutQ.value_or("SBH");
    std::string layoutKvStr = layoutKv.value_or("SBH");
    const char *layoutQPtr = layoutQStr.c_str();
    const char *layoutKvPtr = layoutKvStr.c_str();
    at::Tensor metadata = at::empty(1024, at::TensorOptions(torch_npu::utils::get_npu_device_type()).dtype(at::kInt));
    ACLNN_CMD(aclnnSparseAttnSharedkvMetadata,
              cuSeqLensQ, sequsedOriKv, sequsedCmpKv,
              sequsedQ, sequsedKv,
              numHeadsQ, numHeadsKv, headDim, batchSize,
              maxSeqLenQ, maxSeqLenKv, oriTopk, cmpTopk, cmpRatio,
              oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight,
              layoutQPtr, layoutKvPtr,
              hasOriKv, hasCmpKv,
              metadata);
    return metadata;
}

std::tuple<at::Tensor, at::Tensor> npu_sparse_attn_sharedkv(
    const at::Tensor &query,
    const c10::optional<at::Tensor> &oriKv,
    const c10::optional<at::Tensor> &cmpKv,
    const c10::optional<at::Tensor> &oriSparseIndices,
    const c10::optional<at::Tensor> &cmpSparseIndices,
    const c10::optional<at::Tensor> &oriBlockTable,
    const c10::optional<at::Tensor> &cmpBlockTable,
    const c10::optional<at::Tensor> &cuSeqLensQ,
    const c10::optional<at::Tensor> &cuSeqLensOriKv,
    const c10::optional<at::Tensor> &cuSeqLensCmpKv,
    const c10::optional<at::Tensor> &sequsedQ,
    const c10::optional<at::Tensor> &sequsedKv,
    const c10::optional<at::Tensor> &sinks,
    const c10::optional<at::Tensor> &metadata,
    double softmaxScale,
    int64_t cmpRatio,
    int64_t oriMaskMode,
    int64_t cmpMaskMode,
    int64_t oriKvStride,
    int64_t cmpKvStride,
    int64_t oriWinLeft,
    int64_t oriWinRight,
    const c10::optional<std::string> layoutQ,
    const c10::optional<std::string> layoutKv,
    bool returnSoftmaxLse)
{
    std::string layoutq = layoutQ.value_or("SBH");
    std::string layoutkv = layoutKv.value_or("SBH");
    const char *layoutQPtr = layoutq.c_str();
    const char *layoutKvPtr = layoutkv.c_str();

    at::Tensor attnOutput = at::empty(query.sizes(), query.options());
    at::Tensor softmaxLseOut;
    if (returnSoftmaxLse) {
        std::vector<int64_t> lse_sizes(query.sizes().begin(), query.sizes().end());
        lse_sizes.back() = 1;
        softmaxLseOut = at::empty(lse_sizes, query.options().dtype(c10::ScalarType::Float));
    } else {
        softmaxLseOut = at::Tensor();
    }

    ACLNN_CMD(aclnnSparseAttnSharedkv,
              query, oriKv, cmpKv,
              oriSparseIndices, cmpSparseIndices,
              oriBlockTable, cmpBlockTable,
              cuSeqLensQ, cuSeqLensOriKv, cuSeqLensCmpKv,
              sequsedQ, sequsedKv,
              sinks, metadata,
              softmaxScale, cmpRatio, oriMaskMode, cmpMaskMode,
              oriKvStride, cmpKvStride,
              oriWinLeft, oriWinRight,
              layoutQPtr, layoutKvPtr,
              returnSoftmaxLse,
              attnOutput, softmaxLseOut);
    return std::make_tuple(attnOutput, softmaxLseOut);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_sparse_attn_sharedkv_grad(
    const at::Tensor &query,
    const at::Tensor &oriKv,
    const c10::optional<at::Tensor> &cmpKv,
    const c10::optional<at::Tensor> &dOut,
    const c10::optional<at::Tensor> &out,
    const c10::optional<at::Tensor> &lse,
    const c10::optional<at::Tensor> &oriSparseIndices,
    const c10::optional<at::Tensor> &cmpSparseIndices,
    const c10::optional<at::Tensor> &cuSeqlensQ,
    const c10::optional<at::Tensor> &cuSeqlensOriKv,
    const c10::optional<at::Tensor> &cuSeqlensCmpKv,
    const at::Tensor &sinks,
    double scaleValue,
    int64_t cmpRatio,
    int64_t oriMaskMode,
    int64_t cmpMaskMode,
    int64_t oriWinLeft,
    int64_t oriWinRight,
    const c10::optional<std::string> layout)
{
    std::string layoutValue = layout.value_or("SBH");
    const char *layoutPtr = layoutValue.c_str();

    at::Tensor dQuery = at::empty(query.sizes(), query.options());
    at::Tensor dOriKv = at::empty(oriKv.sizes(), oriKv.options());
    at::Tensor dSinks = at::empty(sinks.sizes(), sinks.options());

    at::Tensor dCmpKv;
    if (cmpRatio > 1 && cmpKv.has_value() && cmpKv.value().defined()) {
        dCmpKv = at::empty(cmpKv.value().sizes(), cmpKv.value().options());
    } else {
        dCmpKv = at::Tensor();
    }

    ACLNN_CMD(aclnnSparseAttnSharedkvGrad,
              query, oriKv, cmpKv,
              dOut, out, lse,
              oriSparseIndices, cmpSparseIndices,
              cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv,
              sinks,
              scaleValue, cmpRatio, oriMaskMode, cmpMaskMode,
              oriWinLeft, oriWinRight,
              layoutPtr,
              dQuery, dOriKv, dCmpKv, dSinks);
    return std::make_tuple(dQuery, dOriKv, dCmpKv, dSinks);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_sparse_attn_sharedkv_metadata", &npu_sparse_attn_sharedkv_metadata,
          "Shared-KV Sparse Attention Metadata");
    m.def("npu_sparse_attn_sharedkv", &npu_sparse_attn_sharedkv,
          "Shared-KV Sparse Attention Forward");
    m.def("npu_sparse_attn_sharedkv_grad", &npu_sparse_attn_sharedkv_grad,
          "Shared-KV Sparse Attention Backward");
}
