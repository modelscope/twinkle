# 分块交叉熵

交叉熵损失的内存优化变体，通过在词表维度上分块处理来减少 GPU 峰值内存使用。

```python
from twinkle.loss import ChunkedCrossEntropyLoss

loss_fn = ChunkedCrossEntropyLoss(
    chunk_size=1024,           # 词表分块大小
    reduction='mean',
)

model.set_loss(loss_fn)
```

**参数:**
- `chunk_size`: 每块处理的词表 token 数量（默认: 1024）
- `reduction`: 归约模式 — `sum`, `mean`, 或 `none`

实现使用自定义 autograd 函数，沿词表维度将 logit 到损失的计算分块进行。这避免了实例化完整的 `[batch*seq_len, vocab_size]` 概率张量，显著减少了大词表模型的内存占用。

> 当训练大词表模型时标准交叉熵导致 OOM 错误时非常有用。
