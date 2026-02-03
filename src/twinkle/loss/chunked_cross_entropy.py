# Copyright (c) ModelScope Contributors. All rights reserved.
import math
from typing import Any
from .base import Loss


class ChunkedCrossEntropyLoss(Loss):

    def __init__(self, chunk_size):
        self.chunk_size = chunk_size

    def __call__(self, inputs, outputs, **kwargs):
        import torch

        class ChunkedCrossEntropyLossFunc(torch.autograd.Function):

            @staticmethod
            def forward(ctx, logits, labels, chunk_size):
                import torch
                ctx.save_for_backward(logits, labels)
                ctx.chunk_size = chunk_size

                losses = []
                for i in range(math.ceil(logits.shape[0] / chunk_size)):
                    l_start = i * chunk_size
                    l_end = min((i + 1) * chunk_size, logits.shape[0])
                    logits_chunk = logits[l_start:l_end]
                    labels_chunk = labels[l_start:l_end]
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    loss_chunk = loss_fct(logits_chunk, labels_chunk)
                    losses.append(loss_chunk)
                    del logits_chunk
                    del labels_chunk
                all_losses = torch.cat(losses)
                return all_losses

            @staticmethod
            def backward(ctx: Any, *grad_outputs: Any):
                import torch
                logits, labels = ctx.saved_tensors
                chunk_size = ctx.chunk_size

                for i in range(math.ceil(logits.shape[0] / chunk_size)):
                    l_start = i * chunk_size
                    l_end = min((i + 1) * chunk_size, logits.shape[0])
                    logits_chunk = logits[l_start:l_end].detach().requires_grad_(True)
                    labels_chunk = labels[l_start:l_end]
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    with torch.enable_grad():
                        loss_chunk = loss_fct(logits_chunk, labels_chunk)
                        grad_output_chunk = grad_outputs[0][l_start:l_end]
                        _loss_chunk = (loss_chunk * grad_output_chunk).sum()
                        grad_chunk = torch.autograd.grad(_loss_chunk, logits_chunk, retain_graph=False)[0]
                        logits[l_start:l_end] = grad_chunk

                return logits, None, None
                
        logits = outputs['logits']
        labels = inputs['labels']
        return ChunkedCrossEntropyLossFunc.apply(logits, labels, self.chunk_size)