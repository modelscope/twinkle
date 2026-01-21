import torch
from tinker import types
from typing import List, Union
from twinkle.model.multi_lora_transformers import MultiLoraTransformersModel
from twinkle import remote_class, remote_function
from .datum import datum_to_input_feature


@remote_class()
class TwinkleCompatTransformersModel(MultiLoraTransformersModel):
    
    @remote_function(collect='flatten')
    def forward(self, *, inputs: List[types.Datum], **kwargs):
        # Convert Datum to InputFeature
        input_features = [datum_to_input_feature(datum) for datum in inputs]
        
        outputs = super().forward(inputs=input_features, **kwargs)
        logits = outputs['logits'] # shape (batch_size, seq_len, vocab_size)
        
        # breakpoint()
        results = []
        for i, feature in enumerate(inputs):
            # Ensure 1D shape and correct device to avoid dimension mismatch and device errors
            labels = feature.loss_fn_inputs['target_tokens'].to_torch().long().view(-1).to(logits.device) # shape (seq_len,)
            weights = feature.loss_fn_inputs['weights'].to_torch().view(-1).to(logits.device) # shape (seq_len,)
            
            # Slice logits to match the sequence length of labels
            # Labels are assumed to be already shifted/aligned with logits
            seq_len = labels.numel()
            feature_logits = logits[i, :seq_len, :]

            # Calculate log probs for all labels
            # Apply log_softmax to convert raw logits to log-probabilities
            feature_log_probs = torch.log_softmax(feature_logits, dim=-1)
            token_log_probs = feature_log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            
            # elementwise_loss: positive NLL loss (0.0 where masked)
            elementwise_loss = -token_log_probs * weights

            results.append({
                'logprobs': types.TensorData.from_torch(token_log_probs),
                'elementwise_loss': types.TensorData.from_torch(elementwise_loss)
            })
    
        return results
    
    @remote_function(collect='sum')
    def calculate_loss(self, **kwargs):
        loss = super().calculate_loss(**kwargs)
        return loss.item()
    
    @remote_function()
    def step(self, *, adam_params: types.AdamParams, **kwargs):
        # Gradient clipping
        grad_clip_norm = adam_params.grad_clip_norm
        if grad_clip_norm > 0.0:
            self.clip_grad_norm(max_grad_norm=grad_clip_norm, norm_type=2, **kwargs)
        # Optimizer step
        optim_params = {
            'lr': adam_params.learning_rate,
            'eps': adam_params.eps,
            'betas': (adam_params.beta1, adam_params.beta2),
            'weight_decay': adam_params.weight_decay,
        }
        super().step(optim_params=optim_params, **kwargs)
        # Zero gradients
        self.zero_grad(**kwargs)