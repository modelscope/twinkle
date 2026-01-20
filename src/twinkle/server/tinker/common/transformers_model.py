import torch
from tinker import types
from typing import List, Union
from twinkle.data_format.input_feature import InputFeature
from twinkle.model.multi_lora_transformers import MultiLoraTransformersModel
from twinkle import remote_class, remote_function


@remote_class()
class TwinkleCompatTransformersModel(MultiLoraTransformersModel):
    
    @remote_function()
    def forward(self, *, inputs: List[InputFeature], **kwargs):
        outputs = super().forward(inputs=inputs, **kwargs)
        logits = outputs['logits'] # shape (batch_size, seq_len, vocab_size)
        # gather log probabilities of each input_feature token with input_ids and attention_mask
        
        batch_log_probs = []
        batch_loss = []
        
        for i, feature in enumerate(inputs):
            device = logits.device
            # Ensure we use the full length of the input feature as requested
            input_ids = torch.tensor(feature['input_ids'], device=device, dtype=torch.long)
            # Use labels provided in feature, assume they align with input_ids
            labels = torch.tensor(feature['labels'], device=device, dtype=torch.long)
            
            length = input_ids.numel()
            
            # Initialize results with zeros matching input length
            # Index 0 will remain 0.0 because it cannot be predicted by previous tokens
            token_log_probs = torch.zeros(length, device=device, dtype=logits.dtype)
            element_loss = torch.zeros(length, device=device, dtype=logits.dtype)
            
            # Causal LM shifting: logits[t] predicts input_ids[t+1]
            if length > 1:
                # We use logits 0..L-2 to predict tokens 1..L-1
                # logits slice shape: (length-1, vocab_size)
                feature_logits = logits[i, :length-1, :]
                
                # Targets are shifted by 1
                target_ids = input_ids[1:]
                target_labels = labels[1:]
                
                # Calculate log soft max
                log_probs = feature_logits.log_softmax(dim=-1)
                
                # Gather log probability of the actual next token
                # shape: (length-1,)
                gathered_log_probs = log_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
                
                # Fill into result tensor starting from index 1
                token_log_probs[1:] = gathered_log_probs
                
                # Calculate elementwise loss (NLL)
                # Mask out loss where label is -100
                loss_mask = (target_labels != -100)
                step_loss = -gathered_log_probs
                
                # Apply mask: where mask is True keep loss, else 0.0
                step_loss = torch.where(loss_mask, step_loss, torch.zeros_like(step_loss))
                
                element_loss[1:] = step_loss
                
            batch_log_probs.append(token_log_probs)
            batch_loss.append(element_loss)
            
        # Return as list of dict, one per input feature
        result = []
        for log_probs, loss in zip(batch_log_probs, batch_loss):
            result.append({
            'logprobs': types.TensorData.from_torch(log_probs),
            'elementwise_loss': types.TensorData.from_torch(loss)
            })
        return result