# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np
import torch
from typing import List, TYPE_CHECKING, Tuple, Optional, Any
from tinker import types
from twinkle.template import Template
from twinkle import remote_class, remote_function
from twinkle.utils import exists, requires
from .datum import datum_to_input_feature, extract_rl_feature
from .io_utils import create_checkpoint_manager


if TYPE_CHECKING:
    from twinkle.model.megatron import MultiLoraMegatronModel as _MegatronBase
elif exists('megatron_core'):
    # Use module-level import to trigger LazyModule's __getattr__ correctly
    import twinkle.model.megatron as megatron_module
    _MegatronBase = megatron_module.MultiLoraMegatronModel
else:
    class _MegatronBase:
        def __init__(self, *args, **kwargs):
            requires('megatron_core')


def _collect_forward_backward_results(results):
    """Custom collect function for forward_backward that handles list [outputs, loss].
    
    Args:
        results: List of lists from each worker, where each list is [outputs_list, loss_float]
        
    Returns:
        List of [flattened_outputs, averaged_loss]
    """
    if not results:
        return results
    
    # results is a list of lists: [[outputs1, loss1], [outputs2, loss2], ...]
    # Flatten outputs (first element of each list)
    all_outputs = []
    all_losses = []
    for result in results:
        outputs, loss = result
        all_outputs.extend(outputs)
        all_losses.append(loss)
    
    # Average the losses
    avg_loss = float(np.mean(all_losses))
    
    return [all_outputs, avg_loss]


@remote_class(execute='all')
class TwinkleCompatMegatronModel(_MegatronBase):
    """
    Compatibility wrapper around :class:`MultiLoraMegatronModel` for Twinkle/Tinker.

    This class adapts the core `MultiLoraMegatronModel` API to the data types and
    remote-call semantics used by Twinkle:

    * Inputs to :meth:`forward_backward` and :meth:`forward_only` are provided as
      ``List[types.Datum]`` and are converted to the underlying model's
      ``InputFeature`` format via :func:`datum_to_input_feature`.
    * The outputs are a list of dictionaries, one per input example, containing:

        - ``"logprobs"``: token-level log-probabilities as ``types.TensorData``.
        - ``"elementwise_loss"``: per-token (masked) NLL loss as ``types.TensorData``.

      These are derived from the underlying logits by applying ``log_softmax``
      and slicing to the label sequence length.
    * :meth:`forward_backward` returns a tuple of (outputs, loss) where loss is a
      Python scalar for the aggregated loss.
    * :meth:`step` accepts optimizer hyperparameters as :class:`types.AdamParams`,
      and updates the optimizer configuration before calling the base ``step``.

    Note: Megatron uses combined forward_backward instead of separate forward/backward.
    This wrapper provides a direct forward_backward interface.
    """

    @remote_function(dispatch='slice_dp', collect=_collect_forward_backward_results, sync=True)
    def forward_backward(self, *, inputs: List[types.Datum], adapter_name: str, loss_fn: str, **kwargs):
        """Combined forward and backward pass.
        
        Returns:
            Tuple of (outputs, loss) where outputs is a list of dicts with
            'logprobs' and 'elementwise_loss', and loss is a scalar.
        """
        if loss_fn == 'importance_sampling':
            super().set_loss('GRPOLoss',
                             adapter_name=adapter_name,
                             epsilon=0.2,  # Default GRPO epsilon
                             beta=0.0)     # No KL penalty by default
        # Get template for input processing
        template = self.get_template(adapter_name=adapter_name)
        # Convert Datum to InputFeature
        input_features = datum_to_input_feature(inputs, template)
        # Extract old_logps and advantages using common utility
        loss_values = extract_rl_feature(inputs)
        loss_kwargs = kwargs.copy()
        loss_kwargs.update(loss_values)
        # Megatron forward_backward returns loss directly
        loss = super().forward_backward(inputs=input_features, adapter_name=adapter_name, **loss_kwargs)
        
        # Get logits from outputs
        optimizer_config = self.optimizer_group.get(adapter_name)
        outputs = optimizer_config.outputs if optimizer_config else {}
        logits_list = outputs.get('logits', [])
        
        # Process logits to match transformers output format
        if logits_list and len(logits_list) > 0:
            if isinstance(logits_list, torch.Tensor):
                logits = logits_list.detach().cpu()
            else:
                # Concatenate logits from multiple microbatches
                logits = torch.cat([l.detach().cpu() for l in logits_list], dim=0)
            results = self._get_forward_output(inputs, logits)
        else:
            # If no logits available (non-last PP stage), return empty results
            results = [{'logprobs': None, 'elementwise_loss': None} for _ in inputs]
        
        # Convert loss to scalar
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        else:
            loss = float(loss)
            
        return [results, loss]

    @remote_function(dispatch='slice_dp', collect='flatten')
    def forward_only(self, *, inputs: List[types.Datum], **kwargs):
        """Forward pass without gradient computation."""
        # Get template for input processing
        template = self.get_template(**kwargs)
        # Convert Datum to InputFeature
        input_features = datum_to_input_feature(inputs, template)
        
        outputs = super().forward_only(inputs=input_features, **kwargs)
        
        # Get logits
        logits = outputs.get('logits', None) if isinstance(outputs, dict) else None
        
        if logits is not None:
            if isinstance(logits, torch.Tensor):
                logits = logits.detach().cpu()
            elif isinstance(logits, list) and len(logits) > 0:
                logits = torch.cat([l.detach().cpu() for l in logits], dim=0)
            results = self._get_forward_output(inputs, logits)
        else:
            # If no logits available (non-last PP stage), return empty results
            results = [{'logprobs': None, 'elementwise_loss': None} for _ in inputs]
        
        return results

    @remote_function(dispatch='all')
    def step(self, *, adam_params: types.AdamParams, **kwargs):
        """Optimizer step with AdamParams configuration.
        
        Updates the optimizer configuration and performs the step.
        """
        adapter_name = kwargs.get('adapter_name')
        optimizer_config = self.optimizer_group.get(adapter_name)
        
        if optimizer_config and optimizer_config.optimizer:
            # Update optimizer config with adam_params
            # Megatron optimizer handles gradient clipping internally
            opt = optimizer_config.optimizer
            if hasattr(opt, 'chained_optimizers'):
                for chained_opt in opt.chained_optimizers:
                    if hasattr(chained_opt, 'config'):
                        chained_opt.config.lr = adam_params.learning_rate
                        chained_opt.config.adam_eps = adam_params.eps
                        chained_opt.config.adam_beta1 = adam_params.beta1
                        chained_opt.config.adam_beta2 = adam_params.beta2
                        chained_opt.config.weight_decay = adam_params.weight_decay
                        if adam_params.grad_clip_norm > 0:
                            chained_opt.config.clip_grad = adam_params.grad_clip_norm
        
        # Perform optimizer step
        super().step(**kwargs)
        # Zero gradients
        super().zero_grad(**kwargs)

    @remote_function(dispatch='all', sync=True)
    def load(self, checkpoint_dir: str, **kwargs):
        """
        Load checkpoint with token-based isolation support.
        
        Args:
            checkpoint_dir: The twinkle:// path to the checkpoint or hub model ID
            **kwargs: Additional keyword arguments including optional 'token'
        """
        # Extract token from kwargs if provided (for user isolation)
        token = kwargs.pop('token', None)
        if not token:
            raise ValueError("Token is required for loading checkpoints")
        
        # Create checkpoint manager with the token
        checkpoint_manager = create_checkpoint_manager(token)
        
        # Use resolve_load_path to handle path resolution
        resolved = checkpoint_manager.resolve_load_path(checkpoint_dir)
        
        if resolved.is_twinkle_path:
            # Load from twinkle checkpoint
            return super().load(name=resolved.checkpoint_name, output_dir=str(resolved.checkpoint_dir), **kwargs)
        else:
            # Load from hub
            return super().load(name=resolved.checkpoint_name, **kwargs)

    def get_template(self, adapter_name: str) -> Template:
        return self.optimizer_group[adapter_name].template
    
    @staticmethod
    def _get_forward_output(inputs: List[types.Datum], logits: torch.Tensor) -> List[dict]:
        """Convert raw logits to the expected output format with logprobs and elementwise_loss."""
        results = []
        for i, feature in enumerate(inputs):
            # Ensure 1D shape and correct device to avoid dimension mismatch and device errors
            labels = feature.loss_fn_inputs['target_tokens'].to_torch(
            ).long().view(-1).to(logits.device)  # shape (seq_len,)
            weights = feature.loss_fn_inputs['weights'].to_torch(
            ).view(-1).to(logits.device)  # shape (seq_len,)

            # Slice logits to match the sequence length of labels
            # Labels are assumed to be already shifted/aligned with logits
            seq_len = labels.numel()
            if i < logits.shape[0]:
                feature_logits = logits[i, :seq_len, :]

                # Calculate log probs for all labels
                # Apply log_softmax to convert raw logits to log-probabilities
                feature_log_probs = torch.log_softmax(feature_logits, dim=-1)
                token_log_probs = feature_log_probs.gather(
                    dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

                # elementwise_loss: positive NLL loss (0.0 where masked)
                elementwise_loss = -token_log_probs * weights

                results.append({
                    'logprobs': types.TensorData.from_torch(token_log_probs),
                    'elementwise_loss': types.TensorData.from_torch(elementwise_loss)
                })
            else:
                # Handle case where batch index exceeds logits batch size
                results.append({
                    'logprobs': None,
                    'elementwise_loss': None
                })
        return results
