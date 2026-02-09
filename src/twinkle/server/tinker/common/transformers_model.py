import torch
import numpy as np
from tinker import types
from typing import List
from twinkle.template import Template
from twinkle.model import MultiLoraTransformersModel
from twinkle import remote_class, remote_function
from .datum import datum_to_input_feature, extract_rl_feature
from .io_utils import create_checkpoint_manager


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


@remote_class()
class TwinkleCompatTransformersModel(MultiLoraTransformersModel):
    """
    Compatibility wrapper around :class:`MultiLoraTransformersModel` for Twinkle/Tinker.

    This class adapts the core `MultiLoraTransformersModel` API to the data types and
    remote-call semantics used by Twinkle:

    * Inputs to :meth:`forward` and :meth:`forward_only` are provided as
      ``List[types.Datum]`` and are converted to the underlying model's
      ``InputFeature`` format via :func:`datum_to_input_feature`.
    * The outputs of :meth:`forward` and :meth:`forward_only` are not the raw
      transformer outputs; instead they are a list of dictionaries, one per
      input example, containing:

        - ``"logprobs"``: token-level log-probabilities as ``types.TensorData``.
        - ``"elementwise_loss"``: per-token (masked) NLL loss as ``types.TensorData``.

      These are derived from the underlying logits by applying ``log_softmax``
      and slicing to the label sequence length.
    * :meth:`calculate_loss` returns a Python scalar (via ``tensor.item()``)
      and is exposed as a remote function with ``collect='sum'``, so the
      distributed caller receives an aggregated scalar loss instead of a
      tensor object.
    * :meth:`step` accepts optimizer hyperparameters as :class:`types.AdamParams`,
      performs optional gradient clipping, translates them into the optimizer
      configuration expected by the base class, invokes the base ``step``
      implementation, and finally zeros gradients.

    Overall, this wrapper ensures that callers using Twinkle's higher-level
    ``Datum``/``TensorData`` abstractions and remote functions can interact
    with a ``MultiLoraTransformersModel`` instance without needing to know its
    internal input feature schema, output structure, or optimizer API.
    """

    @remote_function(dispatch='slice_dp', collect='flatten')
    def forward_only(self, *, inputs: List[types.Datum], **kwargs):
        # Get template for input processing
        template = self.get_template(**kwargs)
        # Convert Datum to InputFeature
        input_features = datum_to_input_feature(inputs, template)
        outputs = super().forward_only(inputs=input_features, **kwargs)
        # shape (batch_size, seq_len, vocab_size)
        logits = outputs['logits'].detach().cpu()
        results = self._get_forward_output(inputs, logits)
        return results

    @remote_function(dispatch='slice_dp', collect=_collect_forward_backward_results)
    def forward_backward(self, *, inputs: List[types.Datum], adapter_name: str, loss_fn: str, **kwargs):
        # Set loss first based on loss_fn
        if loss_fn == 'cross_entropy':
            super().set_loss('CrossEntropyLoss',
                             adapter_name=adapter_name)
        elif loss_fn == 'importance_sampling':
            super().set_loss('GRPOLoss',
                             adapter_name=adapter_name,
                             epsilon=0.2,  # Default GRPO epsilon
                             beta=0.0)     # No KL penalty by default
        else:
            raise ValueError(
                f'Unsupported loss function {loss_fn}')
        # Get template for input processing
        template = self.get_template(adapter_name)
        
        # Convert Datum to InputFeature
        input_features = datum_to_input_feature(inputs, template)

        # Forward pass
        outputs = super().forward(inputs=input_features, adapter_name=adapter_name, **kwargs)

        # Calculate loss with extra parameters
        # Extract old_logps and advantages using common utility
        loss_values = extract_rl_feature(inputs)
        loss_kwargs = kwargs.copy()
        loss_kwargs.update(loss_values)
        loss = super().calculate_loss(adapter_name=adapter_name, **loss_kwargs)

        # Backward pass
        super().backward(adapter_name=adapter_name, **kwargs)

        # shape (batch_size, seq_len, vocab_size)
        logits = outputs['logits'].detach().cpu()
        results = self._get_forward_output(inputs, logits)
        return [results, loss]

    @remote_function()
    def step(self, *, adam_params: types.AdamParams, **kwargs):
        # Gradient clipping
        grad_clip_norm = adam_params.grad_clip_norm
        if grad_clip_norm > 0.0:
            self.clip_grad_norm(max_grad_norm=grad_clip_norm,
                                norm_type=2, **kwargs)
        # Optimizer step
        optim_params = {
            'lr': adam_params.learning_rate,
            'eps': adam_params.eps,
            'betas': (adam_params.beta1, adam_params.beta2),
            'weight_decay': adam_params.weight_decay,
        }
        super().step(optim_params=optim_params, **kwargs)
        # Zero gradients
        super().zero_grad(**kwargs)

    @remote_function()
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
        return results
