from tinker import types
from typing import List

from twinkle import remote_class, remote_function
from twinkle.model import MultiLoraTransformersModel
from .compat_base import TwinkleCompatModelBase, clean_metrics, collect_forward_backward_results
from .datum import datum_to_input_feature, extract_rl_feature
from .io_utils import create_checkpoint_manager


@remote_class()
class TwinkleCompatTransformersModel(MultiLoraTransformersModel, TwinkleCompatModelBase):
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

    @remote_function(dispatch='slice_dp', collect=collect_forward_backward_results)
    def forward_backward(self, *, inputs: List[types.Datum], adapter_name: str, loss_fn: str, **kwargs):
        # Set loss first based on loss_fn
        if loss_fn == 'cross_entropy':
            super().set_loss('CrossEntropyLoss', adapter_name=adapter_name)
        elif loss_fn == 'importance_sampling':
            super().set_loss(
                'GRPOLoss',
                adapter_name=adapter_name,
                epsilon=0.2,  # Default GRPO epsilon
                beta=0.0)  # No KL penalty by default
        else:
            super().set_loss('CrossEntropyLoss', adapter_name=adapter_name)
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
        logits = outputs['logits'].detach()
        results = self._get_forward_output(inputs, logits)
        return [results, loss]

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
        super().zero_grad(**kwargs)

    @remote_function(collect='first', lazy_collect=False)
    def calculate_metric(self, is_training, **kwargs):
        metric = super().calculate_metric(is_training, **kwargs)
        return clean_metrics(metric)

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
            raise ValueError('Token is required for loading checkpoints')

        # Create checkpoint manager with the token
        checkpoint_manager = create_checkpoint_manager(token)

        # Use resolve_load_path to handle path resolution
        resolved = checkpoint_manager.resolve_load_path(checkpoint_dir)

        if resolved.is_twinkle_path:
            # Load from twinkle checkpoint
            return super().load(name=resolved.checkpoint_name, output_dir=resolved.checkpoint_dir, **kwargs)
        else:
            # Load from hub
            return super().load(name=resolved.checkpoint_name, **kwargs)
