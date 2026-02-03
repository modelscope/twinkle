import torch
from tinker import types
from typing import List
from twinkle.model import MultiLoraTransformersModel
from twinkle import remote_class, remote_function
from .datum import datum_to_input_feature
from .io_utils import create_checkpoint_manager

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
    def forward(self, *, inputs: List[types.Datum], **kwargs):
        # Convert Datum to InputFeature
        input_features = [datum_to_input_feature(datum) for datum in inputs]
       
        outputs = super().forward(inputs=input_features, **kwargs)
        logits = outputs['logits'].detach().cpu()  # shape (batch_size, seq_len, vocab_size)
        results = self._get_forward_output(inputs, logits)
        return results

    @remote_function(dispatch='slice_dp', collect='flatten')
    def forward_only(self, *, inputs: List[types.Datum], **kwargs):
        # Convert Datum to InputFeature
        input_features = [datum_to_input_feature(datum) for datum in inputs]

        outputs = super().forward_only(inputs=input_features, **kwargs)
        logits = outputs['logits'].detach().cpu()  # shape (batch_size, seq_len, vocab_size)
        results = self._get_forward_output(inputs, logits)
        return results

    @remote_function(collect='mean')
    def calculate_loss(self, **kwargs):
        loss = super().calculate_loss(**kwargs)
        return loss

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
        self.zero_grad(**kwargs)

    @remote_function()
    def load(self, checkpoint_dir: str, **kwargs):
        """
        Load checkpoint with token-based isolation support.
        
        Args:
            checkpoint_dir: The twinkle:// path to the checkpoint
            **kwargs: Additional keyword arguments including optional 'token'
        """
        # Extract token from kwargs if provided (for user isolation)
        token = kwargs.pop('token', None)
        if not token:
            raise ValueError("Token is required for loading checkpoints")
        
        # Create checkpoint manager with the token
        checkpoint_manager = create_checkpoint_manager(token)
        
        # handle twinkle checkpoint format
        tinker_path = checkpoint_manager.parse_tinker_path(checkpoint_dir)
        if not tinker_path:
            raise ValueError(f"Invalid twinkle checkpoint path: {checkpoint_dir}")
        
        # check adapter files with token-based path
        weight_path = checkpoint_manager.get_ckpt_dir(
            tinker_path.training_run_id, 
            tinker_path.checkpoint_id
        )
        if not weight_path or not weight_path.exists():
            raise ValueError(f"Checkpoint not found at {weight_path}")
        
        if (weight_path / 'adapter_config.json').exists():
            return super().load(name=weight_path.name, output_dir=weight_path.parent, **kwargs)
        elif (weight_path / tinker_path.training_run_id / 'adapter_config.json').exists():
            return super().load(name=weight_path.name, output_dir=weight_path.parent, **kwargs)
        else:
            raise ValueError(f"Adapter files not found in {weight_path}")

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
