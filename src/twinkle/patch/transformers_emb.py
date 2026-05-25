def get_lm_head_model(model, model_meta=None, lm_heads=None):
    if isinstance(model, PeftModel):
        model = model.model
    model_meta = model_meta or model.model_meta
    if lm_heads is None:
        lm_heads = ['lm_head', 'output', 'embed_out', 'output_layer']
    llm_prefix_list = getattr(model_meta.model_arch, 'language_model', None)
    prefix_list = []
    if llm_prefix_list:
        prefix_list = llm_prefix_list[0].split('.')

    current_model = model
    for prefix in prefix_list:
        current_model = getattr(current_model, prefix)
        for lm_head in lm_heads:
            if hasattr(current_model, lm_head):
                return current_model
    return model

def get_last_valid_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Get the last valid (non-padding) token position indices for each sample.

    This function correctly handles sequences with different padding directions (left/right/none)
    within the same batch by computing the last valid index for each sequence individually.

    Args:
        attention_mask: Attention mask [batch_size, seq_len] where 1=valid, 0=padding

    Returns:
        torch.Tensor: Indices of last valid positions [batch_size]

    Examples:
        >>> # Right padding
        >>> attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
        >>> get_last_valid_indices(attention_mask)
        tensor([2, 3])

        >>> # Left padding
        >>> attention_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
        >>> get_last_valid_indices(attention_mask)
        tensor([4, 4])
    """
    seq_len = attention_mask.shape[1]

    # Flip the mask horizontally to bring the last elements to the front.
    # `argmax` will then find the index of the first '1', which corresponds to the last valid token.
    last_valid_indices = torch.fliplr(attention_mask).argmax(dim=1)

    # Convert the index from the right-to-left frame to the original left-to-right frame.
    indices = seq_len - 1 - last_valid_indices

    return indices

def patch_output_normalizer(module: torch.nn.Module, model_meta):

    def lm_head_forward(self, hidden_states):
        return hidden_states

    lm_heads = ['lm_head', 'output', 'embed_out', 'output_layer']
    lm_head_model = get_lm_head_model(module, model_meta=model_meta, lm_heads=lm_heads)

    found = False
    for lm_head in lm_heads:
        if hasattr(lm_head_model, lm_head):
            getattr(lm_head_model, lm_head).forward = MethodType(lm_head_forward, getattr(lm_head_model, lm_head))
            found = True
            break

    assert found, 'Cannot find the proper lm_head name'

    def _output_embedding_hook(module, args, kwargs, output):
        attention_mask = kwargs.get('attention_mask', None)
        hidden_states = output.logits
        sequence_lengths = -1 if attention_mask is None else get_last_valid_indices(attention_mask)
        embeddings = hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), sequence_lengths]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return {
            'last_hidden_state': embeddings.contiguous(),
        }

    lm_head_model.register_forward_hook(_output_embedding_hook, with_kwargs=True)