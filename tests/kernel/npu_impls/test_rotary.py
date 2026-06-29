def test_rotary_imports():
    from twinkle.kernel.npu_impls.rotary import (
        npu_apply_multimodal_rotary_pos_emb,
        npu_apply_rotary_pos_emb,
    )
    assert callable(npu_apply_rotary_pos_emb)
    assert callable(npu_apply_multimodal_rotary_pos_emb)


def test_rotary_signature_compat():
    """Signature must match HF apply_rotary_pos_emb so setattr swap is safe."""
    import inspect

    from twinkle.kernel.npu_impls.rotary import npu_apply_rotary_pos_emb

    sig = inspect.signature(npu_apply_rotary_pos_emb)
    params = list(sig.parameters)
    assert params[:4] == ['q', 'k', 'cos', 'sin']
    # position_ids and unsqueeze_dim must be optional
    assert sig.parameters['position_ids'].default is None
    assert sig.parameters['unsqueeze_dim'].default == 1