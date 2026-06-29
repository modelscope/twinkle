def test_attention_imports():
    from twinkle.kernel.npu_impls.attention import npu_sdpa_attention_forward
    assert callable(npu_sdpa_attention_forward)


def test_attention_signature():
    import inspect

    from twinkle.kernel.npu_impls.attention import npu_sdpa_attention_forward

    sig = inspect.signature(npu_sdpa_attention_forward)
    params = list(sig.parameters)
    assert params[:5] == ['module', 'query', 'key', 'value', 'attention_mask']
    assert sig.parameters['dropout'].default == 0.0
    assert sig.parameters['scaling'].default is None
    assert sig.parameters['is_causal'].default is None