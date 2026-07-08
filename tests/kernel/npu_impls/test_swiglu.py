def test_swiglu_imports():
    from twinkle.kernel.npu_impls.swiglu import npu_swiglu_forward
    assert callable(npu_swiglu_forward)


def test_swiglu_signature():
    import inspect

    from twinkle.kernel.npu_impls.swiglu import npu_swiglu_forward

    params = list(inspect.signature(npu_swiglu_forward).parameters)
    assert params == ['self', 'hidden_state']