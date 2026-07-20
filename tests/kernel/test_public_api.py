def test_public_exports_exactly_four_symbols():
    import twinkle.kernel as k
    assert sorted(k.__all__) == ['hub', 'kernelize', 'liger_builtin', 'npu_builtin']
    assert callable(k.kernelize)
    assert callable(k.npu_builtin)
    assert callable(k.liger_builtin)
    assert callable(k.hub)


def test_no_legacy_symbols():
    """Legacy registrar / patch helpers must be gone."""
    import twinkle.kernel as k
    legacy = [
        'kernelize_model', 'register_layer_kernel', 'register_function_kernel',
        'register_kernels', 'register_external_layer', 'apply_npu_patch',
        'apply_npu_fused_ops', 'apply_function_kernel', 'apply_layer_kernel',
        'register_layer_batch', 'register_npu_fused_function_kernels',
        'get_global_layer_registry', 'get_global_function_registry',
        'get_global_external_layer_registry', 'LayerRegistry',
        'ExternalLayerRegistry', 'FunctionRegistry',
    ]
    for name in legacy:
        assert not hasattr(k, name), f'unexpected legacy symbol: {name}'