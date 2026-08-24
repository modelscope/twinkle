def test_public_exports_exactly_four_symbols():
    import twinkle.kernel as k
    assert sorted(k.__all__) == ['DEFAULT_KERNEL_CONFIG', 'KernelChoice', 'hub', 'kernelize']
    assert callable(k.kernelize)
    assert callable(k.hub)
    assert isinstance(k.DEFAULT_KERNEL_CONFIG, dict)
    assert k.DEFAULT_KERNEL_CONFIG  # non-empty built-in default mapping


def test_no_legacy_symbols():
    """Legacy registrar / patch helpers and the retired builtin bundles must be gone."""
    import twinkle.kernel as k
    legacy = [
        'kernelize_model', 'register_layer_kernel', 'register_function_kernel',
        'register_kernels', 'register_external_layer', 'apply_npu_patch',
        'apply_npu_fused_ops', 'apply_function_kernel', 'apply_layer_kernel',
        'register_layer_batch', 'register_npu_fused_function_kernels',
        'get_global_layer_registry', 'get_global_function_registry',
        'get_global_external_layer_registry', 'LayerRegistry',
        'ExternalLayerRegistry', 'FunctionRegistry',
        'npu_builtin', 'liger_builtin',
    ]
    for name in legacy:
        assert not hasattr(k, name), f'unexpected legacy symbol: {name}'