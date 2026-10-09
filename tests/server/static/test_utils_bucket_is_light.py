# Copyright (c) ModelScope Contributors. All rights reserved.
"""The ``twinkle.server.utils`` bucket must re-export only the two
dependency-light helpers, not the queue / session-resource machinery.

The original intent was to assert ``import twinkle.server.utils`` pulls no
OpenTelemetry. That side-effect is dominated by the parent package's eager
``twinkle.server.__init__`` -> ``launcher`` -> ``application_spec`` -> ``task_queue.config``
chain (which triggers ``task_queue/__init__`` -> ``mixin`` -> telemetry) and is out of
this Requirement's scope, so we assert the directly-controlled property instead: the
bucket's re-export surface. Re-exporting the mixins is what used to make every one of the
five light callers drag in the OpenTelemetry SDK.
"""
import twinkle.server.utils as bucket

# The two genuinely dependency-free helpers the five call sites actually use.
_LIGHT_EXPORTS = ('get_template_for_model', 'wrap_builder_with_device_group_env')
# The heavy machinery that must no longer be re-exported through the bucket.
_HEAVY_EXPORTS = (
    'TaskQueueMixin',
    'TaskQueueConfig',
    'RateLimiter',
    'QueueState',
    'TaskStatus',
    'SessionResourceMixin',
    'AdapterManagerMixin',
    'ProcessorManagerMixin',
)


def test_bucket_exports_only_light_helpers():
    for name in _LIGHT_EXPORTS:
        assert hasattr(bucket, name), f'{name} should be re-exported by the utils bucket'
    leaked = [name for name in _HEAVY_EXPORTS if hasattr(bucket, name)]
    assert not leaked, f'utils bucket must not re-export heavy machinery: {leaked}'
