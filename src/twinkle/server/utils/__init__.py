# Copyright (c) ModelScope Contributors. All rights reserved.
"""Stateless server-side helpers.

Deliberately re-exports only ``device_utils`` and ``template_utils``: the five call sites
that import through this bucket all want just those two (``get_template_for_model``, a
52-line string map, and ``wrap_builder_with_device_group_env``), while the code that
actually needs the queue / session-resource machinery imports the full path. Re-exporting
the mixins pulled the whole OpenTelemetry SDK into every one of those five importers
(``task_queue.mixin`` -> ``telemetry`` -> the OpenTelemetry SDK + OTLP exporter).
"""
from .device_utils import auto_fill_device_group_visible_devices, wrap_builder_with_device_group_env
from .template_utils import get_template_for_model
