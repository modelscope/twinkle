# Copyright (c) ModelScope Contributors. All rights reserved.
"""Client-internal helpers shared across the twinkle_client subpackages.

Regular package (carries this ``__init__``) so the modules here -- ``component_rpc``,
``serialize``, ``json_utils`` -- are included by ``setuptools.packages.find`` in a
built wheel; a namespace-only directory would be dropped from the distribution.
"""
