# Copyright (c) ModelScope Contributors. All rights reserved.
"""Client-internal utilities.

Regular package (carries this ``__init__``) so ``patch_tinker`` is included by
``setuptools.packages.find`` in a built wheel; a namespace-only directory would be
dropped from the distribution.
"""
