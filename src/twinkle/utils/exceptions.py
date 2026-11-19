# Copyright (c) ModelScope Contributors. All rights reserved.


class FullWeightLoadError(RuntimeError):
    """A checkpoint failed after the engine began replacing base weights."""
