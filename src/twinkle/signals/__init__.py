# Copyright (c) ModelScope Contributors. All rights reserved.
"""Per-sample RL signal operations: response-token rows, KL estimates, reductions, blends.

Each class is one stateless operation whose knobs are call arguments. Their inputs are per-token
rows, so a caller that only holds references builds them with ``remote_group`` and the rows are
read where they live; built without one they run in the caller's process.
"""
from .combine import SignalCombine
from .kl import KLEstimator
from .reduce import Reduce
from .response import response_positions, response_token_scores

__all__ = [
    'KLEstimator',
    'Reduce',
    'SignalCombine',
    'response_positions',
    'response_token_scores',
]
