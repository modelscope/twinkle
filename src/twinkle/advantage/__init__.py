# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import Advantage, weighted_score
from .estimate import ESTIMATORS, SCALINGS, compute_advantages
from .gae import GAEAdvantage
from .gpg import GPGAdvantage
from .grpo import GRPOAdvantage
from .grpo_passk import GRPOPassKAdvantage
from .opo import OPOAdvantage
from .reinforce_plus_plus import ReinforcePlusPlusAdvantage
from .reinforce_pp_baseline import ReinforcePPBaselineAdvantage
from .remax import ReMaxAdvantage
from .rloo import RLOOAdvantage
from .teacher_signal import TeacherReweight
# The reward fold and the NaN-aware std are public because a caller that decides whether a group is
# worth training on has to reach the same verdict as the estimator that will train on it: folding or
# spreading the scores differently admits groups whose advantages are all zero.
from ._utils import nanstd, reduce_rewards

__all__ = [
    'Advantage',
    'ESTIMATORS',
    'GAEAdvantage',
    'GPGAdvantage',
    'GRPOAdvantage',
    'GRPOPassKAdvantage',
    'OPOAdvantage',
    'ReinforcePlusPlusAdvantage',
    'ReinforcePPBaselineAdvantage',
    'ReMaxAdvantage',
    'RLOOAdvantage',
    'SCALINGS',
    'TeacherReweight',
    'compute_advantages',
    'nanstd',
    'reduce_rewards',
    'weighted_score',
]
