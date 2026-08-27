# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import Advantage


class GAEAdvantage(Advantage):
    """Generalized Advantage Estimation over one response (per-token, value-based).

    Unlike the group-relative estimators (:class:`GRPOAdvantage` / :class:`RLOOAdvantage`), which
    baseline a scalar reward against its prompt group, GAE bootstraps a learned per-token value
    ``V(s_t)``. Walking backwards over a single response (terminal bootstrap ``V_T = 0`` past the last
    token)::

        delta_t = r_t + gamma * V_{t+1} - V_t
        A_t     = delta_t + gamma * lam * A_{t+1}

    and the critic's return target is ``A_t + V_t``. This is PPO's advantage; because it needs a value
    function it returns BOTH the advantages and the returns, rather than advantages alone.
    """

    def __call__(self,
                 rewards: list,
                 values: list = None,
                 gamma: float = 1.0,
                 lam: float = 0.95,
                 **kwargs) -> tuple:
        """Compute per-token advantages and returns for one response.

        Args:
            rewards: per-response-token rewards, length ``R``.
            values: per-response-token value estimates ``V(s_t)``, length ``R``.
            gamma: discount factor.
            lam: GAE lambda (0 -> one-step TD error; 1 -> full Monte-Carlo credit).

        Returns:
            ``(advantages, returns)``, both length ``R``.
        """
        if values is None:
            raise ValueError('GAEAdvantage requires per-token `values` (a learned critic estimate).')
        n = len(rewards)
        advantages = [0.0] * n
        last = 0.0
        for t in range(n - 1, -1, -1):
            next_value = values[t + 1] if t + 1 < n else 0.0
            delta = rewards[t] + gamma * next_value - values[t]
            last = delta + gamma * lam * last
            advantages[t] = last
        returns = [advantages[t] + values[t] for t in range(n)]
        return advantages, returns
