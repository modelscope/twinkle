"""Default rule-based score dispatch."""
import warnings
from typing import Any, Callable, Dict


_SCORERS: Dict[str, Callable] = {}


def register_score(data_source: str):
    def decorator(func):
        _SCORERS[data_source.lower()] = func
        return func
    return decorator


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict | None = None,
                  unknown_rewards: str = "warn"):
    scorer = _SCORERS.get((data_source or "").lower())
    if scorer is None:
        message = f"no default reward scorer registered for data_source={data_source!r}"
        if unknown_rewards == "raise":
            raise ValueError(message)
        if unknown_rewards == "warn":
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        return 0.0, {"warning": message}
    return scorer(solution_str, ground_truth, extra_info or {})
