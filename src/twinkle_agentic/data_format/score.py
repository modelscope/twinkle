from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class RoundContext:
    """Per-round payload passed to scorers."""
    row_idx: int
    rnd_idx: int
    asst_idx: int
    row: Dict[str, Any]
    intent: Optional[str]
    messages: List[Dict[str, Any]]
    context_messages: List[Dict[str, Any]]
    cond_ids: List[int]
    n_prompt: int
    asst_ids: List[int]
    asst_text: str
    user_prompt: str
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreResult:
    score: Optional[float] = None
    passed: bool = True
    extras: Dict[str, Any] = field(default_factory=dict)


class Scorer(Protocol):
    name: str
    requires_logprobs: bool

    def score(self, contexts: List[RoundContext]) -> List[ScoreResult]:
        ...
