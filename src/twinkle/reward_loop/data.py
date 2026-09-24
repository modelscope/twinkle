"""Framework-independent reward loop data contracts."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RewardItem:
    item_id: str
    data_source: str = ""
    solution_str: str = ""
    ground_truth: str = ""
    extra_info: Dict[str, Any] = field(default_factory=dict)
    response_ids: Any = None
    attention_mask: Any = None
    raw_prompt: Optional[str] = None


@dataclass
class RewardResult:
    item_id: str
    reward_score: float
    reward_extra_info: Dict[str, Any] = field(default_factory=dict)


def split_items(items: List[RewardItem], num_workers: int) -> List[List[RewardItem]]:
    if num_workers < 1:
        raise ValueError("num_workers must be positive")
    chunks = [[] for _ in range(min(num_workers, len(items)))]
    for index, item in enumerate(items):
        chunks[index % len(chunks)].append(item)
    return chunks


def reorder_by_id(results: List[RewardResult]) -> Dict[str, RewardResult]:
    ordered: Dict[str, RewardResult] = {}
    for result in results:
        if result.item_id in ordered:
            raise ValueError(f"duplicate reward item_id: {result.item_id}")
        ordered[result.item_id] = result
    return ordered


def assemble_scores(results: List[RewardResult], items: List[RewardItem], mode: str = "scalar") -> List[float]:
    if mode != "scalar":
        raise NotImplementedError("token reward assembly is reserved for a future release")
    by_id = reorder_by_id(results)
    missing = [item.item_id for item in items if item.item_id not in by_id]
    if missing:
        raise KeyError(f"missing reward results: {missing}")
    return [float(by_id[item.item_id].reward_score) for item in items]
