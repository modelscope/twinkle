from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RewardLoopArgs:
    num_workers: int = 8
    custom_reward_function_path: Optional[str] = None
    custom_reward_function_name: str = "compute_score"
    manager_name: str = "naive"
    manager_source: str = "register"
    manager_module_path: Optional[str] = None
    manager_module_name: str = "RewardLoopManager"
    unknown_rewards: str = "warn"
    mode: str = "async"
    backlog: int = 2
    on_backlog_full: str = "block"
    on_error: str = "raise"
    max_rpm: Optional[int] = None
    max_tpm: Optional[int] = None
    max_concurrent: int = 1
    timeout: float = 300.0
    reward_worker_executors: Optional[int] = None
    reward_kwargs: dict = field(default_factory=dict)
