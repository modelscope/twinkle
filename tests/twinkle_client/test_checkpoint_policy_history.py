from types import SimpleNamespace

from twinkle_agentic.async_rl import StreamingDriver
from twinkle_client.rollout import ClientCheckpointManager


class _Model:
    def __init__(self):
        self.saved = []

    def save(self, name, **_kwargs):
        self.saved.append(name)
        return SimpleNamespace(twinkle_path=f"/policies/{name}.adapter")


class _Sampler:
    def __init__(self):
        self.adapter_path = None


def test_checkpoint_manager_retains_each_policy_adapter_for_resume():
    model = _Model()
    sampler = _Sampler()
    manager = ClientCheckpointManager(model, sampler, name_prefix="durable")

    manager.sync_weights()
    first = sampler.adapter_path
    manager.sync_weights()
    second = sampler.adapter_path

    assert first == "/policies/durable-0.adapter"
    assert second == "/policies/durable-1.adapter"
    assert manager.adapter_path(0) is None
    assert manager.adapter_path(1) == first
    assert manager.adapter_path(2) == second
    assert manager.adapter_path(99) is None
    assert manager.version() == 2


def test_checkpoint_manager_policy_versions_match_driver_admission():
    model = _Model()
    sampler = _Sampler()
    manager = ClientCheckpointManager(model, sampler, name_prefix="durable")
    manager.sync_weights()

    class _Rollout:
        def step(self, trajectory, **_kwargs):
            self.trajectory = trajectory
            return SimpleNamespace(add_done_callback=lambda _callback: None)

    rollout = _Rollout()
    driver = StreamingDriver([{"messages": []}], rollout, batch_size=1, num_generations=1, checkpoint_manager=manager)
    driver._version = manager.version()
    driver._admit()

    group = driver._groups["0"]
    assert group.version == 1
    assert group.adapter_path == sampler.adapter_path
    assert rollout.trajectory["adapter_path"] == sampler.adapter_path
