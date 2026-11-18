from types import SimpleNamespace

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
    assert manager.adapter_path(0) == first
    assert manager.adapter_path(1) == second
    assert manager.adapter_path(99) is None
    assert manager.version() == 2
