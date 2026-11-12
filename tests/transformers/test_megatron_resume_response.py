from types import SimpleNamespace

import pytest
import ray

import twinkle.infra as infra
from twinkle.model.megatron.megatron import MegatronModel
from twinkle.model.megatron.multi_lora_megatron import MultiLoraMegatronModel
from twinkle.protocol.types.model import TrainingProgressResponse


@ray.remote
class ProgressWorker:
    def __init__(self):
        self.calls = 0

    def resume_from_checkpoint(self, path, **kwargs):
        self.calls += 1
        return {'cur_step': 4, 'consumed_train_samples': 16, 'gradient_accumulation_steps': 1}

    def call_count(self):
        return self.calls


@pytest.fixture(scope='module')
def workers():
    owns_runtime = not ray.is_initialized()
    if owns_runtime:
        ray.init(address='local', num_cpus=2, include_dashboard=False, log_to_driver=False)
    actors = [ProgressWorker.remote() for _ in range(2)]
    try:
        yield actors
    finally:
        for actor in actors:
            ray.kill(actor, no_restart=True)
        if owns_runtime:
            ray.shutdown()


@pytest.mark.parametrize('model_class', [MegatronModel, MultiLoraMegatronModel])
def test_megatron_resume_returns_one_typed_progress_after_calling_all_workers(workers, monkeypatch, model_class):
    monkeypatch.setattr(infra, '_mode', 'ray')
    monkeypatch.setattr(infra, '_lazy_collect', False)
    before = ray.get([worker.call_count.remote() for worker in workers])
    driver = SimpleNamespace(_actors=workers, _max_concurrency=None, device_mesh=None)
    result = model_class.resume_from_checkpoint(driver, 'checkpoint')
    progress = TrainingProgressResponse.model_validate({'result': result})
    assert progress.result['cur_step'] == 4
    assert ray.get([worker.call_count.remote() for worker in workers]) == [count + 1 for count in before]
