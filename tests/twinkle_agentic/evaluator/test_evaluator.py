import pytest

from twinkle_agentic.evaluator import Evaluator
from twinkle_agentic.evaluator._contracts import EvaluatorConfigError

from .conftest import RecordingAPI, RecordingSampler


def test_constructor_validates_and_copies_inputs():
    sampler = RecordingSampler()
    config = {'limit': 1}
    kwargs = {'adapter_name': 'a'}
    evaluator = Evaluator(datasets=['gsm8k'], sampler=sampler, task_config=config, sampler_kwargs=kwargs)
    config['limit'] = 2
    kwargs['adapter_name'] = 'changed'
    assert evaluator._task_config['limit'] == 1
    assert evaluator._sampler_kwargs['adapter_name'] == 'a'


@pytest.mark.parametrize('sampler,api', [(None, None), (RecordingSampler(), RecordingAPI())])
def test_requires_exactly_one_backend(sampler, api):
    with pytest.raises(EvaluatorConfigError, match='exactly one'):
        Evaluator(datasets=['x'], sampler=sampler, api=api)


@pytest.mark.parametrize('datasets', [[], [''], 'gsm8k'])
def test_datasets_are_validated(datasets):
    with pytest.raises(EvaluatorConfigError, match='datasets'):
        Evaluator(datasets=datasets, sampler=RecordingSampler())


@pytest.mark.parametrize('key', ['model', 'model_id', 'datasets', 'eval_type', 'eval_backend', 'model_task', 'api_url', 'api_key', 'model_args'])
def test_managed_config_keys_are_rejected(key):
    with pytest.raises(EvaluatorConfigError, match=key):
        Evaluator(datasets=['x'], sampler=RecordingSampler(), task_config={key: 'value'})


def test_api_mode_rejects_sampler_only_options():
    with pytest.raises(EvaluatorConfigError, match='sampler'):
        Evaluator(datasets=['x'], api=RecordingAPI(), sampler_batch_size=2)


def test_single_use_after_success(monkeypatch, recording_api):
    sentinel = {'x': object()}
    import evalscope.run
    def run_task(config):
        config.work_dir = 'outputs/resolved'
        return sentinel
    monkeypatch.setattr(evalscope.run, 'run_task', run_task)
    evaluator = Evaluator(datasets=['x'], api=recording_api)
    assert evaluator.output_dir is None
    assert evaluator.run() is sentinel
    assert evaluator.resolved_task_config.model is not None
    assert evaluator.output_dir == 'outputs/resolved'
    with pytest.raises(RuntimeError, match='single-use'):
        evaluator.run()


def test_single_use_after_failure(monkeypatch, recording_api):
    import evalscope.run
    monkeypatch.setattr(evalscope.run, 'run_task', lambda config: (_ for _ in ()).throw(ValueError('boom')))
    evaluator = Evaluator(datasets=['x'], api=recording_api)
    with pytest.raises(ValueError, match='boom'):
        evaluator.run()
    with pytest.raises(RuntimeError, match='single-use'):
        evaluator.run()


@pytest.mark.parametrize('backend', ['api', 'sampler'])
def test_offline_native_evalscope_run(tmp_path, recording_api, recording_sampler, backend):
    dataset = tmp_path / 'questions.jsonl'
    dataset.write_text('{"question": "Say ok", "answer": "ok"}\n', encoding='utf-8')
    kwargs = {'api': recording_api} if backend == 'api' else {'sampler': recording_sampler}
    evaluator = Evaluator(
        datasets=['general_qa'],
        task_config={
            'dataset_args': {'general_qa': {'local_path': str(dataset)}},
            'dataset_hub': 'Local',
            'work_dir': str(tmp_path / 'outputs'),
            'no_timestamp': True,
            'generation_config': {'temperature': 0.0},
        },
        **kwargs,
    )
    reports = evaluator.run()
    assert reports
    assert evaluator.output_dir == str(tmp_path / 'outputs')
    if backend == 'sampler':
        assert recording_sampler.calls
