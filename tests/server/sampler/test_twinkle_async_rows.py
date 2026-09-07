from __future__ import annotations

import pytest
from fastapi import FastAPI
from starlette.requests import Request

import twinkle_client.types as types
from twinkle.data_format import SampledSequence, SampleResponse
from twinkle.server.sampler.twinkle_handlers import (
    _register_twinkle_sampler_routes,
    _build_rollout_rows_and_tags,
)


def _response(tokens: list[int]) -> types.SampleResponseModel:
    return types.SampleResponseModel(
        sequences=[
            types.SampledSequenceModel(
                stop_reason='stop',
                tokens=[token],
                logprobs=[[(token, -0.1)]],
                new_input_feature={'input_ids': [token], 'labels': [token]},
            )
            for token in tokens
        ],
        prompt_logprobs=[-0.2],
    )


def test_async_sampler_flattens_generations_to_tagged_tq_rows() -> None:
    rows, tags = _build_rollout_rows_and_tags(
        [_response([10, 11]), _response([20, 21])],
        group_ids=['group-a', 'group-b'],
        policy_version=7,
        adapter_uri='twinkle://policy-7',
    )

    assert [row['tokens'] for row in rows] == [[10], [11], [20], [21]]
    assert [row['sampled_logprobs'] for row in rows] == [[-0.1]] * 4
    assert [row['train_input']['input_ids'] for row in rows] == [[10], [11], [20], [21]]
    assert all('new_input_feature' not in row for row in rows)
    assert [(tag['group_id'], tag['generation_idx']) for tag in tags] == [
        ('group-a', 0),
        ('group-a', 1),
        ('group-b', 0),
        ('group-b', 1),
    ]
    assert {tag['rollout_policy_version'] for tag in tags} == {7}
    assert {tag['rollout_adapter_uri'] for tag in tags} == {'twinkle://policy-7'}


def test_async_sampler_rejects_group_id_count_mismatch() -> None:
    with pytest.raises(ValueError, match='group_ids contains 1 values for 2'):
        _build_rollout_rows_and_tags(
            [_response([10]), _response([20])],
            group_ids=['only-one'],
            policy_version=0,
            adapter_uri=None,
        )


class _SamplerManagement:

    def __init__(self):
        self.sampler = self
        self.data_plane = self
        self.enabled = True
        self.scheduled = []
        self.put_rows = None

    async def _on_request_start(self, _request):
        return 'token'

    async def schedule_task_and_wait(self, task, **kwargs):
        self.scheduled.append(kwargs)
        return await task()

    def submit_generation(self, submission_id, inputs, params, **kwargs):
        self.submission_id = submission_id
        self.inputs = inputs
        self.params = params
        self.generation_kwargs = kwargs

    def get_generation_status(self, submission_id):
        assert submission_id == self.submission_id
        return {'status': 'completed'}

    def collect_generation(self, submission_id):
        assert submission_id == self.submission_id
        return [SampleResponse(sequences=[SampledSequence(
            stop_reason='stop',
            tokens=[7],
            logprobs=[[(7, -0.25)]],
            decoded='answer',
            new_input_feature={'input_ids': [1, 7], 'labels': [-100, 7]},
        )])]

    def cancel_generation(self, _submission_id):
        raise AssertionError('completed generation must not be cancelled')

    async def put(self, rows, *, kind, tags):
        self.put_rows = rows
        self.put_tags = tags
        return types.DataRef(ref_id='rollout-ref', size=len(rows), fields=list(rows[0]), kind=kind)


@pytest.mark.asyncio
async def test_sample_to_data_plane_returns_ref_after_short_admission() -> None:
    management = _SamplerManagement()
    app = FastAPI()
    _register_twinkle_sampler_routes(app, lambda: management)
    route = next(
        route for route in app.routes
        if getattr(route, 'path', None) == '/twinkle/sample_to_data_plane'
    )
    request = Request({'type': 'http', 'headers': []})
    request.state.session_id = 'session'
    body = types.DataPlaneSampleRequest(
        inputs=[{'input_ids': [1]}],
        adapter_name='adapter',
        group_ids=['group-1'],
        policy_version=3,
        num_samples=1,
        sampling_params={'max_tokens': 4},
    )

    ref = await route.endpoint(request, body, management)

    assert ref.ref_id == 'rollout-ref'
    assert management.scheduled == [{
        'model_id': 'session-adapter',
        'token': 'token',
        'input_tokens': 1,
        'task_type': 'sample_admission',
    }]
    assert management.put_rows == [{
        'train_input': {'input_ids': [1, 7], 'labels': [-100, 7]},
        'sampled_logprobs': [-0.25],
        'tokens': [7],
        'decoded': 'answer',
        'stop_reason': 'stop',
        'prompt_logprobs': None,
        'topk_prompt_logprobs': None,
    }]
    assert management.put_tags[0]['group_id'] == 'group-1'
