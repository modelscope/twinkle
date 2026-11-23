from threading import Lock
from types import SimpleNamespace

from cookbook.client.async_rl import client_orchestrated_grpo as recipe
from cookbook.client.twinkle import short_math_grpo
from twinkle.data_format import SampleResponse, SampledSequence
from twinkle.dataset import DatasetMeta


class _Template:
    """One prompt token and one generated token, with labels in output order."""

    truncation_strategy = 'delete'

    def encode(self, trajectory, **kwargs):
        return {**trajectory, 'input_ids': [10], 'labels': [-100], 'length': 1}

    @staticmethod
    def parse_tool_call(text):
        return []

    tool_call_errors = parse_tool_call


def test_streaming_grpo_uses_sampled_tokens_and_publishes_only_after_updates(monkeypatch):
    saved, trained, sampled = [], [], []
    closed, lock = [], Lock()

    class Model:
        completed_step_calls = 0

        def save(self, name, *, save_optimizer):
            assert not save_optimizer
            saved.append(name)
            return SimpleNamespace(twinkle_path=name)

        def forward_backward(self, *, inputs):
            trained.append(inputs)

        def clip_grad_and_step(self, **kwargs):
            self.completed_step_calls += 1

        def calculate_metric(self, **kwargs):
            return {}

    model = Model()
    monkeypatch.setattr(Model, 'add_adapter_to_model', lambda *args, **kwargs: None, raising=False)
    for method in ('set_loss', 'set_optimizer', 'set_processor', 'set_template'):
        monkeypatch.setattr(Model, method, lambda *args, **kwargs: None, raising=False)

    def sample(inputs, params, *, adapter_uri, **kwargs):
        (entry,) = inputs
        assert params.num_samples == 1 and params.logprobs == 1
        question = entry['messages'][-1]['content']
        with lock:
            index = sum(prompt == question for prompt, _ in sampled)
            sampled.append((question, adapter_uri))
        correct = question == 'same reward' or index == 0
        token = 20 if correct else 21
        feature = {**entry, 'input_ids': [10, token], 'labels': [token, -100], 'length': 2}
        reply = r'\boxed{1}' if correct else r'\boxed{0}'
        return [SampleResponse(sequences=[SampledSequence(
            stop_reason='stop', tokens=[token], decoded=reply,
            logprobs=[[(token, -0.5)]], new_input_feature=feature)])]

    sampler = SimpleNamespace(set_template=lambda *args, **kwargs: None, sample=sample)
    client = SimpleNamespace(model=lambda _: model, sampler=lambda _: sampler, close=lambda: closed.append(True))
    monkeypatch.setattr(recipe, 'init_twinkle_client', lambda: client)
    monkeypatch.setattr(recipe, 'Qwen3_5Template', lambda **kwargs: _Template())
    monkeypatch.setattr(recipe, 'NUM_GENERATIONS', 2)
    monkeypatch.setattr(recipe, 'MAX_STEPS', 2)
    data = DatasetMeta(data=[{'question': question, 'answer': '#### 1'}
                             for question in ('same reward', 'different rewards')])
    monkeypatch.setattr(recipe, 'DatasetMeta', lambda *args, **kwargs: data)

    recipe.train()

    assert model.completed_step_calls == len(trained) == 1
    assert len(saved) == 2 and saved[0].endswith('-0') and saved[1].endswith('-1')
    assert len(sampled) == 4 and {uri for _, uri in sampled} == {saved[0]}
    nonzero = trained[0]
    assert {row['response_token_ids'][0] for row in nonzero} == {20, 21}
    for row in nonzero:
        token, = row['response_token_ids']
        assert row['labels'] == [token, -100]
        assert row['old_logps'] == [-0.5]
        assert (row['advantages'][0] > 0) == (token == 20)
    assert closed == [True]


def test_sync_grpo_trains_with_correctness_gated_brevity_and_sampled_logprobs(monkeypatch):
    prompts = [{'user_data': [('ground_truth', '1')]}]
    replies = [r'\boxed{1}', r'\boxed{0}']
    sequences = [SampledSequence(
        stop_reason='stop', tokens=[token], decoded=reply,
        logprobs=[[(token, -0.5)]], new_input_feature={'input_ids': [10, token], 'labels': [token, -100]})
        for token, reply in zip((20, 21), replies)]
    calls = []
    model = SimpleNamespace(
        save=lambda **kwargs: SimpleNamespace(twinkle_path='policy'),
        forward_backward=lambda **kwargs: calls.append(kwargs),
        clip_grad_and_step=lambda: calls.append('update'),
        calculate_metric=lambda **kwargs: SimpleNamespace(result={}))
    for method in ('add_adapter_to_model', 'set_loss', 'set_optimizer', 'set_processor', 'set_template'):
        setattr(model, method, lambda *args, **kwargs: None)
    sampler = SimpleNamespace(set_template=lambda *args, **kwargs: None,
                              sample=lambda **kwargs: [SampleResponse(sequences=sequences)])
    client = SimpleNamespace(model=lambda _: model, sampler=lambda _: sampler,
                             close=lambda: calls.append('closed'))
    monkeypatch.setattr(short_math_grpo, 'init_twinkle_client', lambda: client)
    monkeypatch.setattr(short_math_grpo, 'create_gsm8k_dataset', lambda: None)
    monkeypatch.setattr(short_math_grpo, 'DataLoader', lambda **kwargs: [prompts])
    monkeypatch.setattr(short_math_grpo, 'NUM_GENERATIONS', 2)
    monkeypatch.setattr(short_math_grpo, 'MAX_STEPS', 1)

    short_math_grpo.train()

    backward, update, close = calls
    assert backward['inputs'] == [sequence.new_input_feature for sequence in sequences]
    assert backward['old_logps'] == [[-0.5], [-0.5]]
    assert backward['advantages'][0] > 0 > backward['advantages'][1]
    assert (update, close) == ('update', 'closed')
    assert short_math_grpo.compute_rewards([
        {'messages': [{'role': 'assistant', 'content': reply}], **prompts[0]} for reply in replies
    ])[0] == [1.25, 0.0]
