# OpenAI-compatible endpoint E2E test
#
# Requires: server with vLLM sampler running (server_e2e.py config).
# Tests both non-streaming and streaming /v1/chat/completions via OpenAI SDK.
#
# Usage:
#   python -u tests/server/integration/openai_e2e.py

import sys
import time
from openai import OpenAI

BASE_URL = 'http://127.0.0.1:8000/api/v1'
API_KEY = 'EMPTY_API_KEY'
MODEL = 'Qwen/Qwen3.5-4B'


def test_list_models(client: OpenAI):
    print('--- Step 1: GET /models ---')
    models = client.models.list()
    model_ids = [m.id for m in models.data]
    print(f'  Available models: {model_ids}')
    assert MODEL in model_ids, f'{MODEL} not in {model_ids}'
    print('  PASS\n')


def test_non_streaming(client: OpenAI):
    print('--- Step 2: Non-streaming chat completion ---')
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                'role': 'system',
                'content': 'You are a helpful assistant.'
            },
            {
                'role': 'user',
                'content': 'What is 2+2? Answer in one word.'
            },
        ],
        max_tokens=32,
        temperature=0.1,
    )
    elapsed = time.time() - t0
    print(f'  Model: {resp.model}')
    print(f'  Choices: {len(resp.choices)}')
    content = resp.choices[0].message.content
    print(f'  Content: {content!r}')
    print(f'  Finish reason: {resp.choices[0].finish_reason}')
    print(f'  Usage: prompt={resp.usage.prompt_tokens}, completion={resp.usage.completion_tokens}')
    print(f'  Elapsed: {elapsed:.2f}s')
    assert content and len(content) > 0, 'Empty response'
    assert resp.choices[0].finish_reason in ('stop', 'length')
    print('  PASS\n')


def test_streaming(client: OpenAI):
    print('--- Step 3: Streaming chat completion ---')
    t0 = time.time()
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                'role': 'user',
                'content': 'Count from 1 to 5.'
            },
        ],
        max_tokens=64,
        temperature=0.1,
        stream=True,
    )

    chunks = []
    full_content = ''
    for chunk in stream:
        chunks.append(chunk)
        delta = chunk.choices[0].delta
        if delta.content:
            full_content += delta.content
            print(f'  chunk: {delta.content[:60]!r}...' if len(delta.content or '') >
                  60 else f'  chunk: {delta.content!r}')

    elapsed = time.time() - t0
    print(f'  Total chunks: {len(chunks)}')
    print(f'  Full content length: {len(full_content)} chars')
    print(f'  Elapsed: {elapsed:.2f}s')
    assert len(chunks) >= 1, 'Expected at least one chunk'
    assert full_content and len(full_content) > 0, 'Empty streamed response'
    # Find the last chunk with a finish_reason
    last_finish = None
    for c in reversed(chunks):
        if c.choices[0].finish_reason:
            last_finish = c.choices[0].finish_reason
            break
    print(f'  Finish reason: {last_finish}')
    assert last_finish in ('stop', 'length')
    print('  PASS\n')


def test_multi_turn(client: OpenAI):
    print('--- Step 4: Multi-turn conversation ---')
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                'role': 'system',
                'content': 'You are a math tutor.'
            },
            {
                'role': 'user',
                'content': 'What is 3*7?'
            },
            {
                'role': 'assistant',
                'content': '3*7 = 21'
            },
            {
                'role': 'user',
                'content': 'Now add 4 to that.'
            },
        ],
        max_tokens=32,
        temperature=0.1,
    )
    content = resp.choices[0].message.content
    print(f'  Content: {content!r}')
    assert content and len(content) > 0, 'Empty response'
    print('  PASS\n')


def test_sticky_session(base_url: str):
    """Verify sticky session routing with X-Twinkle-Replica-Id header.

    Sends multiple requests with the same model key and asserts they all
    report the same replica ID. This proves Serve-Multiplexed-Model-Id
    correctly pins requests to a single replica.

    Requires the sampler to have the inject_replica_id middleware
    (added in sampler/app.py).
    """
    import httpx

    print('--- Step 5: Sticky session verification (replica-id) ---')

    replica_ids = []
    for i in range(5):
        resp = httpx.post(
            f'{base_url}/chat/completions',
            json={
                'model': MODEL,
                'messages': [{
                    'role': 'user',
                    'content': f'Say {i}.'
                }],
                'max_tokens': 5,
                'temperature': 0.1,
            },
            headers={'Authorization': 'Bearer EMPTY_API_KEY'},
            timeout=60,
        )
        assert resp.status_code == 200, f'Request {i} failed: {resp.status_code}'
        rid = resp.headers.get('x-twinkle-replica-id')
        replica_ids.append(rid)
        print(f'  Request {i}: replica_id={rid}')

    # All requests with same model must hit the same replica
    unique = set(replica_ids)
    assert None not in unique, 'X-Twinkle-Replica-Id header missing from responses'
    assert len(unique) == 1, (f'Sticky session broken: requests routed to {len(unique)} different replicas: {unique}')
    print(f'  All 5 requests → replica {replica_ids[0]}')
    print('  PASS\n')


def main():
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    test_list_models(client)
    test_non_streaming(client)
    test_streaming(client)
    test_multi_turn(client)
    test_sticky_session(BASE_URL)

    print('=' * 50)
    print('ALL STEPS PASSED')
    print('=' * 50)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'\nFAILED: {e}', file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
