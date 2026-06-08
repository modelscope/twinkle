# OpenAI-compatible endpoint E2E test
#
# Requires: server with vLLM sampler running (server_e2e.py config).
# Tests both non-streaming and streaming /v1/chat/completions via OpenAI SDK.
#
# Usage:
#   python -u openai_e2e.py

import sys
import time

from openai import OpenAI

BASE_URL = 'http://127.0.0.1:8000/api/v1'
API_KEY = 'EMPTY_API_KEY'
MODEL = 'Qwen/Qwen3.5-4B'


def test_list_models(client: OpenAI):
    print('--- Phase 1: GET /models ---')
    models = client.models.list()
    model_ids = [m.id for m in models.data]
    print(f'  Available models: {model_ids}')
    assert MODEL in model_ids, f'{MODEL} not in {model_ids}'
    print('  PASS\n')


def test_non_streaming(client: OpenAI):
    print('--- Phase 2: Non-streaming chat completion ---')
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'What is 2+2? Answer in one word.'},
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
    print('--- Phase 3: Streaming chat completion ---')
    t0 = time.time()
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {'role': 'user', 'content': 'Count from 1 to 5.'},
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
            print(f'  chunk: {delta.content!r}')

    elapsed = time.time() - t0
    print(f'  Total chunks: {len(chunks)}')
    print(f'  Full content: {full_content!r}')
    print(f'  Final finish_reason: {chunks[-1].choices[0].finish_reason}')
    print(f'  Elapsed: {elapsed:.2f}s')
    assert len(chunks) > 1, 'Expected multiple chunks'
    assert full_content and len(full_content) > 0, 'Empty streamed response'
    assert chunks[-1].choices[0].finish_reason in ('stop', 'length')
    print('  PASS\n')


def test_multi_turn(client: OpenAI):
    print('--- Phase 4: Multi-turn conversation ---')
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {'role': 'system', 'content': 'You are a math tutor.'},
            {'role': 'user', 'content': 'What is 3*7?'},
            {'role': 'assistant', 'content': '3*7 = 21'},
            {'role': 'user', 'content': 'Now add 4 to that.'},
        ],
        max_tokens=32,
        temperature=0.1,
    )
    content = resp.choices[0].message.content
    print(f'  Content: {content!r}')
    assert content and len(content) > 0, 'Empty response'
    print('  PASS\n')


def main():
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    test_list_models(client)
    test_non_streaming(client)
    test_streaming(client)
    test_multi_turn(client)

    print('=' * 50)
    print('ALL PHASES PASSED')
    print('=' * 50)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'\nFAILED: {e}', file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
