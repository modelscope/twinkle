#!/usr/bin/env python3
"""Convert a SelfCognition JSON/JSONL dataset to chat messages JSONL."""

import argparse
import json
from pathlib import Path
from typing import Any, Iterator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert query/response SelfCognition samples into messages format.')
    parser.add_argument('input', type=Path, help='Input .json or .jsonl file')
    parser.add_argument('output', type=Path, help='Output .jsonl file')
    parser.add_argument('--model-name', default='twinkle模型', help='Value used for {{NAME}}')
    parser.add_argument('--model-author', default='twinkle团队', help='Value used for {{AUTHOR}}')
    return parser.parse_args()


def read_samples(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == '.jsonl':
        with path.open('r', encoding='utf-8') as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f'Line {line_number} must be a JSON object')
                yield value
        return

    with path.open('r', encoding='utf-8') as source:
        value = json.load(source)

    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise ValueError('A JSON input must contain an object or a list of objects')
    for index, sample in enumerate(value):
        if not isinstance(sample, dict):
            raise ValueError(f'Item {index} must be a JSON object')
        yield sample


def replace_placeholders(text: str, model_name: str, model_author: str) -> str:
    return text.replace('{{NAME}}', model_name).replace('{{AUTHOR}}', model_author)


def convert_sample(sample: dict[str, Any], model_name: str, model_author: str) -> dict[str, Any]:
    if 'query' not in sample or 'response' not in sample:
        raise ValueError("Each sample must contain both 'query' and 'response'")
    if not isinstance(sample['query'], str) or not isinstance(sample['response'], str):
        raise ValueError("The 'query' and 'response' fields must be strings")

    return {
        'messages': [
            {
                'role': 'system',
                'content': 'You are a helpful assistant.'
            },
            {
                'role': 'user',
                'content': replace_placeholders(sample['query'], model_name, model_author),
            },
            {
                'role': 'assistant',
                'content': replace_placeholders(sample['response'], model_name, model_author),
            },
        ]
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.output.open('w', encoding='utf-8') as destination:
        for count, sample in enumerate(read_samples(args.input), start=1):
            converted = convert_sample(sample, args.model_name, args.model_author)
            destination.write(json.dumps(converted, ensure_ascii=False) + '\n')

    print(f'Converted {count} samples to {args.output}')


if __name__ == '__main__':
    main()
