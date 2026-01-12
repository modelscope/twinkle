# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import List, Dict, Any, Tuple
from copy import deepcopy

from transformers import PreTrainedTokenizer

from twinkle.data_format import Trajectory

PLACEHOLDER = "<<<ASSISTANT_PLACEHOLDER_7f3d2a1b>>>"


def find_subsequence(seq: List[int], subseq: List[int], start: int = 0) -> int:
    """Find the first index of `subseq`"""
    subseq_len = len(subseq)
    for i in range(start, len(seq) - subseq_len + 1):
        if seq[i:i + subseq_len] == subseq:
            return i
    return -1


def split_by_subsequence(seq: List[int], subseq: List[int]) -> List[List[int]]:
    """Split seq by subseq"""
    parts = []
    start = 0
    subseq_len = len(subseq)

    while True:
        pos = find_subsequence(seq, subseq, start)
        if pos == -1:
            parts.append(seq[start:])
            break
        parts.append(seq[start:pos])
        start = pos + subseq_len

    return parts


def build_labels(
        full_ids: List[int],
        template_parts: List[List[int]],
) -> List[int]:
    labels = list(full_ids)
    pos = 0

    for part in template_parts:
        if not part:
            continue

        match_pos = find_subsequence(full_ids, part, pos)

        if match_pos == -1:
            # should not happen
            raise ValueError(f"Template part not found in full_ids at position {pos}")

        for i in range(match_pos, match_pos + len(part)):
            labels[i] = -100

        pos = match_pos + len(part)

    return labels


def tokenize_with_assistant_labels(
        tokenizer: PreTrainedTokenizer,
        trajectory: Trajectory,
        placeholder: str = PLACEHOLDER,
) -> Tuple[List[int], List[int]]:
    messages = [dict(message) for message in trajectory['messages']]
    tools = [dict(tool) for tool in trajectory.get('tools', [])]
    placeholder_ids = tokenizer.encode(placeholder, add_special_tokens=False)

    messages_with_placeholder = deepcopy(messages)
    assistant_count = 0
    for msg in messages_with_placeholder:
        if msg["role"] == "assistant":
            msg["content"] = placeholder
            assistant_count += 1

    full_ids = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
    )

    template_ids = tokenizer.apply_chat_template(
        messages_with_placeholder,
        tools=tools,
        tokenize=True,
    )

    template_parts = split_by_subsequence(template_ids, placeholder_ids)

    if len(template_parts) != assistant_count + 1:
        raise ValueError(
            f"Expected {assistant_count + 1} parts, got {len(template_parts)}. "
            "Placeholder might appear in original content."
        )

    labels = build_labels(full_ids, template_parts)
    if labels and labels[-1] == -100:
        end_idx = len(labels)
        start_idx = end_idx - 1
        while start_idx > 0 and labels[start_idx - 1] == -100:
            start_idx -= 1

        for i in range(max(start_idx, end_idx - 2), end_idx):
            labels[i] = full_ids[i]

    return full_ids, labels
