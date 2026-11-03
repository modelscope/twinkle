# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np

from twinkle.template.contract import append_and_finalize


def _prompt(mm_token_type_ids):
    return {"input_ids": [11, 12, 13], "labels": [-100, -100, -100], "mm_token_type_ids": mm_token_type_ids}


def _append(prompt, tokens):
    return append_and_finalize(prompt, tokens, scored=True, sampled=True, finalize=lambda rows: rows)


def test_batched_list_token_type_ids_extend_on_last_axis():
    # A dataset-encoded prompt stores the processor's [1, L] mm_token_type_ids as a nested list.
    result = _append(_prompt([[0, 0, 0]]), [21, 22])

    assert np.asarray(result["mm_token_type_ids"]).tolist() == [[0, 0, 0, 0, 0]]
    assert list(result["input_ids"]) == [11, 12, 13, 21, 22]


def test_flat_list_token_type_ids_stay_flat():
    result = _append(_prompt([0, 0, 0]), [21, 22])

    assert list(result["mm_token_type_ids"]) == [0, 0, 0, 0, 0]


def test_prompt_feature_is_left_untouched():
    prompt = _prompt([[0, 0, 0]])
    _append(prompt, [21])

    assert prompt["mm_token_type_ids"] == [[0, 0, 0]]
