import pytest

from twinkle.model.multi_lora import MultiLora


def test_check_length_checks_each_sample_independently():
    multi_lora = MultiLora(max_length=4)

    multi_lora.check_length([
        {'input_ids': [1, 2, 3]},
        {'input_ids': [4, 5, 6]},
    ])


def test_check_length_accepts_a_single_input_feature():
    multi_lora = MultiLora(max_length=4)

    multi_lora.check_length({'input_ids': [1, 2, 3, 4]})


def test_check_length_reports_the_oversized_sample():
    multi_lora = MultiLora(max_length=4)

    with pytest.raises(ValueError, match=r'Input length 5 exceeds max_length 4 at sample 1'):
        multi_lora.check_length([
            {'input_ids': [1, 2]},
            {'input_ids': [1, 2, 3, 4, 5]},
        ])
