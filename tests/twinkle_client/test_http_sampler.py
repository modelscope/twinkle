# Copyright (c) ModelScope Contributors. All rights reserved.
from unittest.mock import Mock

import pytest

from twinkle.protocol.types.sampler import SampleResponseModel
from twinkle_client.rollout import HTTPSampler


@pytest.mark.parametrize(('flag', 'prompt_ids', 'accepted'), [
    ({}, [1, 2], True),
    ({}, [1, 3, 2], True),
    ({}, [1, 4, 2], False),
    ({'prompt_token_ids_echoed': True}, [1, 2], True),
    ({'prompt_token_ids_echoed': True}, [1, 3, 2], False),
    ({'prompt_token_ids_echoed': False}, [1, 3, 2], True),
    ({'prompt_token_ids_echoed': False}, [1, 2], False),
    ({'prompt_token_ids_echoed': False}, [1, 4, 2], False),
])
def test_multimodal_prompt_validation(flag, prompt_ids, accepted):
    image = 'data:image/png;base64,AAA='
    visual_tensor = object()
    prompt = {
        'images': [image],
        'engine_input_ids': [1, 2],
        'input_ids': [1, 3, 2],
        'mm_processor_kwargs': {'do_resize': False},
    }
    completion = {'input_ids': [1, 3, 2, 9], 'images': [visual_tensor]}
    template = Mock()
    template.append_completion.return_value = completion
    template.decode.return_value = 'answer'
    response = SampleResponseModel.model_validate({
        'prompt_token_ids': prompt_ids,
        'sequences': [{'stop_reason': 'length', 'tokens': [9]}],
        **flag,
    })
    http = Mock()
    http.sample.return_value = [response]
    sampler = HTTPSampler(http, template)

    if accepted:
        result, = sampler.sample(prompt)
        assert result.sequences[0].new_input_feature['images'] == [visual_tensor]
        assert result.sequences[0].new_input_feature['input_ids'] == [1, 3, 2, 9]
    else:
        with pytest.raises(ValueError, match='prompt_token_ids do not match'):
            sampler.sample(prompt)
        template.append_completion.assert_not_called()

    wire_input, = http.sample.call_args.args[0]
    assert wire_input == {'input_ids': [1, 2], 'images': [image], 'mm_processor_kwargs': {'do_resize': False}}


def test_media_reference_survives_merge_when_template_only_keeps_visual_tensors():
    raw_image = {'bytes': None, 'path': 'unread.png'}
    image = 'data:image/png;base64,AAA='
    pixels = object()
    prompt = {'images': [image], 'engine_input_ids': [1, 2], 'input_ids': [1, 3, 2], 'pixel_values': pixels}
    template = Mock()
    template.append_completion.return_value = {'input_ids': [1, 3, 2, 9], 'pixel_values': pixels}
    template.decode.return_value = 'answer'
    response = SampleResponseModel(
        prompt_token_ids=[1, 3, 2], prompt_token_ids_echoed=False,
        sequences=[{'stop_reason': 'length', 'tokens': [9]}])
    http = Mock()
    http.sample.return_value = [response]

    result, = HTTPSampler(http, template).sample(prompt)
    row = {'images': [raw_image], **result.sequences[0].new_input_feature}
    assert row['images'] == [image]
    assert row['pixel_values'] is pixels
