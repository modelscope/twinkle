import json
import numpy as np
import pytest
import torch
from PIL import Image

from twinkle_agentic.async_rl.component_state_codec import (
    decode_component_state,
    encode_component_state,
)


def test_component_state_json_round_trips_multimodal_values():
    tensor = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    image = Image.new("RGB", (3, 2))
    image.putpixel((1, 1), (12, 34, 56))
    array = np.arange(6, dtype=np.int16).reshape(2, 3)
    state = {
        "pixel_values": tensor,
        "image_grid_thw": torch.tensor([[1, 2, 3]], dtype=torch.int64),
        "images": [
            image,
            {"bytes": b"\x89PNG\r\n", "path": None},
        ],
        "image_array": array,
        "tuple": ("left", 7),
    }

    encoded = encode_component_state(state)
    decoded = decode_component_state(json.loads(json.dumps(encoded)))

    assert torch.equal(decoded["pixel_values"], tensor)
    assert decoded["pixel_values"].dtype == tensor.dtype
    assert torch.equal(decoded["image_grid_thw"], state["image_grid_thw"])
    assert decoded["images"][0].tobytes() == image.tobytes()
    assert decoded["images"][1]["bytes"] == b"\x89PNG\r\n"
    assert np.array_equal(decoded["image_array"], array)
    assert decoded["image_array"].dtype == array.dtype
    assert decoded["tuple"] == ("left", 7)


def test_component_state_rejects_unsupported_values_with_the_field_path():
    with pytest.raises(TypeError, match=r"queries\[0\]\.row\.bad"):
        encode_component_state({
            "queries": [{
                "row": {
                    "bad": object(),
                },
            }],
        })


def test_component_state_escapes_a_user_mapping_with_the_codec_marker():
    state = {
        "__twinkle_component_state_v1__": "bytes",
        "value": b"payload",
    }

    decoded = decode_component_state(json.loads(json.dumps(encode_component_state(state))))

    assert decoded == state
