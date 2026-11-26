"""Client-side Datum conversion preserves labels and multimodal payloads."""
import numpy as np

from twinkle_client.utils.tinker_datum import input_feature_to_datum


def test_masked_labels_and_multimodal_payload():
    feature = {
        'input_ids': [[11, 12, 13]],
        'labels': [-100, 12, 13],
        'pixel_values': np.array([[1.0, 2.0]], dtype=np.float32),
        'image_grid_thw': [[1, 1, 2]],
    }
    datum = input_feature_to_datum(feature)
    assert datum.model_input.to_ints() == [11, 12, 13]
    assert datum.loss_fn_inputs['target_tokens'].to_numpy().tolist() == [0, 12, 13]
    assert datum.loss_fn_inputs['weights'].to_numpy().tolist() == [0.0, 1.0, 1.0]
    np.testing.assert_array_equal(datum.loss_fn_inputs['pixel_values'].to_numpy(), feature['pixel_values'])
    assert datum.loss_fn_inputs['image_grid_thw'].to_numpy().tolist() == [[1, 1, 2]]
