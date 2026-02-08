# Copyright (c) ModelScope Contributors. All rights reserved.
from .datum import datum_to_input_feature, input_feature_to_datum
from .transformers_model import _extract_rl_fields_from_inputs as extract_rl_fields_from_inputs
from twinkle.utils import exists, requires
