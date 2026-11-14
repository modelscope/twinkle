# Copyright (c) ModelScope Contributors. All rights reserved.
from .encoding import ENCODED_INPUT_KEYS, is_encoded
from .input_feature import InputFeature
from .message import Message, Tool, ToolCall
from .output import LossOutput, ModelOutput
from .sampling import (PoolingParams, PoolingResponse, SampledSequence, SampleResponse, SamplingMask, SamplingParams,
                       pooling_to_list)
from .tq_fields import (REQUIRED_MODEL_INPUT_FIELDS, ROLLOUT_TRAIN_FIELDS, TRANSFORMERS_INPUT_FIELDS,
                        columns_to_tq_fields, rows_to_tq_fields)
from .trajectory import (CONVERSATION_FIELDS, RESPONSE_FRAME_FIELDS, Trajectory, assert_response_frame,
                         attach_user_data, conversation, ground_truth, last_assistant_text, msg_content_text,
                         pack_user_data, pack_value, trainable_runs, turn_token_spans, user_data_get,
                         validate_trajectory)

# A collated batch contains placeholder media whose aligned features must contribute zero.
VISION_DUMMY_FLAG = "_vision_keepalive_dummy"
