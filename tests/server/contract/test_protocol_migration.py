# Copyright (c) ModelScope Contributors. All rights reserved.
"""Guards for the wire-contract migration to :mod:`twinkle.protocol`."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

_MIGRATED_NAMES = frozenset(
    """
AdapterRequest AddAdapterRequest AddMetricRequest AddMetricResponse ApplyPatchRequest ApplyPatchResponse
BACKEND_ONLY_KEY BackwardResponse CORE_INPUT_KEYS CalculateLossResponse CalculateMetricRequest
CalculateMetricResponse CancelRequest CancelResponse CapacityInfoResponse Checkpoint CheckpointPathResponse
CheckpointsListResponse ClientFeatures ClipGradAndStepRequest ClipGradAndStepResponse ClipGradNormRequest
ClipGradNormResponse CreateModelRequest CreateRequest CreateResponse CreateSessionRequest CreateSessionResponse Cursor
DataAppendRequest DataGetRequest DataModel DataPlaneForwardOnlyRequest DataPlaneForwardRequest DataPlaneSampleRequest
DataPutRequest DataRef DataReleaseRequest EncodeRequest DataRowsResponse DeleteCheckpointResponse FieldRole ForwardBackwardResponse
ForwardBackwardTaskRequest ForwardOnlyRequest ForwardRequest ForwardResponse GetServerCapabilitiesResponse
GetTrainConfigsResponse HealthResponse LoadRequest LoadResponse LoraConfig LrStepRequest LrStepResponse ModelResult
OkResponse ParsedCheckpointTwinklePath ProcessorCallRequest ProcessorCallResponse ProcessorCreateRequest
ProcessorCreateResponse ProcessorHeartbeatRequest ProcessorHeartbeatResponse ProtocolLimits ResolvedLoadPath
ResponseModel ResumeFromCheckpointRequest RetrieveFutureRequest SampleRequest SampleResponseModel
SampleResponseModelList SampledSequenceModel SamplerAddAdapterRequest SamplerAddAdapterResponse SamplerApplyPatchRequest
SamplerCreateResponse
SamplerSetTemplateRequest SamplerSetTemplateResponse SaveRequest SaveResponse SessionHeartbeatRequest
SessionHeartbeatResponse SetLossRequest SetLossResponse SetLrSchedulerRequest SetLrSchedulerResponse SetOptimizerRequest
SetOptimizerResponse SetProcessorRequest SetProcessorResponse SetTemplateRequest SetTemplateResponse StepRequest
StepResponse StrictRequest SupportedModel TERMINAL_STATUSES TaskEnvelope TaskStatus TrainingProgressResponse TrainingRun
TrainingRunsResponse UnloadAdapterPathsRequest UploadToHubRequest VLM_TENSOR_FIELDS WeightsInfoRequest
WeightsInfoResponse WireInputBatch WireInputFeature WireInputs WireMessage WireTrajectory ZeroGradResponse backend_kwarg
backend_only declared_wire_keys export_batch fields_with_role passthrough read_backend_only read_field_role
""".split())


def test_protocol_exports_match_committed_snapshot() -> None:
    import twinkle.protocol.types as types

    assert set(types.__all__) == _MIGRATED_NAMES
    assert len(types.__all__) == 122
    assert 'ErrorResponse' not in types.__all__



def test_sampler_patch_request_has_no_training_handle() -> None:
    from twinkle.protocol.types import ApplyPatchRequest, SamplerApplyPatchRequest
    from twinkle.server.lifecycle.submit import backend_kwargs
    from twinkle_client._request_builder import build_request

    request = build_request(SamplerApplyPatchRequest, patch_cls='ExamplePatch', enabled=True)
    assert request.model_dump() == {'patch_cls': 'ExamplePatch', 'init_kwargs': {'enabled': True}}
    assert backend_kwargs(request) == {'enabled': True}
    with pytest.raises(ValidationError, match='training_id'):
        SamplerApplyPatchRequest.model_validate({'patch_cls': 'ExamplePatch', 'training_id': 'training-1'})
    with pytest.raises(ValidationError, match='training_id'):
        ApplyPatchRequest(patch_cls='ExamplePatch', adapter_name='train-adapter')
