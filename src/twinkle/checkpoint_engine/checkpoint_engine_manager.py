from typing import Optional, Dict, Any


class CheckpointEngineManager:

    def __init__(
        self,
        model: "TwinkleModel",
        sampler: "VLLMSampler",
        bucket_size_mb: int = 2048,
    ) -> None:
        assert hasattr(model, '_actors') and model._actors, \
            "CheckpointEngineManager requires model to be deployed as Ray actors"
        assert hasattr(sampler, '_actors') and sampler._actors, \
            "CheckpointEngineManager requires sampler to be deployed as Ray actors"

        self.model = model
        self.sampler = sampler
        self.bucket_size_mb = bucket_size_mb

        self.base_sync_done: bool = False
        self._peft_config: Optional[Dict[str, Any]] = None

    def sync_weights(self, adapter_name: str = ''):
        model_world_size = len(self.model._actors)
        sampler_world_size = len(self.sampler._actors)
