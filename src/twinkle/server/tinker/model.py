# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Tinker-compatible model management server.

This module provides a Ray Serve deployment that manages distributed training models.
It handles:
1. Model and adapter lifecycle (create, load, unload)
2. Training operations (forward, backward, optimizer steps)
3. Checkpoint management (save/load weights)
4. Multi-user support with token-based isolation
"""
import traceback
from fastapi import FastAPI, Request
from peft import LoraConfig
from ray import serve
from tinker import types
from typing import Any, Dict, Optional

import twinkle
from twinkle import DeviceGroup, DeviceMesh
from twinkle.server.utils.adapter_manager import AdapterManagerMixin
from twinkle.server.utils.state import ServerStateProxy, get_server_state
from twinkle.server.utils.task_queue import TaskQueueConfig, TaskQueueMixin
from twinkle.server.utils.validation import verify_request_token
from twinkle.utils.logger import get_logger
from .common.io_utils import create_checkpoint_manager, create_training_run_manager

logger = get_logger()


def build_model_app(model_id: str,
                    nproc_per_node: int,
                    device_group: Dict[str, Any],
                    device_mesh: Dict[str, Any],
                    deploy_options: Dict[str, Any],
                    use_megatron: bool = False,
                    adapter_config: Dict[str, Any] = {},
                    queue_config: Optional[Dict[str, Any]] = {},
                    **kwargs):
    """Build a model management application for distributed training.

    This factory function creates a Ray Serve deployment that manages a training model
    with support for multiple adapters (LoRA) and multi-user isolation.

    Args:
        model_id: Base model identifier (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
        nproc_per_node: Number of processes per node for distributed training
        device_group: Device group configuration dict
        device_mesh: Device mesh configuration dict for tensor parallelism
        deploy_options: Ray Serve deployment options
        use_megatron: Whether to use Megatron backend (vs Transformers)
        queue_config: Task queue configuration (rate limiting, etc.)
        **kwargs: Additional model initialization arguments

    Returns:
        Configured Ray Serve deployment bound with parameters
    """
    app = FastAPI()

    @app.middleware('http')
    async def verify_token(request: Request, call_next):
        """Middleware to verify authentication token for all requests."""
        return await verify_request_token(request=request, call_next=call_next)

    @serve.deployment(name='ModelManagement')
    @serve.ingress(app)
    class ModelManagement(TaskQueueMixin, AdapterManagerMixin):
        """Model management service handling training operations.

        This class manages:
        - Base model and multiple adapter instances (multi-user LoRA)
        - Training operations (forward, backward, optimizer steps)
        - Adapter lifecycle with automatic cleanup via AdapterManagerMixin
        - Per-user adapter limits and tracking
        """

        def __init__(self,
                     nproc_per_node: int,
                     device_group: Dict[str, Any],
                     device_mesh: Dict[str, Any],
                     use_megatron: bool = False,
                     queue_config: Optional[Dict[str, Any]] = None,
                     **kwargs):
            """Initialize the model management service.

            Args:
                nproc_per_node: Number of processes per node
                device_group: Device group configuration
                device_mesh: Device mesh configuration for parallelism
                use_megatron: Whether to use Megatron backend
                queue_config: Task queue configuration dict
                **kwargs: Additional model initialization arguments
            """
            self.device_group = DeviceGroup(**device_group)
            twinkle.initialize(
                mode='ray', nproc_per_node=nproc_per_node, groups=[self.device_group], lazy_collect=False)
            if 'mesh_dim_names' in device_mesh:
                self.device_mesh = DeviceMesh(**device_mesh)
            else:
                self.device_mesh = DeviceMesh.from_sizes(**device_mesh)
            self.use_megatron = use_megatron
            replica_context = serve.get_replica_context()
            replica_id = replica_context.replica_id.unique_id
            # Initialize model immediately - choose backend based on use_megatron
            if use_megatron:
                from .common.megatron_model import TwinkleCompatMegatronModel
                self.model = TwinkleCompatMegatronModel(
                    model_id=model_id,
                    device_mesh=self.device_mesh,
                    remote_group=self.device_group.name,
                    instance_id=replica_id,
                    **kwargs)
            else:
                from .common.transformers_model import TwinkleCompatTransformersModel
                self.model = TwinkleCompatTransformersModel(
                    model_id=model_id,
                    device_mesh=self.device_mesh,
                    remote_group=self.device_group.name,
                    instance_id=replica_id,
                    **kwargs)
            self.base_model = model_id
            self.state: ServerStateProxy = get_server_state()

            # Initialize task queue
            self._init_task_queue(TaskQueueConfig.from_dict(queue_config))

            self._init_adapter_manager(**adapter_config)
            self.start_adapter_countdown()

        """
        TODO This is a cache system, we must change to sticky routing
            Reference docs:
            1. [Now]https://docs.ray.io/en/latest/serve/model-multiplexing.html
            2. https://docs.ray.io/en/latest/serve/llm/architecture/routing-policies.html
            3. https://github.com/ray-project/ray/pull/56855/changes
            4. Direct call actor instead of http or handler in server.py
        """

        # @serve.multiplexed(max_num_models_per_replica=kwargs.get('max_loras', 5))
        # async def get_multiplexed_adapter(self, request_id: str):
        # return request_id

        def _cleanup_adapter(self, adapter_name: str) -> None:
            """Common adapter cleanup logic used by both manual unload and automatic expiration.

            This method handles:
            1. Clearing adapter state
            2. Removing adapter from model
            3. Unregistering from adapter manager
            4. Removing from server state

            Args:
                adapter_name: Name of the adapter to clean up
            """
            # Remove from model if it exists
            if self.get_adapter_info(adapter_name):
                # Clear adapter state
                self.clear_adapter_state(adapter_name)

                self.model.remove_adapter(adapter_name)
                # Unregister from adapter manager
                self.unregister_adapter(adapter_name)

                # Remove from server state
                self.state.unload_model(adapter_name)

        def _on_adapter_expired(self, adapter_name: str) -> None:
            # Called from AdapterManagerMixin's countdown thread.
            # Fail any pending tasks for this adapter/model.
            self.fail_pending_tasks_for_model(adapter_name, reason='Adapter expired')
            # Perform common cleanup (without token since it's automatic)
            self._cleanup_adapter(adapter_name)

        @app.post('/create_model')
        async def create_model(self, request: Request, body: types.CreateModelRequest) -> types.UntypedAPIFuture:
            """Create a new model adapter for training.

            This endpoint:
            1. Registers the model in server state
            2. Creates a LoRA adapter with specified config
            3. Sets up processor, loss, and optimizer for the adapter
            4. Saves metadata to training run manager

            Args:
                request: FastAPI request with auth token
                body: CreateModelRequest with base_model and lora_config

            Returns:
                UntypedAPIFuture wrapping CreateModelResponse with model_id
            """
            # Register a new model_id for each create_model call
            model_id = self.state.register_model(body.model_dump(), token=request.state.token)

            async def _create_adapter():
                try:
                    if body.lora_config:
                        # TODO: support more lora config parameters, train_unembed, etc.
                        lora_cfg = LoraConfig(r=body.lora_config.rank, target_modules='all-linear')

                        adapter_name = self.get_adapter_name(adapter_name=model_id)

                        # Register adapter FIRST (limit check happens inside register_adapter)
                        self.register_adapter(adapter_name, request.state.token, session_id=body.session_id)

                        # Create adapter AFTER successful registration
                        self.model.add_adapter_to_model(adapter_name=adapter_name, config_or_dir=lora_cfg)

                        self.model.set_template('Template', adapter_name=adapter_name, model_id=self.base_model)
                        self.model.set_processor('InputProcessor', adapter_name=adapter_name)
                        self.model.set_optimizer('Adam', adapter_name=adapter_name)

                        # Fresh adapter has no accumulated gradients.
                        self.set_adapter_state(adapter_name, 'grad_ready', False)

                    training_run_manager = create_training_run_manager(request.state.token)
                    training_run_manager.save(model_id, body)

                    return types.CreateModelResponse(model_id=model_id)
                except Exception:
                    # Ensure we don't leave stale grad state.
                    adapter_name = self.get_adapter_name(adapter_name=model_id)
                    self._cleanup_adapter(adapter_name)

                    logger.error(traceback.format_exc())
                    return types.RequestFailedResponse(
                        error=traceback.format_exc(),
                        category=types.RequestErrorCategory.Server,
                    )

            return await self.schedule_task(
                _create_adapter,
                model_id=model_id,
                token=request.state.token,
                task_type='create_model',
            )

        @app.post('/get_info')
        async def get_info(self, request: Request, body: types.GetInfoRequest) -> types.GetInfoResponse:
            """Get information about a model.

            Args:
                request: FastAPI request with auth token
                body: GetInfoRequest with model_id

            Returns:
                GetInfoResponse with model metadata (name, lora_rank, etc.)
            """
            # Note: get_info doesn't require token for reading metadata in tinker
            # Using a default token or None since this is read-only
            training_run_manager = create_training_run_manager(request.state.token)
            metadata = training_run_manager.get(str(body.model_id))
            model_name = metadata.base_model if metadata else model_id
            lora_rank = None
            is_lora = False
            if metadata and hasattr(metadata, 'lora_rank') and metadata.lora_rank:
                lora_rank = metadata.lora_rank
                is_lora = metadata.is_lora
            return types.GetInfoResponse(
                model_data=types.ModelData(model_name=model_name),
                model_id=body.model_id,
                is_lora=is_lora,
                lora_rank=lora_rank,
                model_name=model_name,
            )

        @app.post('/unload_model')
        async def unload_model(self, request: Request, body: types.UnloadModelRequest) -> types.UntypedAPIFuture:
            """Unload a model adapter from memory.

            Removes the adapter and updates user adapter counts.

            Args:
                request: FastAPI request with auth token
                body: UnloadModelRequest with model_id

            Returns:
                UntypedAPIFuture wrapping UnloadModelResponse
            """

            async def _do_unload():
                # Only remove adapter, not the base model
                adapter_name = self.get_adapter_name(adapter_name=body.model_id)
                # Use common cleanup logic
                self._cleanup_adapter(adapter_name)
                return types.UnloadModelResponse(model_id=body.model_id)

            return await self.schedule_task(
                _do_unload,
                model_id=body.model_id,
                token=request.state.token,
                task_type='unload_model',
            )

        @app.post('/forward')
        async def forward(self, request: Request, body: types.ForwardRequest) -> types.UntypedAPIFuture:
            """Execute forward pass without backward pass.

            Used for inference or evaluation without gradient computation.

            Args:
                request: FastAPI request with auth token
                body: ForwardRequest with input data

            Returns:
                UntypedAPIFuture wrapping ForwardBackwardOutput with loss
            """

            async def _do_forward():
                try:
                    adapter_name = self.get_adapter_name(adapter_name=body.model_id)
                    self.assert_adapter_exists(adapter_name=adapter_name)

                    # Touch adapter to reset inactivity counter
                    self.touch_adapter(adapter_name)

                    datum_list = body.forward_input.data
                    loss_fn_config = body.forward_input.loss_fn_config or {}

                    output = self.model.forward_only(inputs=datum_list, adapter_name=adapter_name)
                    loss = self.model.calculate_loss(adapter_name=adapter_name, **loss_fn_config)
                    return types.ForwardBackwardOutput(
                        loss_fn_output_type='CrossEntropyLossReturn',
                        loss_fn_outputs=output,
                        metrics={'loss:sum': loss},
                    )
                except Exception:
                    logger.error(traceback.format_exc())
                    return types.RequestFailedResponse(
                        error=traceback.format_exc(),
                        category=types.RequestErrorCategory.Server,
                    )

            # Calculate input tokens and batch size for validation
            datum_list = body.forward_input.data
            input_tokens = sum(len(d.model_input.to_ints()) for d in datum_list)
            batch_size = len(datum_list)
            return await self.schedule_task(
                _do_forward,
                model_id=body.model_id,
                token=request.state.token,
                input_tokens=input_tokens,
                batch_size=batch_size,
                data_world_size=self.device_mesh.data_world_size,
                task_type='forward',
            )

        @app.post('/forward_backward')
        async def forward_backward(self, request: Request,
                                   body: types.ForwardBackwardRequest) -> types.UntypedAPIFuture:
            """Execute forward and backward pass for training.

            This combines forward pass and gradient computation. The implementation
            differs based on backend:
            - Megatron: Uses combined forward_backward method
            - Transformers: Separate forward, calculate_loss, backward calls

            Args:
                request: FastAPI request with auth token
                body: ForwardBackwardRequest with training data

            Returns:
                UntypedAPIFuture wrapping ForwardBackwardOutput with loss and metrics
            """

            async def _do_forward_backward():
                try:
                    adapter_name = self.get_adapter_name(adapter_name=body.model_id)
                    self.assert_adapter_exists(adapter_name=adapter_name)

                    # Touch adapter to reset inactivity counter
                    self.touch_adapter(adapter_name)

                    datum_list = body.forward_backward_input.data
                    loss_fn = body.forward_backward_input.loss_fn
                    loss_fn_config = body.forward_backward_input.loss_fn_config or {}

                    # Unified forward_backward for both Megatron and Transformers
                    output, loss = self.model.forward_backward(
                        inputs=datum_list, adapter_name=adapter_name, loss_fn=loss_fn, **loss_fn_config)
                    if loss_fn == 'importance_sampling':
                        output_type = 'ImportanceSamplingLossReturn'
                    else:
                        output_type = 'CrossEntropyLossReturn'
                    # Mark gradients as ready after a successful forward_backward.
                    self.set_adapter_state(adapter_name, 'grad_ready', True)
                    return types.ForwardBackwardOutput(
                        loss_fn_output_type=output_type,
                        loss_fn_outputs=output,
                        metrics={'loss:avg': loss},
                    )
                except Exception:
                    logger.error(traceback.format_exc())
                    return types.RequestFailedResponse(
                        error=traceback.format_exc(),
                        category=types.RequestErrorCategory.Server,
                    )

            # Calculate input tokens and batch size for validation
            datum_list = body.forward_backward_input.data
            input_tokens = sum(len(d.model_input.to_ints()) for d in datum_list)
            batch_size = len(datum_list)
            return await self.schedule_task(
                _do_forward_backward,
                model_id=body.model_id,
                token=request.state.token,
                input_tokens=input_tokens,
                batch_size=batch_size,
                data_world_size=self.device_mesh.data_world_size,
                task_type='forward_backward',
            )

        @app.post('/optim_step')
        async def optim_step(self, request: Request, body: types.OptimStepRequest) -> types.UntypedAPIFuture:
            """Execute optimizer step to update model weights.

            Applies accumulated gradients to update adapter parameters.

            Args:
                request: FastAPI request with auth token
                body: OptimStepRequest with optimizer parameters

            Returns:
                UntypedAPIFuture wrapping OptimStepResponse
            """

            async def _do_optim():
                try:
                    adapter_name = self.get_adapter_name(adapter_name=body.model_id)
                    self.assert_adapter_exists(adapter_name=adapter_name)

                    # Disallow empty step (must have at least one forward_backward since last step)
                    if not self.get_adapter_state(adapter_name, 'grad_ready', False):
                        raise RuntimeError(
                            f'No accumulated gradients for adapter={adapter_name}; call forward_backward before optim_step'  # noqa: E501
                        )

                    # Touch adapter to reset inactivity counter
                    self.touch_adapter(adapter_name)

                    self.model.step(adam_params=body.adam_params, adapter_name=adapter_name)
                    # Clear grad-ready after a successful step.
                    self.set_adapter_state(adapter_name, 'grad_ready', False)
                    metrics = self.model.calculate_metric(is_training=True, adapter_name=adapter_name)
                    return types.OptimStepResponse(metrics=metrics)
                except Exception:
                    logger.error(traceback.format_exc())
                    return types.RequestFailedResponse(
                        error=traceback.format_exc(),
                        category=types.RequestErrorCategory.Server,
                    )

            return await self.schedule_task(
                _do_optim,
                model_id=body.model_id,
                token=request.state.token,
                task_type='optim_step',
            )

        @app.post('/save_weights')
        async def save_weights(self, request: Request, body: types.SaveWeightsRequest) -> types.UntypedAPIFuture:
            """Save model adapter weights to storage.

            Saves both model weights and optimizer state for training resumption.
            Uses token-based isolation for user-specific storage.

            Args:
                request: FastAPI request with auth token
                body: SaveWeightsRequest with path and model_id

            Returns:
                UntypedAPIFuture wrapping SaveWeightsResponse with saved path
            """

            async def _do_save():
                try:
                    adapter_name = self.get_adapter_name(adapter_name=body.model_id)
                    self.assert_adapter_exists(adapter_name=adapter_name)

                    # Touch adapter to reset inactivity counter
                    self.touch_adapter(adapter_name)

                    # Extract token from request for user isolation
                    token = request.state.token
                    checkpoint_manager = create_checkpoint_manager(token)

                    # get save dir with token-based isolation
                    checkpoint_name = checkpoint_manager.get_ckpt_name(body.path)
                    save_dir = checkpoint_manager.get_save_dir(model_id=body.model_id, is_sampler=False)

                    self.model.save(
                        name=checkpoint_name, output_dir=save_dir, adapter_name=adapter_name, save_optimizer=True)

                    tinker_path = checkpoint_manager.save(body.model_id, name=checkpoint_name, is_sampler=False)

                    return types.SaveWeightsResponse(path=tinker_path, type='save_weights')
                except Exception:
                    logger.error(traceback.format_exc())
                    return types.RequestFailedResponse(
                        error=traceback.format_exc(),
                        category=types.RequestErrorCategory.Server,
                    )

            return await self.schedule_task(
                _do_save,
                model_id=body.model_id,
                token=request.state.token,
                task_type='save_weights',
            )

        @app.post('/save_weights_for_sampler')
        async def save_weights_for_sampler(self, request: Request,
                                           body: types.SaveWeightsForSamplerRequest) -> types.UntypedAPIFuture:
            """Save/convert weights for inference use.

            Saves adapter weights without optimizer state for use with sampler.
            Creates a sampling session for tracking.

            Args:
                request: FastAPI request with auth token
                body: SaveWeightsForSamplerRequest with model_id and path

            Returns:
                UntypedAPIFuture wrapping SaveWeightsForSamplerResponseInternal
            """

            async def _do_save_for_sampler():
                try:

                    adapter_name = self.get_adapter_name(adapter_name=body.model_id)
                    self.assert_adapter_exists(adapter_name=adapter_name)

                    # Touch adapter to reset inactivity counter
                    self.touch_adapter(adapter_name)

                    # Extract token from request for user isolation
                    token = request.state.token
                    checkpoint_manager = create_checkpoint_manager(token)

                    # get save dir with token-based isolation
                    checkpoint_name = checkpoint_manager.get_ckpt_name(body.path)
                    save_dir = checkpoint_manager.get_save_dir(model_id=body.model_id, is_sampler=True)
                    # NOTE: Need to save meta first to ensure only one sample weight exists
                    tinker_path = checkpoint_manager.save(body.model_id, name=checkpoint_name, is_sampler=True)

                    logger.info(f'Saving weights to {save_dir}')
                    # Save weights with save_optimizer=False for sampler use
                    self.model.save(
                        name=checkpoint_name, output_dir=save_dir, adapter_name=adapter_name, save_optimizer=False)

                    # Create sampling session with resolved model_path/base_model.
                    payload = body.model_dump()
                    payload['model_path'] = tinker_path
                    metadata = self.state.get_model_metadata(body.model_id) or {}
                    if metadata.get('base_model'):
                        payload['base_model'] = metadata['base_model']
                    sampling_session_id = self.state.create_sampling_session(payload)

                    return types.SaveWeightsForSamplerResponseInternal(
                        path=None,  # Disable path return for internal use
                        sampling_session_id=sampling_session_id)
                except Exception:
                    logger.error(traceback.format_exc())
                    return types.RequestFailedResponse(
                        error=traceback.format_exc(),
                        category=types.RequestErrorCategory.Server,
                    )

            return await self.schedule_task(
                _do_save_for_sampler,
                model_id=body.model_id,
                token=request.state.token,
                task_type='save_weights_for_sampler',
            )

        @app.post('/load_weights')
        async def load_weights(self, request: Request, body: types.LoadWeightsRequest) -> types.UntypedAPIFuture:
            """Load model adapter weights from storage.

            Loads weights and optionally optimizer state for training resumption.
            Uses token-based isolation for user-specific storage access.

            Args:
                request: FastAPI request with auth token
                body: LoadWeightsRequest with path and optimizer flag

            Returns:
                UntypedAPIFuture wrapping LoadWeightsResponse
            """

            async def _do_load():
                try:
                    assert self.model is not None, 'Model not loaded, please load model first'

                    adapter_name = self.get_adapter_name(adapter_name=body.model_id)
                    self.assert_adapter_exists(adapter_name=adapter_name)

                    # Touch adapter to reset inactivity counter
                    self.touch_adapter(adapter_name)

                    # Extract token from request for user isolation
                    token = request.state.token

                    weight_path = body.path
                    load_optimizer = body.optimizer

                    self.model.load(
                        checkpoint_dir=weight_path,
                        load_optimizer=load_optimizer,
                        adapter_name=adapter_name,
                        token=token)

                    # Loading a checkpoint should reset step readiness.
                    self.set_adapter_state(adapter_name, 'grad_ready', False)
                    return types.LoadWeightsResponse(path=body.path, type='load_weights')
                except Exception:
                    logger.error(traceback.format_exc())
                    return types.RequestFailedResponse(
                        error=traceback.format_exc(),
                        category=types.RequestErrorCategory.Server,
                    )

            return await self.schedule_task(
                _do_load,
                model_id=body.model_id,
                token=request.state.token,
                task_type='load_weights',
            )

    return ModelManagement.options(**deploy_options).bind(nproc_per_node, device_group, device_mesh, use_megatron,
                                                          queue_config, **kwargs)
