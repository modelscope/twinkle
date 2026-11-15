# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from datetime import timedelta
from typing import Any, Dict, Literal, Mapping, Optional

from twinkle import DeviceMesh
from twinkle.model.state_dict_utils import collect_local_state, is_saveable_adapter_key
from twinkle.utils import get_logger
from .base import TransformerStrategyBase
from .base import fsdp_pretrained_load_context

logger = get_logger()


class AccelerateStrategy(TransformerStrategyBase):
    """A training strategy that uses `accelerate` to wrap models.

    Args:
        device_mesh: The model device mesh.
        mixed_precision: The mixed precision type.
        ddp_config: Any ddp config passed into accelerate.
        fsdp_config: Any fsdp config passed into accelerate.
    """

    def __init__(
        self,
        device_mesh: Optional[DeviceMesh] = None,
        mixed_precision: Literal['no', 'fp8', 'fp16', 'bf16'] = 'bf16',
        ddp_config: Dict[str, Any] = None,
        fsdp_config: Dict[str, Any] = None,
        memory_efficient_init: bool = False,
    ):
        from accelerate import Accelerator
        from accelerate.utils import InitProcessGroupKwargs

        self.device_mesh = device_mesh
        self.mixed_precision = mixed_precision
        self._memory_efficient_init = memory_efficient_init
        # Tensor parallelism state. The actual sharding is done by transformers at load time
        # (see `init_kwargs`); here we only record the degree and reject mesh shapes
        # the current transformers version cannot honor.
        self._tp_size = device_mesh.get_dim_size('tp') if (device_mesh is not None and device_mesh.has_dim('tp')) else 1
        self._tp_enabled = self._tp_size > 1
        if self._tp_enabled:
            self._validate_tp_device_mesh(device_mesh)
        parallelism_config = self._parallelism_config_from_device_mesh(device_mesh)
        fsdp_plugin = self._fsdp_config_from_device_mesh(device_mesh, fsdp_config, memory_efficient_init)

        kwargs_handlers = [
            InitProcessGroupKwargs(
                timeout=timedelta(seconds=int(os.environ.get('TWINKLE_DIST_TIMEOUT_SECONDS', '7200'))))
        ]
        if ddp_config is not None:
            from accelerate import DistributedDataParallelKwargs
            ddp_config = DistributedDataParallelKwargs(**ddp_config)
            kwargs_handlers.append(ddp_config)

        self.accelerator = Accelerator(
            parallelism_config=parallelism_config,
            mixed_precision=mixed_precision,
            fsdp_plugin=fsdp_plugin,
            kwargs_handlers=kwargs_handlers,
        )

    def pretrained_load_context(self):
        return fsdp_pretrained_load_context(self._memory_efficient_init and self.device_mesh is not None)

    def init_kwargs(self) -> Dict[str, Any]:
        """Extra kwargs the model construction call needs for this strategy.

        For TP this is what makes transformers shard the model at load time. accelerate does not
        apply TP itself: `Accelerator.prepare` only checks that the model was already sharded
        (`model.tp_size == parallelism_config.tp_size`) and replicates whatever is left on its own
        device mesh. We also hand transformers accelerate's own device mesh so both sides operate on
        a single mesh -- otherwise the sharded and the replicated params end up on different meshes
        and forward fails with a cross-mesh DTensor error.
        """
        if not self._tp_enabled:
            return {}
        load_kwargs = self._tp_load_config(self._tp_size)
        device_mesh = self.accelerator.torch_device_mesh
        if device_mesh is not None:
            load_kwargs['device_mesh'] = device_mesh
        return load_kwargs

    @staticmethod
    def _tp_load_config(tp_size: int) -> Dict[str, Any]:
        """Version-adaptive native-TP entry for `from_pretrained`.

        transformers >=5.16 shards via `DistributedConfig(tp_size=...)`; <=5.12 via `tp_plan="auto"`
        plus `tp_size`. Tell them apart by whether `DistributedConfig` carries a `tp_size` field.
        """
        try:
            from transformers.distributed import DistributedConfig
            if 'tp_size' in getattr(DistributedConfig, '__dataclass_fields__', {}):
                return {'distributed_config': DistributedConfig(tp_size=tp_size)}
        except ImportError:
            pass
        return {'tp_plan': 'auto', 'tp_size': tp_size}

    @staticmethod
    def _validate_tp_device_mesh(device_mesh) -> None:
        """Fail fast on TP mesh shapes this integration does not yet support.

        v1 of transformers-backend TP handles pure tensor parallelism only (`tp_size ==
        world_size`); composing TP with FSDP/DP is not validated yet. So a TP mesh must not carry
        any other non-trivial parallel dimension.
        """
        offending = {}
        for name in (device_mesh.mesh_dim_names or ()):
            if name == 'tp':
                continue
            if device_mesh.get_dim_size(name) > 1:
                offending[name] = device_mesh.get_dim_size(name)
        if offending:
            raise ValueError('Tensor parallelism in the transformers backend currently supports pure TP only '
                             '(composing TP with FSDP/DP is not validated yet), but the device mesh also has '
                             f'non-trivial dimensions {offending}. Use a mesh where only `tp` > 1.')
        tp_size = device_mesh.get_dim_size('tp')
        if tp_size != device_mesh.world_size:
            raise ValueError(f'Pure TP requires tp_size ({tp_size}) to equal the mesh world size '
                             f'({device_mesh.world_size}).')

    @staticmethod
    def _parallelism_config_from_device_mesh(device_mesh: DeviceMesh):
        # TODO should test with transformers v5.0
        from accelerate import ParallelismConfig
        if device_mesh is None:
            return None

        dp_size = device_mesh.get_dim_size('dp') if device_mesh.has_dim('dp') else 1
        fsdp_size = device_mesh.get_dim_size('fsdp') if device_mesh.has_dim('fsdp') else 1
        tp_size = device_mesh.get_dim_size('tp') if device_mesh.has_dim('tp') else 1
        cp_size = device_mesh.get_dim_size('cp') if device_mesh.has_dim('cp') else 1
        sp_size = device_mesh.get_dim_size('sp') if device_mesh.has_dim('sp') else 1

        if tp_size == 1 and cp_size == 1 and sp_size == 1:
            # Only ddp
            return None

        parallelism_config = ParallelismConfig(
            dp_replicate_size=dp_size,
            dp_shard_size=fsdp_size,
            tp_size=tp_size,
            cp_size=cp_size,
            sp_size=sp_size,
        )

        return parallelism_config

    def _fsdp_config_from_device_mesh(self, device_mesh: DeviceMesh, fsdp_config: Dict[str, Any],
                                      memory_efficient: bool):
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp import BackwardPrefetch
        from torch.distributed.fsdp import ShardingStrategy as FSDPShardingStrategy

        if device_mesh is None:
            return None

        fsdp_size = device_mesh.get_dim_size('fsdp') if device_mesh.has_dim('fsdp') else 1
        dp_size = device_mesh.get_dim_size('dp') if device_mesh.has_dim('dp') else 1

        if fsdp_size == 1:
            return None

        fsdp_config = fsdp_config or {}

        requested_strategy = fsdp_config.pop('sharding_strategy', None)
        if dp_size > 1:
            # HSDP
            hybrid_strategies = (FSDPShardingStrategy.HYBRID_SHARD, FSDPShardingStrategy._HYBRID_SHARD_ZERO2)
            sharding_strategy = requested_strategy if requested_strategy in hybrid_strategies else \
                FSDPShardingStrategy.HYBRID_SHARD
        else:
            # FSDP
            sharding_strategy = FSDPShardingStrategy.FULL_SHARD
        if requested_strategy is not None and requested_strategy is not sharding_strategy:
            logger.warning(
                f'fsdp_config sharding_strategy={requested_strategy} is overridden with {sharding_strategy} to '
                f'match the device mesh (dp={dp_size}, fsdp={fsdp_size}).')

        fsdp_version = fsdp_config.pop('fsdp_config', 2)
        assert fsdp_version == 2, 'Currently only support fsdp_version = 2'
        fsdp_plugin = FullyShardedDataParallelPlugin(
            fsdp_version=fsdp_version,
            sharding_strategy=sharding_strategy,
            backward_prefetch=fsdp_config.pop('backward_prefetch', BackwardPrefetch.BACKWARD_PRE),
            mixed_precision_policy=self.mixed_precision,
            cpu_offload=fsdp_config.pop('cpu_offload', False),
            activation_checkpointing=fsdp_config.pop('activation_checkpointing', False),
            auto_wrap_policy=fsdp_config.pop('auto_wrap_policy', 'transformer_based_wrap'),  # noqa
            reshard_after_forward=fsdp_config.pop('reshard_after_forward', True),
            cpu_ram_efficient_loading=fsdp_config.pop('cpu_ram_efficient_loading', memory_efficient),
            **fsdp_config,
        )
        return fsdp_plugin

    def offload_to_cpu(self, model, optimizer=None, *, offload_model: bool = True) -> None:
        """Move parameters/buffers and/or optimizer state to host memory for colocation.

        Under colocation the trainer shares its GPU with a vLLM sampler. ``offload_model=False``
        leaves the module resident and moves only ``optimizer`` when one is passed. Only placement
        changes -- nothing is discarded. The device moved off is recorded so the reverse lands
        exactly where the params were. Re-calling either direction is harmless: ``.to`` on an
        already-resident tensor is a no-op, and :meth:`reload_to_gpu` restores only what moved.
        Passing neither piece still releases the caching allocator before return.
        """
        import torch
        from twinkle.model.offload import move_optimizer_state, record_offload, release_caching_allocator
        if not offload_model and optimizer is None:
            record_offload(self, model=False, optimizer=False)
            release_caching_allocator()
            return
        if getattr(self, '_offload_device', None) is None:
            self._offload_device = self.accelerator.device
        if offload_model:
            self.unwrap_model(model).to('cpu')
        if optimizer is not None:
            move_optimizer_state(optimizer, torch.device('cpu'))
        record_offload(self, model=offload_model, optimizer=optimizer is not None)
        # Release reserved blocks on every backend before the colocated sampler wakes.
        release_caching_allocator()

    def reload_to_gpu(self, model, optimizer=None) -> None:
        """Bring back what :meth:`offload_to_cpu` handed over, to the device it was moved off."""
        from twinkle.model.offload import clear_offload_record, move_optimizer_state, offload_record
        moved_model, moved_optimizer = offload_record(self)
        if not moved_model and not moved_optimizer:
            return
        device = getattr(self, '_offload_device', None) or self.accelerator.device
        if moved_model:
            self.unwrap_model(model).to(device)
        if moved_optimizer and optimizer is not None:
            move_optimizer_state(optimizer, device)
        clear_offload_record(self)

    @staticmethod
    def _duplicate_names(model):
        """Map each name that shares its parameter object with an earlier name to that earlier name.

        ``named_parameters(remove_duplicate=False)`` lists a tied weight under every name that
        references it, so the duplicates are exactly the names a tie is hiding.
        """
        seen, duplicate = {}, {}
        for fqn, param in model.named_parameters(remove_duplicate=False):
            if id(param) in seen:
                duplicate[fqn] = seen[id(param)]
            else:
                seen[id(param)] = fqn
        return duplicate

    def wrap_model(self, model, *args):
        import torch

        duplicates = self._duplicate_names(model)
        plugin = self._get_fsdp_plugin()
        if duplicates and plugin is not None and getattr(plugin, 'fsdp_version', 1) == 2:
            self._untie_shared_weights(model, [obj for obj in args if isinstance(obj, torch.optim.Optimizer)],
                                       duplicates)
        result = self.accelerator.prepare(model, *args)
        # ``Accelerator.prepare`` ends with ``return result if len(result) > 1 else result[0]``, and
        # ``result`` is a list, so the container's type says nothing about what was prepared and a
        # lone argument comes back bare. Testing for a tuple here silently skipped the repair.
        prepared = result if isinstance(result, (tuple, list)) else (result,)
        for optimizer in prepared[1:]:
            if isinstance(optimizer, torch.optim.Optimizer):
                # Repair first, then record: the map has to describe an optimizer whose entries are
                # the model's own objects, and an argument would be evaluated before the call.
                self._rebind_unsharded_optimizer_params(
                    prepared[0], optimizer, self.__dict__.get('_optimizer_param_names'))
                self._remember_optimizer_params(prepared[0], optimizer)
        return result

    @staticmethod
    def _untie_shared_weights(model, optimizers, duplicates):
        """Give each name of a tied weight its own parameter, before sharding splits them anyway.

        ``fully_shard`` converts every module's parameters on its own, so a weight two modules share
        comes out of ``prepare`` as two objects whether or not anything asked for it. Left tied, the
        optimizer -- built over the tied view -- keeps only one of them, so the other stays at its
        pretrained value while the forward pass computes with it. The checkpoint then carries a head
        that differs from the embedding it claims to be tied to, and a reload that honors
        ``tie_word_embeddings`` binds the head back to the embedding and throws the trained one away:
        the model that resumes is not the model that trained. Declaring the untie up front makes both
        weights real, trainable and saveable, and the config written beside them agrees with what is
        on disk -- the resolution accelerate's own tied-embedding error message advises.

        The new parameter joins the group that holds its partner, so it inherits the partner's
        learning rate and weight decay, which is what the tie would have implied.
        """
        import torch

        groups = []
        for optimizer in optimizers:
            inner = getattr(optimizer, 'optimizer', optimizer)
            groups.extend(getattr(inner, 'param_groups', None) or [])
        by_name = dict(model.named_parameters(remove_duplicate=False))
        for fqn, partner_name in duplicates.items():
            param = by_name.get(fqn)
            if param is None or not param.requires_grad:
                # A frozen duplicate is never stepped and never saved as optimizer state, so leaving
                # it shared costs nothing and splitting it would only spend its memory twice.
                continue
            parent, _, attribute = fqn.rpartition('.')
            module = model.get_submodule(parent) if parent else model
            fresh = torch.nn.Parameter(param.detach().clone(), requires_grad=True)
            setattr(module, attribute, fresh)
            partner = by_name.get(partner_name)
            group = next((group for group in groups if partner is not None
                          and any(item is partner for item in group['params'])), None)
            if group is None and groups:
                group = groups[0]
            if group is not None:
                group['params'].append(fresh)
                # ``param_names`` is the parallel list of FQNs that the checkpoint's state-dict
                # mapping is built from. Appending the parameter without its name leaves the two
                # lists different lengths inside the saved artifact -- measured on a 4B run: group 0
                # held 398 params against 397 names -- and anything that pairs them by position then
                # associates the wrong name with the wrong tensor. That is how the untied head ends
                # up under a different weight_decay group on resume than it had in the run that
                # produced the checkpoint, which changes the update without changing the forward.
                names = group.get('param_names')
                if isinstance(names, list):
                    names.append(fqn)
        config = getattr(model, 'config', None)
        for holder in (config, getattr(config, 'text_config', None)):
            if holder is not None and getattr(holder, 'tie_word_embeddings', False):
                holder.tie_word_embeddings = False

    def _remember_optimizer_params(self, model, optimizer):
        """Record the name each optimizer parameter had while the model and the optimizer agreed.

        Right after ``prepare`` every optimizer parameter IS one of the model's own objects, so
        ``named_parameters`` gives each entry an exact name. A later mixed-precision cast can replace
        a parameter object again, and for a tied weight it replaces the single object the tie covered
        with one fresh object per name -- the embedding and the tied head then both sit outside the
        optimizer with the SAME shape, so shape alone cannot say which of them the stale entry meant.
        The name recorded here can, which is why the save-time repair is handed this map.
        """
        ids = {id(param) for group in optimizer.param_groups for param in group['params']}
        names = {id(param): fqn for fqn, param in model.named_parameters() if id(param) in ids}
        self.__dict__['_optimizer_param_names'] = names
        return names

    @staticmethod
    def _rebind_unsharded_optimizer_params(model, optimizer, param_names=None) -> None:
        """Point the optimizer at the model's own objects for the parameters FSDP2 left unsharded.

        Casting a parameter to the mixed-precision compute dtype replaces the ``Parameter`` object,
        and ``accelerator.prepare`` rebuilds the optimizer over the sharded (DTensor) parameters --
        but the parameters that sit OUTSIDE every ``fully_shard`` unit (a vision tower, and the
        embedding/tail that accelerate carves into their own units) keep their pre-wrap objects in
        the optimizer's ``param_groups``. Training then steps the model's objects while the optimizer
        still names stale ones, which is invisible until a checkpoint asks torch to map optimizer
        state onto parameter FQNs BY IDENTITY and the save dies -- after the whole run has trained.

        This runs before the first optimizer step, so no state exists yet and re-pointing an object
        loses nothing; doing it any later would have to migrate Adam moments across dtypes.

        The pairing is taken only when it is provable, shape bucket by shape bucket. The offenders are
        the optimizer parameters the wrapped model does not expose; the candidate replacements are the
        model's plain (non-DTensor) trainable parameters that are NOT already in the optimizer -- the
        set FSDP2 left alone, which is exactly the set that went stale, minus the members the wrap did
        rebind. Within one shape bucket the pairing is by traversal order, so a shape occurring once is
        matched exactly and a shape occurring twice (an untied embedding and lm_head) is matched by
        order. A bucket whose two sides differ in count is left alone, and so is every other bucket:
        the save-time guard then still refuses loudly rather than guessing which weight an optimizer
        moment belongs to.
        """
        from torch.distributed.tensor import DTensor

        inner = getattr(optimizer, 'optimizer', optimizer)
        param_groups = getattr(inner, 'param_groups', None)
        if not param_groups:
            return
        model_params = list(model.named_parameters())
        optimizer_ids = {id(param) for group in param_groups for param in group['params']}
        stale = [param for group in param_groups for param in group['params'] if id(param) not in
                 {id(owned) for _, owned in model_params}]
        if not stale:
            return
        named = dict(model_params)
        replacement = {}
        for param in stale:
            fqn = (param_names or {}).get(id(param))
            current = named.get(fqn)
            # The name is exact where the shape is not: a tied weight's two names share one shape, so
            # only the recorded name says which of them the stale entry stood for. The target must
            # still be a weight the optimizer does not already hold, or the repair would duplicate.
            if (current is not None and current.requires_grad and not isinstance(current, DTensor)
                    and id(current) not in optimizer_ids):
                replacement[id(param)] = current

        # Whatever the names did not settle falls back to the shape rule, over the offenders still
        # unmatched and the candidates the names did not consume.
        remaining = [param for param in stale if id(param) not in replacement]
        taken = {id(param) for param in replacement.values()}
        if remaining:
            candidates = [
                param for _, param in model_params
                if param.requires_grad and not isinstance(param, DTensor)
                and id(param) not in optimizer_ids and id(param) not in taken
            ]

            def buckets(params):
                grouped = {}
                for param in params:
                    grouped.setdefault(tuple(param.shape), []).append(param)
                return grouped

            stale_buckets, candidate_buckets = buckets(remaining), buckets(candidates)
            for shape, group in stale_buckets.items():
                peers = candidate_buckets.get(shape)
                # Only a bucket that matches one-for-one is provable; a partial match would leave some
                # offender pointing at a weight that belongs to another offender.
                if peers is None or len(peers) != len(group):
                    continue
                replacement.update(zip((id(param) for param in group), peers))
        if not replacement:
            return
        for group in param_groups:
            group['params'] = [replacement.get(id(param), param) for param in group['params']]
        # Adam's moments are keyed by parameter OBJECT, so repointing the groups alone orphans them,
        # and torch's Optimizer.state_dict() then remaps every state key through the groups and dies
        # with a bare KeyError on the orphan's id. Both rules feed the same migration. Every offender
        # is a parameter FSDP2 left unsharded, hence full-shaped on both sides and differing only in
        # dtype -- and the moments are fp32 whatever the parameter dtype is -- so the state carries
        # over by identity. Empty at wrap time, which makes this a no-op there.
        state = getattr(inner, 'state', None)
        if state:
            for old in [key for key in list(state) if id(key) in replacement]:
                state[replacement[id(old)]] = state.pop(old)

    def unwrap_model(self, model):
        return self.accelerator.unwrap_model(model, keep_torch_compile=False)

    def load_peft_weights(self, model, adapter_weights: Mapping[str, Any], adapter_name: str) -> None:
        from peft.utils import set_peft_model_state_dict

        # ``set_peft_model_state_dict`` copies into the parameters' EXISTING dtype, so a float32
        # adapter checkpoint loaded while the LoRA modules are still bfloat16 is rounded on the way in
        # -- the same truncation ``_match_master_weight_precision`` documents for the full-parameter
        # path, reaching the adapter through a different entry point. Upcast first so the copy is
        # lossless and the wrap's later upcast becomes a no-op.
        self._match_master_weight_precision(model)
        self._probe_load_precision(model, adapter_weights, after=False)
        load_result = set_peft_model_state_dict(model, adapter_weights, adapter_name=adapter_name)
        self._probe_load_precision(model, adapter_weights, after=True)
        # A key the checkpoint holds but the model does not (or the reverse) is dropped silently by
        # peft, leaving the resumed adapter short of the source run's -- so report rather than discard.
        # ``missing_keys`` is peft's "every model key the state dict did not supply", which for an
        # adapter load is the entire frozen base; only the adapter's own keys being absent is a defect.
        unexpected = list(getattr(load_result, 'unexpected_keys', []) or [])
        absent = [
            key for key in (getattr(load_result, 'missing_keys', []) or [])
            if 'lora_' in key or key in adapter_weights
        ]
        logger.info(f'Loaded adapter "{adapter_name}": {len(adapter_weights)} tensors, '
                    f'{len(absent)} adapter keys absent, {len(unexpected)} unexpected.')
        if absent:
            logger.warning(f'Adapter "{adapter_name}" is missing {len(absent)} of its own keys, '
                           f'first few: {absent[:5]}. The resumed adapter will not match its source run.')
        if unexpected:
            logger.warning(f'Adapter "{adapter_name}" checkpoint carries {len(unexpected)} keys the model '
                           f'does not have, first few: {unexpected[:5]}. They were dropped.')

    def _get_fsdp_plugin(self):
        state = self.accelerator.state
        return state.fsdp_plugin if hasattr(state, 'fsdp_plugin') else None

    @staticmethod
    def _prepare_full_optimizer_state_dict_options(*, for_load: bool):
        from torch.distributed.checkpoint.state_dict import StateDictOptions

        return StateDictOptions(
            full_state_dict=True,
            cpu_offload=not for_load,
            broadcast_from_rank0=for_load,
        )

    def needs_wrapped_optimizer_state(self) -> bool:
        fsdp_plugin = self._get_fsdp_plugin()
        return fsdp_plugin is not None and fsdp_plugin.fsdp_version == 2

    @staticmethod
    def _check_optimizer_params_reachable(model, optimizer) -> None:
        """Raise before torch does, naming the parameters its FSDP2 optimizer save cannot map.

        ``get_optimizer_state_dict`` maps optimizer state onto parameter FQNs by object IDENTITY:
        torch's ``_get_optim_state_dict`` zips the live ``param_groups`` into a pid table, looks every
        ``model.named_parameters()`` entry up in it, and then indexes that table with each pid of the
        optimizer's own state. A parameter the wrapped model does not expose is skipped by the lookup
        but still present in the state, so the save dies as a bare ``KeyError: <pid>`` naming neither
        the parameter nor the cause.

        FSDP2 makes that reachable: accelerate carves the embedding/tail into their own units and
        switches the optimizer onto the sharded parameters, so a parameter the model still holds
        unsharded has an optimizer entry pointing at a different object. Only a final checkpoint asks
        for optimizer state, which is why such a run trains to completion and fails at the very end.
        """
        from torch.distributed.tensor import DTensor

        # An optimizer this strategy cannot introspect has nothing to check; the repair tolerates the
        # same shape, and torch's own get_optimizer_state_dict still refuses it in production.
        param_groups = getattr(optimizer, 'param_groups', None)
        if not param_groups:
            return
        params = [param for group in param_groups for param in group['params']]
        model_params = list(model.named_parameters())
        model_ids = {id(param) for _, param in model_params}
        unreachable = [(index, param) for index, param in enumerate(params) if id(param) not in model_ids]
        # The other direction: a moment whose parameter no longer sits in any group. torch's
        # Optimizer.state_dict() remaps every state key through the groups, so an orphan dies there
        # with a bare KeyError on its id -- the same unmappable state, reached from the other side.
        group_ids = {id(param) for param in params}
        orphans = [key for key in getattr(optimizer, 'state', {}) if id(key) not in group_ids]
        if not unreachable and not orphans:
            return
        unsharded = [fqn for fqn, param in model_params if not isinstance(param, DTensor)]
        # The other half of the divergence: a parameter the model trains but the optimizer never
        # received. Reported beside the offenders because the two counts are what tell the reader
        # whether the optimizer merely holds stale objects (offenders only) or also lost track of
        # weights (offenders plus missing), which is a different and worse failure.
        optimizer_ids = {id(param) for param in params}
        missing = [fqn for fqn, param in model_params
                   if param.requires_grad and id(param) not in optimizer_ids]
        # Name each offender two ways, because either alone can mislead:
        #   - BY POSITION: an optimizer is built over the model's trainable parameters in order, so the
        #     i-th optimizer parameter and the i-th trainable FQN are the same weight even when the
        #     wrap replaced the object. This identifies the weight exactly.
        #   - BY SHAPE: the fallback when the counts have diverged (a group was dropped or rebuilt),
        #     plus the dtype and DTensor-ness pair that says whether the optimizer holds a pre-wrap
        #     copy (plain fp32) of what the model now exposes as a shard (bf16 DTensor).
        trainable = [fqn for fqn, param in model_params if param.requires_grad]
        by_shape: dict = {}
        for fqn, param in model_params:
            by_shape.setdefault(tuple(param.shape), []).append(fqn)
        offenders = []
        for index, param in unreachable[:8]:
            positional = trainable[index] if index < len(trainable) else '<position past the trainable list>'
            model_param = dict(model_params).get(positional)
            # What the rebind rule saw for this shape: how many model parameters share it, and how
            # many of those the optimizer already names. A bucket is repaired only when the offenders
            # and the not-yet-named parameters of that shape are equal in number, so these two counts
            # are what decide whether a remaining offender was unprovable or merely unmatched.
            same_shape = [fqn for fqn, other in model_params if tuple(other.shape) == tuple(param.shape)]
            named = sum(1 for fqn, other in model_params
                        if tuple(other.shape) == tuple(param.shape) and id(other) in optimizer_ids)
            offenders.append(
                f'#{index} {positional}: optimizer holds {tuple(param.shape)} {param.dtype}; '
                f'model exposes '
                f'{"nothing under that name" if model_param is None else f"{tuple(model_param.shape)} {model_param.dtype}" + (" (DTensor)" if isinstance(model_param, DTensor) else " (plain)")}'
                f'; same-shape model params {same_shape or ["<none>"]} '
                f'({named} of them already named by the optimizer)')
        raise RuntimeError(
            f'{len(orphans)} optimizer state entr(y/ies) belong to a parameter no group holds any '
            f'more, and {len(unreachable)} of {len(params)} optimizer parameter(s) are not reachable '
            f'from the wrapped '
            f'model {type(model).__name__} (which exposes {len(trainable)} trainable parameter(s)), so its '
            f'FSDP2 optimizer state cannot be mapped to parameter names and no resumable checkpoint can be '
            f'written. Offenders: {offenders}. The model trains {len(missing)} parameter(s) the '
            f'optimizer never received: {missing[:12]}. The model still holds '
            f'{len(unsharded)} unsharded parameter(s): {unsharded[:12]}. This is the full-parameter FSDP2 '
            'path; an adapter run (--tuner lora) keeps every optimizer parameter inside a wrapped unit '
            'and saves normally.')

    def save_optimizer_checkpoint(self, model, optimizer, output_path: str, *, param_name_mapping=None):
        import torch

        from twinkle.model.state_dict_utils import remap_optimizer_state_names
        fsdp_plugin = self._get_fsdp_plugin()
        if fsdp_plugin is not None and fsdp_plugin.fsdp_version == 2:
            from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

            # PyTorch initializes Adam state through the native optimizer step.
            # AcceleratedOptimizer may skip that step when gradients are not synchronized.
            inner_optimizer = getattr(optimizer, 'optimizer', optimizer)
            # The wrap-time repair can come too early: FSDP2's mixed precision replaces a parameter
            # object when it casts it, and that cast does not necessarily happen inside prepare. So
            # repair again here, where the divergence is observable, before torch maps state by
            # identity. Nothing to do when the wrap already covered it.
            self._rebind_unsharded_optimizer_params(
                model, inner_optimizer, self.__dict__.get('_optimizer_param_names'))
            self._check_optimizer_params_reachable(model, inner_optimizer)
            optim_state = get_optimizer_state_dict(
                model,
                inner_optimizer,
                options=self._prepare_full_optimizer_state_dict_options(for_load=False),
            )
            if self.accelerator.process_index == 0:
                remap_optimizer_state_names(optim_state, param_name_mapping or {})
                torch.save(optim_state, output_path)
            return

        if self.accelerator.process_index == 0:
            optim_state = optimizer.state_dict()
            remap_optimizer_state_names(optim_state, param_name_mapping or {})
            torch.save(optim_state, output_path)

    def load_optimizer_checkpoint(self, model, optimizer, input_path: str, *, param_name_mapping=None):
        import torch

        from twinkle.model.state_dict_utils import remap_optimizer_state_names
        fsdp_plugin = self._get_fsdp_plugin()
        if fsdp_plugin is not None and fsdp_plugin.fsdp_version == 2:
            from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict

            inner_optimizer = getattr(optimizer, 'optimizer', optimizer)
            optim_state = {}
            if self.accelerator.process_index == 0:
                optim_state = torch.load(input_path, map_location='cpu', weights_only=True)
                remap_optimizer_state_names(optim_state, param_name_mapping or {})
            set_optimizer_state_dict(
                model,
                inner_optimizer,
                optim_state,
                options=self._prepare_full_optimizer_state_dict_options(for_load=True),
            )
            probe = os.environ.get('TWINKLE_OPT_LOAD_PROBE')
            if probe:
                self._probe_loaded_optimizer_state(model, inner_optimizer, input_path, probe)
            return

        # weights_only=True: optimizer state is tensors + plain containers (verified), so a
        # hostile checkpoint's pickle payload never reaches an unrestricted unpickler.
        optim_state = torch.load(input_path, map_location='cpu', weights_only=True)
        remap_optimizer_state_names(optim_state, param_name_mapping or {})
        optimizer.load_state_dict(optim_state)

    def _probe_loaded_optimizer_state(self, model, optimizer, checkpoint_path, path) -> None:
        """Compare the moments in the live optimizer against the ones the checkpoint holds.

        Both sides must be reduced to the same granularity before any number means anything: the
        checkpoint is a FULL state dict, while an FSDP2 parameter -- and therefore its Adam moments
        -- is a DTensor whose ``_local_tensor`` is only this rank's shard. Comparing a shard's L1
        norm against the full tensor's says nothing, so each saved tensor is redistributed into the
        live DTensor's own placement and the two shards are compared element by element.
        """
        import json

        import torch
        from torch.distributed.tensor import DTensor, distribute_tensor

        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        saved_state = saved.get('state') or {}
        state = getattr(optimizer, 'state', {})
        rank = self.accelerator.process_index
        detail = {}
        worst = {'name': None, 'key': None, 'diff': 0.0}
        checked = 0
        watched = ('model.language_model.embed_tokens.weight', 'lm_head.weight',
                   'model.language_model.norm.weight', 'model.visual.patch_embed.proj.weight')
        for name, param in model.named_parameters():
            live, saved = state.get(param), saved_state.get(name)
            if live is None or saved is None:
                continue
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                live_value, saved_value = live.get(key), saved.get(key)
                if live_value is None or saved_value is None:
                    continue
                if isinstance(live_value, DTensor):
                    # Every rank runs this loop over the same names, so the collectives inside
                    # redistribute match up; a rank that skipped one would deadlock the others.
                    target = distribute_tensor(saved_value.to(dtype=live_value.dtype,
                                                              device=live_value.device),
                                               live_value.device_mesh, live_value.placements)
                    diff = float((target.to_local() - live_value.to_local()).abs().max())
                else:
                    diff = float((saved_value.to(dtype=live_value.dtype, device=live_value.device)
                                  - live_value).abs().max())
                checked += 1
                if diff > worst['diff']:
                    worst = {'name': name, 'key': key, 'diff': diff}
                if name in watched:
                    detail[name] = detail.get(name, {})
                    detail[name][key] = {
                        'diff': diff,
                        'live_l1': float(live_value.to_local().abs().sum()) if isinstance(live_value, DTensor)
                        else float(live_value.abs().sum()),
                        'saved_l1': float(saved_value.abs().sum()),
                        'is_dtensor': isinstance(live_value, DTensor),
                    }
        payload = {
            'rank': rank,
            'group_params': sum(len(group['params']) for group in optimizer.param_groups),
            'state_entries': len(state),
            'group_params_without_state': sum(
                1 for group in optimizer.param_groups for param in group['params'] if param not in state),
            'saved_state_entries': len(saved_state),
            'tensors_compared': checked,
            'worst': worst,
            # The moments are only half of what an Adam step consumes: the update is also a function
            # of the group's hyperparameters, and those are rebuilt from the config on resume rather
            # than read back from the checkpoint. ``lr`` is visible in the per-step log; weight_decay,
            # betas and eps are not, so a group rebuilt with different values would leave the loss and
            # grad_norm of the first resumed step matching while every later weight diverges.
            'groups': [{k: v for k, v in group.items() if k != 'params'}
                       for group in optimizer.param_groups],
            'group_sizes': [len(group['params']) for group in optimizer.param_groups],
            'saved_groups': [{k: v for k, v in group.items() if k != 'params'}
                             for group in (saved.get('param_groups') or [])],
            'saved_group_sizes': [len(group.get('params') or [])
                                  for group in (saved.get('param_groups') or [])],
            'detail': detail,
        }
        with open(f'{path}.rank{rank}.json', 'w') as handle:
            json.dump(payload, handle, indent=1)

    def get_full_state_dict(self, model) -> dict:
        """Collect full state dict, including persistent buffers."""
        from twinkle.utils import torch_util
        unwrapped = self.unwrap_model(model)
        state = collect_local_state(unwrapped, include_buffers=True)
        return {name: torch_util.to_local_tensor(tensor).cpu() for name, tensor in state.items()}

    def _match_master_weight_precision(self, model) -> None:
        """Put trainable parameters in the dtype the FSDP2 wrap will train them at, before a load.

        ``accelerate.utils.fsdp_utils`` upcasts every trainable parameter to a float32 master when
        mixed precision is on -- ``MixedPrecisionPolicy.param_dtype`` then supplies the
        low-precision compute copy -- and it does that inside ``prepare``, which runs AFTER this
        load. Loading a float32 checkpoint into parameters that are still bfloat16 therefore rounds
        away every bit the source run had trained past bf16 precision, and the run continues from a
        master that never held them.

        The forward cannot see this: it casts to bf16 either way, so the first resumed step matches
        the source run's loss and grad_norm exactly, and so do its clipped gradients. The next
        optimizer update lands on the rounded master, whose bf16 compute cast can now cross a
        rounding boundary, so the trajectory diverges from the following step onward -- a resume
        that looks correct for one step and then drifts. Upcasting first makes the copy lossless
        and turns the wrap's own upcast into a no-op.

        The quantized guard mirrors accelerate's: it skips its upcast entirely when the model
        carries 4-bit parameters, so this must not upcast them either.
        """
        import torch

        if self.accelerator.mixed_precision == 'no':
            return
        if any(hasattr(param, 'quant_state') for param in model.parameters()):
            return
        for param in model.parameters():
            if param.requires_grad and param.is_floating_point() and param.dtype != torch.float32:
                param.data = param.data.to(torch.float32)

    def load_full_state_dict(self, model, state_dict) -> None:
        """Load a full (non-sharded) state dict into the model in-place.

        Used by full-parameter training to (re)load base weights, e.g. after a
        tenant releases an exclusive full-parameter deployment.
        """
        fsdp_plugin = self._get_fsdp_plugin()
        if fsdp_plugin is not None and fsdp_plugin.fsdp_version == 2:
            from torch.distributed.checkpoint.state_dict import set_model_state_dict

            from twinkle.model.state_dict_utils import with_tied_weight_aliases
            self._match_master_weight_precision(model)
            self._probe_load_precision(model, state_dict, after=False)
            set_model_state_dict(
                model,
                with_tied_weight_aliases(self.unwrap_model(model), state_dict),
                options=self._prepare_full_optimizer_state_dict_options(for_load=True),
            )
            self._probe_load_precision(model, state_dict, after=True)
            return
        self._probe_load_precision(model, state_dict, after=False)
        unwrapped = self.unwrap_model(model)
        unwrapped.load_state_dict(state_dict, strict=False)
        self._probe_load_precision(model, state_dict, after=True)

    def get_adapter_state_dict(self, model, adapter_name: str) -> dict:
        """Collect LoRA parameters, trained modules_to_save copies, and their buffers."""
        from twinkle.utils import torch_util
        unwrapped = self.unwrap_model(model)
        adapter_suffix = f'.{adapter_name}.'
        state = collect_local_state(unwrapped, include_buffers=True)
        return {
            name: torch_util.to_local_tensor(tensor).cpu()
            for name, tensor in state.items()
            if is_saveable_adapter_key(name) and adapter_suffix in name
        }


def _move_optimizer_state(optimizer, device) -> None:
    """Move an optimizer's per-parameter state tensors (Adam moments, etc.) to ``device``."""
    from twinkle.model.offload import move_optimizer_state
    move_optimizer_state(optimizer, device)
