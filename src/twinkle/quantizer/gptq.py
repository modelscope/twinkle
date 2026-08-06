# Copyright (c) ModelScope Contributors. All rights reserved.
from collections import defaultdict
from contextlib import contextmanager
from typing import List, Optional

from twinkle.quantizer.base import CalibrationQuantizer
from twinkle.utils import get_logger, requires

logger = get_logger()


def _deep_getattr(obj, attr: str):
    """getattr through a dotted path ('model.layers' -> obj.model.layers)."""
    for a in attr.split('.'):
        obj = getattr(obj, a)
    return obj


class GptqQuantizer(CalibrationQuantizer):
    """GPTQ via optimum's ``GPTQQuantizer``.

    Unlike AWQ this takes a plain transformers model. The quantizer object itself owns the writer
    (it has to persist the packing metadata), so it is kept on ``self.gptq_quantizer`` after
    ``quantize`` and reused by ``save``.

    Set ``v2=True`` for the gptq_v2 checkpoint format.
    """

    quant_method = 'gptq'

    def __init__(self,
                 quant_bits: Optional[int] = 4,
                 group_size: int = 128,
                 batch_size: int = 1,
                 modules_to_not_convert: Optional[List[str]] = None,
                 v2: bool = False,
                 block_name_to_quantize: Optional[str] = None,
                 modules_in_block_to_quantize: Optional[List[List[str]]] = None,
                 is_moe_model: bool = False,
                 model_type: Optional[str] = None,
                 **kwargs):
        requires('optimum')
        # optimum delegates the actual packing to gptqmodel; without it GPTQQuantizer.__init__
        # itself raises NameError on QuantizeConfig, long before any weight is touched.
        requires('gptqmodel')
        super().__init__(
            quant_bits=quant_bits,
            group_size=group_size,
            batch_size=batch_size,
            modules_to_not_convert=modules_to_not_convert,
            **kwargs)
        self.v2 = v2
        self.block_name_to_quantize = block_name_to_quantize
        self.modules_in_block_to_quantize = modules_in_block_to_quantize
        self.is_moe_model = is_moe_model
        self.model_type = model_type
        self.gptq_quantizer = None

    @property
    def checkpoint_format(self) -> str:
        return 'gptq_v2' if self.v2 else 'gptq'

    @staticmethod
    def get_block_name_to_quantize(model, language_model_prefix: Optional[str] = None) -> Optional[str]:
        """Locate the transformer-block ModuleList GPTQ should walk.

        Picks the longest ModuleList/Sequential of at least 10 entries, skipping MoE expert lists
        (whose element class name contains 'mlp') so the blocks -- not the experts -- are found.
        ``language_model_prefix`` scopes the search into the LLM of a multimodal model.
        """
        import torch.nn as nn

        prefix = ''
        if language_model_prefix:
            prefix = language_model_prefix
            model = _deep_getattr(model, prefix)

        module_lists = []
        for n, m in model.named_modules():
            if (isinstance(m, (nn.ModuleList, nn.Sequential)) and len(m) >= 10
                    and 'mlp' not in m[0].__class__.__name__.lower()):  # fix moe
                module_lists.append((n, m))
        if module_lists:
            module_list = max(module_lists, key=lambda x: len(x[1]))
            return f'{prefix}.{module_list[0]}'.strip('.')

    @staticmethod
    def _get_experts(block):
        import torch.nn as nn

        for n, m in block.named_modules():
            if isinstance(m, (nn.ModuleList, nn.Sequential)):
                return n, m
        return None, None

    def get_modules_in_block_to_quantize(self, model, block_name: str) -> Optional[List[List[str]]]:
        """Group a block's linears for MoE models.

        Only meaningful for MoE: the router ('mlp.gate') must stay in full precision, and the expert
        linears sharing a suffix are quantized as one group. Dense models return None, letting
        optimum use its own default grouping.
        """
        if not self.is_moe_model:
            return None
        from optimum.gptq.utils import get_layers

        # Do not quantize the gate part.
        block = _deep_getattr(model, block_name)[-1]
        prefix, _ = self._get_experts(block)
        layers = get_layers(block)
        res = []
        experts = defaultdict(list)
        experts_idx = None
        for name, layer in layers.items():
            if self.model_type == 'qwen3_next' and name.startswith('self_attn.'):
                # ignore attn
                continue
            if prefix and name.startswith(prefix):
                suffix = name.rsplit('.', 1)[-1]
                experts[suffix].append(name)
                experts_idx = len(res)
            elif 'mlp.gate' not in name:
                res.append([name])
        if experts_idx is not None:
            res[experts_idx:experts_idx] = experts.values()
        return res

    @contextmanager
    def _patch_calib_dataset(self):
        """Feed our own samples to optimum.

        optimum's ``get_dataset`` only knows a handful of built-in dataset names, and its
        ``prepare_dataset`` re-batches with its own collator. Both are replaced so the caller's
        already-encoded samples are used verbatim.
        """
        from optimum.gptq import quantizer

        _get_dataset_origin = quantizer.get_dataset
        _prepare_dataset_origin = quantizer.prepare_dataset
        calib_data = self.calib_data
        quantizer.get_dataset = lambda *args, **kwargs: calib_data
        quantizer.prepare_dataset = lambda examples, *args, **kwargs: examples
        try:
            yield
        finally:
            quantizer.get_dataset = _get_dataset_origin
            quantizer.prepare_dataset = _prepare_dataset_origin

    @staticmethod
    @contextmanager
    def _patch_block_output(model, block_name_to_quantize):
        """Force decoder blocks to return a tuple on transformers>=4.54.

        Those versions let a block return a bare Tensor, while optimum's GPTQ loop still indexes
        ``output[0]`` -- which would silently take the first ROW of the hidden states. The hook
        re-wraps any non-tuple output.
        """
        import transformers
        from packaging import version

        if version.parse(transformers.__version__) < version.parse('4.54'):
            yield
            return
        blocks = _deep_getattr(model, block_name_to_quantize)
        hooks = []

        def _to_tuple(module, input, output):
            if not isinstance(output, (list, tuple)):
                output = (output, )
            return output

        for block in blocks:
            hooks.append(block.register_forward_hook(_to_tuple))
        try:
            yield
        finally:
            for hook in hooks:
                hook.remove()

    def _build_optimum_quantizer(self, block_name_to_quantize, modules_in_block_to_quantize):
        """Construct optimum's GPTQQuantizer across its two incompatible signatures.

        optimum renamed ``checkpoint_format`` to ``format`` and dropped ``serialization_keys``.
        Both spellings are probed by inspection rather than by version number, because the rename
        is not tied to a clean release boundary. Passing the wrong one would be swallowed by
        ``**kwargs`` and silently produce a checkpoint in the default format.
        """
        import inspect

        from optimum.gptq import GPTQQuantizer

        kwargs = dict(
            bits=self.quant_bits,
            group_size=self.group_size,
            # optimum requires a non-empty dataset name even though _patch_calib_dataset
            # replaces the loader; the value itself is never read.
            dataset='twinkle',
            batch_size=self.batch_size,
            block_name_to_quantize=block_name_to_quantize,
            modules_in_block_to_quantize=modules_in_block_to_quantize)
        params = inspect.signature(GPTQQuantizer.__init__).parameters
        if 'checkpoint_format' in params:
            kwargs['checkpoint_format'] = self.checkpoint_format
        elif 'format' in params:
            kwargs['format'] = self.checkpoint_format
        else:
            raise RuntimeError('Cannot tell optimum which GPTQ checkpoint format to write: neither '
                               '`checkpoint_format` nor `format` is accepted by this optimum version.')
        gptq_quantizer = GPTQQuantizer(**kwargs)
        # block_name_to_quantize must land in the saved config: the loader needs it to find the
        # blocks again, and older optimum does not serialize it by default. Newer versions dropped
        # serialization_keys and persist it themselves.
        if hasattr(gptq_quantizer, 'serialization_keys'):
            gptq_quantizer.serialization_keys.append('block_name_to_quantize')
        return gptq_quantizer

    def quantize(self, model):
        import torch

        self._check_ready()
        block_name_to_quantize = self.block_name_to_quantize or self.get_block_name_to_quantize(model)
        modules_in_block_to_quantize = (
            self.modules_in_block_to_quantize or self.get_modules_in_block_to_quantize(model, block_name_to_quantize))
        logger.info(f'block_name_to_quantize: {block_name_to_quantize}')
        logger.info(f'modules_in_block_to_quantize: {modules_in_block_to_quantize}')
        with self._patch_calib_dataset():
            gptq_quantizer = self._build_optimum_quantizer(block_name_to_quantize, modules_in_block_to_quantize)
            logger.info('Start quantizing the model...')
            logger.warning('The process of packing the model takes a long time and there is no progress bar. '
                           'Please be patient and wait...')
            if not hasattr(model, 'hf_device_map'):
                model.hf_device_map = {'': torch.device('cuda:0')}
            with self._patch_block_output(model, block_name_to_quantize):
                gptq_quantizer.quantize_model(model, self.tokenizer)
            # The calibration data has no business in the checkpoint config.
            model.config.quantization_config.pop('dataset', None)
        if self.v2:
            self._register_v2_tied_weights(model)
        self.gptq_quantizer = gptq_quantizer
        return model

    @staticmethod
    def _register_v2_tied_weights(model) -> None:
        """gptq_v2 adds two buffers that must be declared as dynamically tied.

        Without this, saving reports them as unexpected shared tensors and the write fails.
        """
        if not getattr(model, '_dynamic_tied_weights_keys', None):
            model._dynamic_tied_weights_keys = []
        model._dynamic_tied_weights_keys += ['wf_unsqueeze_zero', 'wf_unsqueeze_neg_one']

    def save(self, model, output_dir: str, *, safe_serialization: bool = True, max_shard_size='5GB') -> None:
        if self.gptq_quantizer is None:
            raise RuntimeError('call quantize() before save(): the GPTQQuantizer holds the packing metadata.')
        self.gptq_quantizer.save(
            model, output_dir, safe_serialization=safe_serialization, max_shard_size=max_shard_size)


class GptqV2Quantizer(GptqQuantizer):
    """GPTQ writing the gptq_v2 checkpoint format."""

    quant_method = 'gptq_v2'

    def __init__(self, *args, **kwargs):
        kwargs['v2'] = True
        super().__init__(*args, **kwargs)
