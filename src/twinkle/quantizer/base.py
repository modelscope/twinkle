# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, List, Optional


class Quantizer:
    """Post-training quantization of an already-loaded model.

    Subclasses split into two families, which differ in WHEN the weights change:

      - Calibration-based (AWQ / GPTQ): ``quantize`` runs the model over calibration samples to
        pick per-channel scales, so it needs ``calib_data`` and mutates the weights in place.
      - Config-based (BNB / FP8 / HQQ / Quanto / EETQ): there is nothing to calibrate. The
        quantization happens when transformers LOADS the model, so these subclasses only build a
        ``transformers`` quantization_config; ``quantize`` on an already-loaded model is a no-op and
        says so instead of pretending to have quantized anything.

    ``quant_method`` is the string transformers/vLLM use for the same scheme, so it doubles as the
    registry key in ``QUANTIZER_MAPPING``.
    """

    quant_method: str = ''
    # Whether quantize() needs calibration samples to compute scales.
    requires_calibration: bool = False

    def __init__(self, quant_bits: Optional[int] = None, **kwargs):
        self.quant_bits = quant_bits
        self.kwargs = kwargs

    def quantize(self, model):
        """Quantize ``model`` and return it.

        Calibration-based subclasses require ``calib_data`` to have been supplied (see
        ``set_calib_data``), and return a model whose weights are already packed.
        """
        raise NotImplementedError

    def get_quantization_config(self):
        """The transformers quantization_config for this scheme, or None if the scheme does not
        load through transformers' quantizer hook (AWQ/GPTQ pack the checkpoint themselves)."""
        return None

    def save(self, model, output_dir: str, *, safe_serialization: bool = True, max_shard_size='5GB') -> None:
        """Write the quantized model out. Overridden where the backend owns its own writer."""
        model.save_pretrained(output_dir, safe_serialization=safe_serialization, max_shard_size=max_shard_size)


class CalibrationQuantizer(Quantizer):
    """Base for schemes that need forward passes over calibration data (AWQ, GPTQ)."""

    requires_calibration = True

    def __init__(self,
                 quant_bits: Optional[int] = None,
                 group_size: int = 128,
                 batch_size: int = 1,
                 modules_to_not_convert: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(quant_bits=quant_bits, **kwargs)
        self.group_size = group_size
        self.batch_size = batch_size
        self.modules_to_not_convert = modules_to_not_convert
        self.calib_data: Optional[List[Any]] = None
        self.tokenizer = None

    def set_calib_data(self, calib_data: List[Any], tokenizer=None) -> None:
        """Supply calibration samples.

        Kept separate from __init__ so the caller owns dataset loading/encoding: the shape each
        backend wants differs (AWQ takes batched token tensors, GPTQ takes input_ids dicts), and
        twinkle has no opinion on where the data comes from.
        """
        self.calib_data = calib_data
        self.tokenizer = tokenizer

    def _check_ready(self) -> None:
        if self.quant_bits is None:
            raise ValueError(f'{type(self).__name__} requires quant_bits.')
        if not self.calib_data:
            raise ValueError(f'{type(self).__name__} is calibration-based; call set_calib_data() '
                             f'with calibration samples before quantize().')


class ConfigQuantizer(Quantizer):
    """Base for schemes applied by transformers at load time.

    These cannot quantize a model that is already in memory -- transformers swaps the linear layers
    while materialising the weights. So ``quantize`` deliberately does NOT touch the model; use
    ``get_quantization_config()`` and pass it to ``from_pretrained``.
    """

    def quantize(self, model):
        raise RuntimeError(f'{self.quant_method} quantization is applied when the model is LOADED, not '
                           f'afterwards. Pass {type(self).__name__}(...).get_quantization_config() as '
                           f'`quantization_config` to from_pretrained instead of calling quantize().')
