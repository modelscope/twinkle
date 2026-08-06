# Copyright (c) ModelScope Contributors. All rights reserved.
from contextlib import contextmanager
from typing import List, Optional

from twinkle.quantizer.base import CalibrationQuantizer
from twinkle.utils import get_logger, requires

logger = get_logger()


class AwqQuantizer(CalibrationQuantizer):
    """AWQ (Activation-aware Weight Quantization) via autoawq.

    autoawq owns both the packing and the writer, so ``model`` here must be an
    ``AutoAWQForCausalLM``, not a plain transformers model -- it is the AWQ wrapper that exposes
    ``.quantize()`` / ``.save_quantized()``. Load it with
    ``AutoAWQForCausalLM.from_pretrained(...)`` before handing it over.
    """

    quant_method = 'awq'

    def __init__(self,
                 quant_bits: Optional[int] = 4,
                 group_size: int = 128,
                 batch_size: int = 1,
                 modules_to_not_convert: Optional[List[str]] = None,
                 zero_point: bool = True,
                 version: str = 'GEMM',
                 lm_head_key: str = 'lm_head',
                 **kwargs):
        requires('autoawq')
        super().__init__(
            quant_bits=quant_bits,
            group_size=group_size,
            batch_size=batch_size,
            modules_to_not_convert=modules_to_not_convert,
            **kwargs)
        self.zero_point = zero_point
        self.version = version
        self.lm_head_key = lm_head_key

    @staticmethod
    @contextmanager
    def _patch_move_embed(awq_model):
        """Keep autoawq from moving embeddings on an accelerate-offloaded model.

        autoawq shuttles the embedding between devices around each calibration block. When the model
        carries an accelerate hook (``_hf_hook``), that hook already owns placement and the extra
        move corrupts it, so the move is skipped for every target except an explicit 'cpu'.
        """
        _origin_move_embed = awq_model.move_embed

        def _move_embed(model, device: str):
            if hasattr(model, '_hf_hook') and device != 'cpu':
                return
            _origin_move_embed(model, device)

        awq_model.move_embed = _move_embed
        try:
            yield
        finally:
            awq_model.move_embed = _origin_move_embed

    @contextmanager
    def _patch_calib_dataset(self):
        """Feed our own samples to autoawq.

        autoawq's ``get_calib_dataset`` downloads and tokenizes a fixed HF dataset; replacing it is
        the only way to calibrate on caller-supplied data (autoawq takes no such argument).
        """
        from awq.quantize import quantizer

        _origin = quantizer.get_calib_dataset
        calib_data = self.calib_data
        quantizer.get_calib_dataset = lambda *args, **kwargs: calib_data
        try:
            yield
        finally:
            quantizer.get_calib_dataset = _origin

    def get_quant_config(self) -> dict:
        quant_config = {
            'zero_point': self.zero_point,
            'q_group_size': self.group_size,
            'w_bit': self.quant_bits,
            'version': self.version,
        }
        if self.modules_to_not_convert:
            quant_config['modules_to_not_convert'] = self.modules_to_not_convert
        return quant_config

    def quantize(self, model):
        self._check_ready()
        quant_config = self.get_quant_config()
        logger.info(f'quant_config: {quant_config}')
        logger.info('Start quantizing the model...')
        with self._patch_calib_dataset(), self._patch_move_embed(model):
            model.quantize(self.tokenizer, quant_config=quant_config, n_parallel_calib_samples=self.batch_size)
        self._ensure_lm_head_skipped(model)
        return model

    def _ensure_lm_head_skipped(self, model) -> None:
        """Once anything is excluded from conversion, lm_head must be excluded too.

        A partially converted model keeps an fp16 lm_head, but autoawq only records the modules it
        was told to skip. If lm_head is missing from that list the loader tries to read it as a
        quantized layer and fails, so it is appended here.
        """
        quant_config = getattr(model, 'quant_config', None)
        if quant_config is None or not getattr(quant_config, 'modules_to_not_convert', None):
            return
        if self.lm_head_key not in quant_config.modules_to_not_convert:
            quant_config.modules_to_not_convert.append(self.lm_head_key)

    def save(self, model, output_dir: str, *, safe_serialization: bool = True, max_shard_size='5GB') -> None:
        # autoawq ships its own writer; save_pretrained would drop the packed weights.
        model.save_quantized(output_dir, safetensors=safe_serialization, shard_size=max_shard_size)
