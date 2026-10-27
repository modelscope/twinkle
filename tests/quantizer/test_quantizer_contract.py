# quantizer decision-surface tests (plan T5-01..T5-07, T5-11).
import pytest
import torch

from twinkle.quantizer import QUANTIZER_MAPPING, AwqQuantizer, Fp8Quantizer, get_quantizer
from twinkle.quantizer.base import Quantizer


class _Dummy(Quantizer):
    quant_method = 'dummy'


class TestQuantizerBase:
    """T5-01/T5-02: unknown-kwarg warning and the abstract quantize guard."""

    def test_unknown_kwargs_warn_with_class_name(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            _Dummy(typoed_option=3)
        assert 'typoed_option' in caplog.text
        assert '_Dummy' in caplog.text

    def test_quantize_not_implemented(self):
        with pytest.raises(NotImplementedError):
            _Dummy().quantize(object())


class TestQuantizerRegistry:
    """T5-11: every registered class carries a distinct quant_method key."""

    def test_mapping_covers_all_families(self):
        expected = {'awq', 'gptq', 'gptq_v2', 'bnb', 'fp8', 'hqq', 'quanto', 'eetq'}
        assert expected <= set(QUANTIZER_MAPPING)

    def test_unknown_method_rejected_with_alternatives(self):
        with pytest.raises(ValueError, match='Supported'):
            get_quantizer('bogus')

    def test_factory_builds_registered_class(self):
        pytest.importorskip('autoawq', reason='AwqQuantizer constructor requires the autoawq package')
        q = get_quantizer('awq', quant_bits=4)
        assert isinstance(q, AwqQuantizer)
        assert q.quant_method == 'awq'


class TestAwqQuantizer:
    """T5-05 (awq half): quant_bits value-domain guard (constructor-time)."""

    @pytest.fixture(autouse=True)
    def _autoawq(self):
        pytest.importorskip('autoawq', reason='AwqQuantizer constructor requires the autoawq package')

    @pytest.mark.parametrize('bits', [3, 5, 16, 0])
    def test_invalid_bits_rejected(self, bits):
        with pytest.raises(ValueError, match='only 4 or 8'):
            AwqQuantizer(quant_bits=bits)

    def test_valid_bits_accepted(self):
        assert AwqQuantizer(quant_bits=4).quant_bits == 4


class TestFp8Quantizer:
    """T5-07/T5-08: quant_bits is not a knob; modules_to_not_convert passes through."""

    def test_quant_bits_ignored(self):
        pytest.importorskip('transformers')
        q = Fp8Quantizer(modules_to_not_convert=['lm_head'], quant_bits=8)
        assert q.quant_bits is None

    def test_modules_to_not_convert_forwarded(self):
        pytest.importorskip('transformers')
        q = Fp8Quantizer(modules_to_not_convert=['lm_head'])
        cfg = q.get_quantization_config()
        assert cfg.modules_to_not_convert == ['lm_head']


class TestBnbQuantizer:
    """T5-04/T5-05/T5-06: 4/8bit branches, the value guard, and dtype string resolution."""

    @pytest.fixture()
    def _bnb(self):
        pytest.importorskip('bitsandbytes')
        pytest.importorskip('transformers')

    def test_4bit_branch(self, _bnb):
        cfg = get_quantizer('bnb', quant_bits=4).get_quantization_config()
        assert cfg.load_in_4bit is True and cfg.load_in_8bit is False

    def test_8bit_branch(self, _bnb):
        cfg = get_quantizer('bnb', quant_bits=8).get_quantization_config()
        assert cfg.load_in_4bit is False and cfg.load_in_8bit is True

    def test_invalid_bits_rejected_at_config_time(self, _bnb):
        # The guard fires when the config is BUILT (load-time decision), not at construction.
        q = get_quantizer('bnb', quant_bits=3)
        with pytest.raises(ValueError, match='only 4 or 8'):
            q.get_quantization_config()

    def test_compute_dtype_string_resolved(self, _bnb):
        cfg = get_quantizer('bnb', quant_bits=4, bnb_4bit_compute_dtype='bfloat16').get_quantization_config()
        assert cfg.bnb_4bit_compute_dtype == torch.bfloat16
