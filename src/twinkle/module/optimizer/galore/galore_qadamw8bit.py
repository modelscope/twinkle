# Copyright (c) ModelScope Contributors. All rights reserved.
# QGaLore (quantized GaLore) -- see https://arxiv.org/abs/2407.00088
"""QGaLore's quantized low-rank projection, provisioned from the external ``q_galore_torch`` package.

The in-repo GaLore optimizers (:mod:`.galore_adamw` / :mod:`.galore_adamw8bit`) project the
FULL-PRECISION gradient; the quantized subspace tracking QGaLore adds (int4/int8 projection, cosine-
thresholded subspace reuse, a gradient queue) is a separate algorithm that swift has never carried in
tree -- legacy ``swift sft`` likewise requires ``pip install q_galore_torch`` and uses its
``QGaLoreAdamW8bit`` directly. This module keeps that single source of truth: it subclasses the
external optimizer so ``set_optimizer`` can resolve it by the twinkle-owned name ``QGaLoreAdamW8bit``
through ``construct_class``, while the optional dependency stays optional.

The base is imported lazily and falls back to ``object`` when ``q_galore_torch`` is absent -- the same
pattern :mod:`.galore_adamw8bit` uses for ``bitsandbytes`` -- so importing twinkle never requires the
package; only constructing the optimizer does, and then it fails loudly with the install hint rather
than training an unquantized GaLore and silently ignoring ``galore_quantization``.
"""

try:
    from q_galore_torch import QGaLoreAdamW8bit as _QGaLoreAdamW8bit
except ImportError:
    _QGaLoreAdamW8bit = object


class QGaLoreAdamW8bit(_QGaLoreAdamW8bit):  # type: ignore[misc,valid-type]
    """8-bit AdamW with QGaLore's quantized low-rank gradient projection.

    A thin twinkle-named handle over ``q_galore_torch.QGaLoreAdamW8bit``: construction and ``step``
    are entirely the external optimizer's, which reads the ``rank``/``update_proj_gap``/``scale``/
    ``proj_type`` keys plus the quantization keys (``quant``/``quant_n_bit``/``quant_group_size``/
    ``cos_threshold``/``gamma_proj``/``queue_size``) off each param group. Those keys are installed by
    ``create_galore_param_groups`` from a ``GaLoreConfig`` with ``quantize=True``.
    """

    def __init__(self, *args, **kwargs):
        if _QGaLoreAdamW8bit is object:
            raise ImportError(
                'QGaLoreAdamW8bit (galore_quantization) requires the q_galore_torch package, please '
                'install it by `pip install q_galore_torch`. If you hit an `absmax2` error, downgrade '
                'bitsandbytes to 0.40.0. Alternatively drop galore_quantization to use plain GaLore.')
        super().__init__(*args, **kwargs)
