# Copyright (c) ModelScope Contributors. All rights reserved.
"""Back-compat re-export of the non-blocking generation submission API.

``GenerationSubmissionMixin`` is a generic sampler capability, so it lives in the core
:mod:`twinkle.sampler.generation_submission` (mixed into the plain ``vLLMSampler`` / ``SGLangSampler`` for
driver-overlapped RL). The TransferQueue samplers in this package keep importing it from here; this shim
re-exports the SAME class object, so ``VLLMSamplerTQ(GenerationSubmissionMixin, vLLMSampler)`` resolves the
mixin to one identity in its MRO rather than a duplicate base.
"""
from __future__ import annotations

from twinkle.sampler.generation_submission import GenerationSubmissionMixin, _dispatch_generation

__all__ = ['GenerationSubmissionMixin', '_dispatch_generation']
