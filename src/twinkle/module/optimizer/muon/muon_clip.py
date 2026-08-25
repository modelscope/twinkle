# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import torch
import torch.distributed as dist
from torch.optim import Optimizer
from typing import Optional, Tuple

from twinkle.utils import get_logger
from .max_logits_tracker import MaxLogitsTracker

logger = get_logger()


class MuonClip(Optimizer):
    """Muon with QK-Clip: orthogonalised momentum updates, plus a brake on attention logit blow-up.

    Two mechanisms, controlled per param group:

    - **Muon** (``apply_muon=True``, 2-D weights): the momentum buffer is orthogonalised by a polynomial
      Newton-Schulz iteration before it is applied, so every direction of a weight matrix is updated at a
      comparable magnitude instead of the update being dominated by the leading singular direction.
    - **QK-Clip** (``is_qk=True``): when the peak attention logit of the step exceeds ``qk_clip_tau``,
      the Q/K weights and their pending update are both scaled by ``sqrt(tau / max_logits)``, which pulls
      the logits back under the threshold without discarding the step. See
      :class:`MaxLogitsTracker` for where the peak comes from and how exact it is.

    Groups with ``apply_muon=False`` -- the biases, the norms, the embedding and the LM head -- take a
    plain AdamW step. Muon has nothing to offer them: orthogonalising a vector is just normalising it,
    and the embedding is a lookup table whose rows are per-token, so the ``sqrt(max(rows, cols))`` scaling
    that matches Muon's step to AdamW's is calibrated on the wrong dimension once the vocabulary is one of
    them. Keeping them on AdamW is what the Muon recipes do, and it means every parameter handed to this
    optimizer is trained by something -- unlike ``torch.optim.Muon``, which rejects non-2-D parameters
    outright and leaves the caller to pair it with a second optimizer.

    Note:
        Verified on a single device. The Newton-Schulz iteration multiplies the momentum buffer by its own
        transpose, so under FSDP2 or tensor parallelism -- where a parameter is a shard of the real matrix
        -- it orthogonalises the shard rather than the matrix. Sharded training needs its own verification
        before this is trusted there.

    Args:
        params: Parameters or param groups. Use :func:`create_muon_param_groups` to get the
            ``apply_muon``/``is_qk`` keys onto the right groups.
        lr: Learning rate, shared by both update rules.
        momentum: Momentum coefficient of the buffer the Muon update is built from.
        weight_decay: Decoupled weight decay, applied to the weight before the update, for every group.
        nesterov: Blend the raw gradient into the buffer before orthogonalising it. Muon groups only.
        newton_schulz_steps: Iterations of the orthogonalisation. More is closer to orthogonal, and
            costs a pair of matmuls each.
        qk_clip_tau: The attention logit ceiling QK-Clip aims to keep the model under.
        qk_clip_enabled: Whether to run QK-Clip at all. Off means the attention wrappers are never
            installed, so the model runs at its normal speed.
        rms_scale_factor: Scales the orthogonalised update by ``sqrt(max(rows, cols)) * this``, which is
            what puts it on a comparable footing with an Adam-sized step.
        adamw_betas: The AdamW coefficients for the groups Muon does not apply to.
        adamw_eps: The AdamW denominator epsilon for those same groups.
    """

    def __init__(
            self,
            params,
            lr: float = 2e-4,
            momentum: float = 0.95,
            weight_decay: float = 0.1,
            nesterov: bool = False,
            newton_schulz_steps: int = 5,
            qk_clip_tau: float = 10000.0,
            qk_clip_enabled: bool = True,
            rms_scale_factor: float = 0.2,
            adamw_betas: Tuple[float, float] = (0.9, 0.999),
            adamw_eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            newton_schulz_steps=newton_schulz_steps,
            qk_clip_tau=qk_clip_tau,
            qk_clip_enabled=qk_clip_enabled,
            rms_scale_factor=rms_scale_factor,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        super().__init__(params, defaults)
        # Only pay for the attention wrappers if some group will actually clip. Reading the groups rather
        # than the argument, since `create_muon_param_groups` can turn it off per group.
        if any(
                group.get('qk_clip_enabled', qk_clip_enabled) and group.get('is_qk', False)
                for group in self.param_groups):
            MaxLogitsTracker.install()

    @staticmethod
    @torch.no_grad()
    def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
        """Approximately orthogonalise ``G`` by the Muon/Moonlight polynomial iteration.

        Runs in bfloat16 on the normalised matrix, which is what keeps the iteration stable: the
        polynomial only converges for inputs with singular values in range. A tall matrix is transposed
        first so the ``X @ X.T`` products stay the smaller of the two possible sizes.
        """
        a, b, c = (3.4445, -4.7750, 2.0315)

        X = G.bfloat16() / (G.norm() + eps)
        transposed = False
        if G.size(0) > G.size(1):
            X = X.T
            transposed = True

        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X

        if transposed:
            X = X.T

        return X.to(G.dtype)

    @torch.no_grad()
    def step(self, closure=None, max_logits: Optional[float] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        max_logits = self._resolve_max_logits(max_logits)

        for group in self.param_groups:
            lr = float(group['lr'])
            weight_decay = float(group['weight_decay'])
            qk_clip_tau = float(group.get('qk_clip_tau', 10000.0))
            qk_clip_enabled = bool(group.get('qk_clip_enabled', True))
            apply_muon = bool(group.get('apply_muon', True))
            is_qk_group = bool(group.get('is_qk', False))

            clip = None
            if qk_clip_enabled and is_qk_group and max_logits is not None and max_logits > qk_clip_tau:
                clip = math.sqrt(qk_clip_tau / max_logits)

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                state['step'] = state.get('step', 0) + 1

                if apply_muon and p.ndim >= 2:
                    update = self._muon_update(p, state, group)
                else:
                    update = self._adamw_update(p, state, group)

                # Decoupled weight decay: shrink the weight itself, not the gradient, so it is unaffected
                # by whichever update rule the group uses.
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                if clip is not None:
                    p.mul_(clip)
                    update = update * clip

                p.add_(update, alpha=-lr)

        return loss

    def _muon_update(self, p: torch.Tensor, state: dict, group: dict) -> torch.Tensor:
        """The Muon update: the momentum buffer, orthogonalised and rescaled to an Adam-sized step.

        Nesterov is applied to the buffer *before* orthogonalisation, per the algorithm as stated in
        `torch.optim.Muon` and the papers. Adding the raw gradient afterwards -- which the swift
        implementation this was ported from did -- puts a non-orthogonal term into an update whose whole
        purpose was to be orthogonal.
        """
        momentum = float(group['momentum'])
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros_like(p)
        buf = state['momentum_buffer']
        buf.mul_(momentum).add_(p.grad)

        direction = p.grad.add(buf, alpha=momentum) if group.get('nesterov', False) else buf
        orth = self.newton_schulz(direction, steps=int(group.get('newton_schulz_steps', 5)))
        rms_scale = math.sqrt(max(p.shape[0], p.shape[1])) * float(group.get('rms_scale_factor', 0.2))
        return orth * rms_scale

    @staticmethod
    def _adamw_update(p: torch.Tensor, state: dict, group: dict) -> torch.Tensor:
        """The AdamW update, for the parameters Muon does not apply to.

        AdamW rather than the momentum SGD the swift implementation used here, because this is where the
        embeddings, the LM head, the biases and the norms land -- the parameters that most depend on a
        per-element adaptive step, and the ones the Muon recipes keep on AdamW for exactly that reason.

        Weight decay is not applied here: `step` already does it, decoupled, for every group.
        """
        beta1, beta2 = group.get('adamw_betas', (0.9, 0.999))
        eps = float(group.get('adamw_eps', 1e-8))
        if 'exp_avg' not in state:
            state['exp_avg'] = torch.zeros_like(p)
            state['exp_avg_sq'] = torch.zeros_like(p)
        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

        exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

        step = state['step']
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        denom = (exp_avg_sq / bias_correction2).sqrt_().add_(eps)
        return (exp_avg / bias_correction1).div_(denom)

    def _resolve_max_logits(self, max_logits: Optional[float]) -> Optional[float]:
        """The peak attention logit every rank should clip against, as one number.

        Agreed across ranks by taking the maximum. QK-Clip scales the weight itself, not the gradient, so
        a rank-local peak would have each rank scale its own copy by a different factor -- data-parallel
        replicas only stay equal because they apply equal updates, and nothing would ever bring them back
        together. This is the one collective the optimizer performs, and it is also the point where the
        tracker's device tensor is read, once per step.
        """
        if max_logits is None:
            value = MaxLogitsTracker.consume()
        else:
            value = torch.as_tensor(float(max_logits))
        if not (dist.is_available() and dist.is_initialized()):
            return None if value is None else float(value)
        # The device has to be agreed on without communicating, so it comes from a parameter rather than
        # from the recorded value -- a rank that recorded nothing has no value to take a device from, and
        # a CPU tensor entering an NCCL reduction is an error rather than a slow path.
        device = self._param_device()
        if value is None:
            # Still join the reduction; -inf leaves the other ranks' peaks alone.
            reduced = torch.tensor(float('-inf'), device=device, dtype=torch.float32)
        else:
            reduced = value.detach().to(device=device, dtype=torch.float32).clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
        result = float(reduced)
        return None if result == float('-inf') else result

    def _param_device(self) -> torch.device:
        for group in self.param_groups:
            for p in group['params']:
                return p.device
        return torch.device('cpu')
