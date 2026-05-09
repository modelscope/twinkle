from typing import Any, Dict, List, Optional

import json
import time

import numpy as np

from twinkle.data_format import Trajectory
from twinkle.data_format.sampling import SampleResponse, SamplingParams
from twinkle.template.base import Template

from twinkle_agentic.tools.tool_manager import ToolManager
from .base import Rollout


def _to_plain(obj: Any) -> Any:
    """Recursively convert numpy arrays/scalars to plain Python lists/numbers.

    Mirrors ``vllm_sampler._convert_ndarray_to_list`` but lives locally so we
    do not depend on a private symbol.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        conv = [_to_plain(x) for x in obj]
        return type(obj)(conv) if isinstance(obj, tuple) else conv
    return obj


class MultiTurnRollout(Rollout):
    """Agentic multi-turn rollout with tool use (batched).

    Contract (matches :class:`Rollout`): accepts a ``List[Trajectory]`` and
    returns a ``List[Trajectory]`` of the same length, in the same order.
    Every turn issues a SINGLE batched ``sampler.sample(active_pifs)`` call
    so vLLM can run all live trajectories in parallel; finished trajectories
    are parked and excluded from subsequent batches.

    Per-trajectory loop:
        1. Encode the initial trajectory into an ``InputFeature`` with a
           generation prompt at the tail.
        2. Call ``sampler.sample(pifs)`` (batched). The sampler internally
           invokes ``template.concat_input_feature`` to append the freshly
           sampled assistant tokens; we pick up ``seq.new_input_feature`` as
           the new running ``pif``.
        3. If ``stop_reason == 'length'`` or the decoded assistant output has
           no tool calls, mark the trajectory as done.
        4. Otherwise, invoke the tools via ``ToolManager`` and append each
           tool response as a ``{'role':'tool', 'content': ...}`` message.
           Compute "bridge" tokens (tool turns + next ``<|im_start|>assistant``
           header) with ``labels = -100`` and extend the pif.
        5. Repeat until all trajectories are done or ``max_turns`` is hit.

    Per-call overrides via ``**kwargs``:
        * ``sampling_params``: shared :class:`SamplingParams` for the batch.
        * ``tool_manager``: either a single :class:`ToolManager` (applied to
          every trajectory) or a list of ``ToolManager`` aligned 1:1 with
          ``trajectories`` (used by :class:`MultiTurnCondenseRollout` to
          attach a trajectory-bound ``ExtractCondensed``).

    The class intentionally has no knowledge of condensers/chunkers; they are
    applied upstream (on the trajectory before rollout) or downstream
    (on the returned messages).
    """

    def __init__(
        self,
        sampler,
        template: Template,
        tool_manager: ToolManager,
        sampling_params: Optional[SamplingParams] = None,
        max_turns: int = 6,
        trace_path: Optional[str] = None,
    ):
        if template is None:
            raise ValueError('MultiTurnRollout requires a local Template instance')
        if tool_manager is None:
            raise ValueError('MultiTurnRollout requires a ToolManager')
        if max_turns < 1:
            raise ValueError(f'max_turns must be >= 1, got {max_turns}')
        self.sampler = sampler
        self.template = template
        self.tool_manager = tool_manager
        self.sampling_params = sampling_params or SamplingParams()
        self.max_turns = max_turns
        # When set, every turn writes one JSONL record per active
        # trajectory to ``trace_path``. The file is truncated at
        # construction time (matching the behaviour of the legacy
        # ``_make_dump_rollout_trace`` hook); subsequent writes append.
        # Errors during trace writing are swallowed on purpose so
        # observability can never break a training step.
        self.trace_path = trace_path
        if self.trace_path:
            try:
                # Truncate up front so repeated rollouts start from an
                # empty file. Using a context manager here would be
                # equivalent; explicit ``close()`` is clearer.
                f = open(self.trace_path, 'w', encoding='utf-8')
                f.close()
            except OSError:
                # If we can't even create the file, disable tracing
                # silently rather than crashing the training job.
                self.trace_path = None

        if self.sampling_params.num_samples != 1:
            raise ValueError(
                f'MultiTurnRollout currently supports num_samples=1 only, '
                f'got {self.sampling_params.num_samples}')
        assert self.template.truncation_strategy != 'split', (
            "MultiTurnRollout does not support truncation_strategy='split'; "
            'use left/right/raise on the template.')

    def __call__(self, trajectories: List[Trajectory], **kwargs) -> List[Trajectory]:
        if isinstance(trajectories, dict):
            raise TypeError(
                'MultiTurnRollout.__call__ expects a List[Trajectory]; '
                'wrap a single trajectory as [trajectory].')
        trajectories = list(trajectories)
        n = len(trajectories)
        if n == 0:
            return []

        sampling_params = kwargs.get('sampling_params', self.sampling_params)
        tool_managers = self._resolve_tool_managers(
            kwargs.get('tool_manager', self.tool_manager), n)

        # 1. Encode each trajectory once; ``pifs[i]`` is the live per-turn
        #    state for trajectory ``i``.
        pifs: List[Dict[str, Any]] = []
        for traj in trajectories:
            pif = self.template.encode(traj, add_generation_prompt=True)
            pif = _to_plain(pif)
            pif.setdefault('messages', list(traj.get('messages', [])))
            pifs.append(pif)

        all_logprobs: List[List[Any]] = [[] for _ in range(n)]
        stop_reasons: List[Optional[str]] = [None] * n
        turns: List[int] = [0] * n
        truncated: List[bool] = [False] * n
        done: List[bool] = [False] * n

        for _ in range(self.max_turns):
            active = [i for i in range(n) if not done[i]]
            if not active:
                break

            # 2. One batched sample call for all currently-live trajectories.
            batch_pifs = [pifs[i] for i in active]
            resps = self.sampler.sample(batch_pifs, sampling_params=sampling_params)
            resps = self._unwrap_response_list(resps, len(active))

            pending_bridges: List[tuple] = []  # (global_idx, tool_messages)
            trace_rows: List[Dict[str, Any]] = []  # buffered per-turn records
            for local_idx, global_idx in enumerate(active):
                turns[global_idx] += 1
                seq = resps[local_idx].sequences[0]

                if seq.new_input_feature is None or 'input_ids' not in seq.new_input_feature:
                    raise RuntimeError(
                        f'Sampler returned a SampledSequence without '
                        f'new_input_feature.input_ids at batch index '
                        f'{local_idx} (trajectory {global_idx}); '
                        f'cannot continue multi-turn.')

                pifs[global_idx] = _to_plain(dict(seq.new_input_feature))
                if seq.logprobs is not None:
                    if len(seq.logprobs) != len(seq.tokens):
                        raise RuntimeError(
                            f'logprobs length ({len(seq.logprobs)}) does not '
                            f'match sampled token count ({len(seq.tokens)}) '
                            f'at turn {turns[global_idx]} '
                            f'(trajectory {global_idx})')
                    all_logprobs[global_idx].extend(seq.logprobs)
                stop_reasons[global_idx] = seq.stop_reason

                # 3. Termination conditions
                if seq.stop_reason == 'length':
                    done[global_idx] = True
                    trace_rows.append(self._trace_row(
                        turn=turns[global_idx],
                        global_idx=global_idx,
                        n=n,
                        seq=seq,
                        tool_calls=None,
                        done=True,
                        truncated=False,
                        pif=pifs[global_idx]))
                    continue

                tool_calls = self.template.parse_tool_call(seq.decoded or '')
                if not tool_calls:
                    done[global_idx] = True
                    trace_rows.append(self._trace_row(
                        turn=turns[global_idx],
                        global_idx=global_idx,
                        n=n,
                        seq=seq,
                        tool_calls=tool_calls,
                        done=True,
                        truncated=False,
                        pif=pifs[global_idx]))
                    continue

                if turns[global_idx] >= self.max_turns:
                    truncated[global_idx] = True
                    done[global_idx] = True
                    trace_rows.append(self._trace_row(
                        turn=turns[global_idx],
                        global_idx=global_idx,
                        n=n,
                        seq=seq,
                        tool_calls=tool_calls,
                        done=True,
                        truncated=True,
                        pif=pifs[global_idx]))
                    continue

                # 4. Dispatch tools per trajectory (uses this trajectory's
                #    tool_manager, which may be a trajectory-bound clone).
                tool_messages = [{
                    'role': 'tool',
                    'content': tool_managers[global_idx](tc),
                } for tc in tool_calls]
                pending_bridges.append((global_idx, tool_messages))
                trace_rows.append(self._trace_row(
                    turn=turns[global_idx],
                    global_idx=global_idx,
                    n=n,
                    seq=seq,
                    tool_calls=tool_calls,
                    done=False,
                    truncated=False,
                    pif=pifs[global_idx]))

            # Extend pif with bridge tokens for every trajectory that has
            # outstanding tool turns. Done serially: bridge computation is
            # a cheap decode-diff-encode on python strings / token lists.
            for global_idx, tool_messages in pending_bridges:
                pifs[global_idx] = self._extend_with_bridge(
                    pifs[global_idx], tool_messages)

            # Flush this turn's trace records (one JSONL line each). This
            # happens AFTER bridge extension so a post-turn consumer sees
            # the final pif length for the turn.
            if self.trace_path and trace_rows:
                self._write_trace(trace_rows)

        # 5. Merge pif fields into each trajectory dict at TOP LEVEL so
        #    downstream consumers (VLLMSampler with ``'input_ids' in inputs``)
        #    see an encoded InputFeature and skip re-encoding.
        outs: List[Trajectory] = []
        for i, traj in enumerate(trajectories):
            out = dict(traj)
            out.update(pifs[i])
            out['messages'] = list(pifs[i].get('messages') or out.get('messages', []))
            out['logprobs'] = all_logprobs[i] if all_logprobs[i] else None
            out['turns'] = turns[i]
            out['stop_reason'] = stop_reasons[i]
            out['truncated'] = truncated[i]
            outs.append(out)
        return outs

    # ------------------------------------------------------------------ private

    @staticmethod
    def _resolve_tool_managers(arg, n: int) -> List[ToolManager]:
        """Broadcast a single ``ToolManager`` or validate a per-trajectory list."""
        if isinstance(arg, list):
            if len(arg) != n:
                raise ValueError(
                    f'per-call tool_manager list length ({len(arg)}) does '
                    f'not match number of trajectories ({n})')
            return list(arg)
        return [arg] * n

    @staticmethod
    def _trace_row(
        *,
        turn: int,
        global_idx: int,
        n: int,
        seq,
        tool_calls,
        done: bool,
        truncated: bool,
        pif: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build one per-trajectory trace record for the current turn.

        Deliberately flat + JSON-friendly. ``decoded`` is truncated-safe
        (it's just a string). ``trainable_tokens`` is the count of labels
        not equal to -100 so far, i.e. GRPO-loss-eligible positions.
        """
        labels = pif.get('labels') or []
        trainable = sum(1 for l in labels if l != -100)
        return {
            'ts': time.time(),
            'turn': int(turn),
            'batch_size': int(n),
            'trajectory_idx': int(global_idx),
            'stop_reason': getattr(seq, 'stop_reason', None),
            'decoded': getattr(seq, 'decoded', '') or '',
            'tool_call_count': 0 if not tool_calls else len(tool_calls),
            'done': bool(done),
            'truncated': bool(truncated),
            'input_ids_len': len(pif.get('input_ids') or []),
            'trainable_tokens': trainable,
        }

    def _write_trace(self, rows: List[Dict[str, Any]]) -> None:
        """Append trace rows as JSONL. Errors are swallowed by design.

        Observability must never break training -- any I/O or encoding
        problem is silently ignored so a disk-full / permission issue
        doesn't take down the optimisation loop.
        """
        if not self.trace_path or not rows:
            return
        try:
            lines = [
                json.dumps(r, ensure_ascii=False, default=str)
                for r in rows]
            with open(self.trace_path, 'a', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
        except Exception:
            pass

    @staticmethod
    def _unwrap_response_list(resps, expected: int) -> List[SampleResponse]:
        """Validate that the sampler returned ``expected`` ``SampleResponse``s,
        one per input in the batch.
        """
        if not isinstance(resps, list):
            raise TypeError(
                f'expected List[SampleResponse] from sampler.sample (batched '
                f'call), got {type(resps).__name__}')
        if len(resps) != expected:
            raise RuntimeError(
                f'sampler returned {len(resps)} responses for a batch of '
                f'{expected} trajectories; expected one per input.')
        for i, r in enumerate(resps):
            if not isinstance(r, SampleResponse):
                raise TypeError(
                    f'expected SampleResponse at batch index {i}, got '
                    f'{type(r).__name__}')
            if not r.sequences:
                raise RuntimeError(
                    f'SampleResponse at batch index {i} has no sequences')
        return resps

    def _extend_with_bridge(
        self,
        pif: Dict[str, Any],
        tool_messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Append tool messages and the next generation prompt as -100 bridge.

        Strategy: decode the CURRENT pif input_ids back to a string, render
        the canonical chat-template string for ``messages + tool_messages``
        with ``add_generation_prompt=True``, diff at the STRING level, and
        tokenize ONLY the delta. This avoids retokenising history (which would
        drift through the ``decode(tokens, skip_special_tokens=True)`` round
        trip that ``concat_input_feature`` does).
        """
        tokenizer = self.template.tokenizer

        messages_before = list(pif.get('messages') or [])
        messages_after = messages_before + list(tool_messages)

        current_text = tokenizer.decode(pif['input_ids'], skip_special_tokens=False)
        s_after = tokenizer.apply_chat_template(
            messages_after, tokenize=False, add_generation_prompt=True)

        bridge_text = self._compute_bridge_text(current_text, s_after)
        if not bridge_text:
            raise RuntimeError(
                'Bridge text computation returned empty string; '
                'tool turn would add no tokens (template misconfiguration?).')

        bridge_ids = tokenizer.encode(bridge_text, add_special_tokens=False)
        if not bridge_ids:
            raise RuntimeError(
                f'Bridge text tokenised to empty id list: {bridge_text!r}')

        new_pif = self._append_bridge_tokens(pif, bridge_ids)
        new_pif['messages'] = messages_after
        return new_pif

    @staticmethod
    def _compute_bridge_text(current_text: str, s_after: str) -> str:
        """Return the suffix of ``s_after`` beyond ``current_text``.

        Handles the case where ``current_text`` has trailing whitespace that
        the canonical chat_template rendering already consumed (e.g. the
        assistant ``<|im_end|>`` is emitted by vLLM without a trailing ``\\n``
        while the chat template always appends one between messages).
        """
        if s_after.startswith(current_text):
            return s_after[len(current_text):]
        # Tolerate trailing whitespace mismatch at the boundary.
        ct_stripped = current_text.rstrip()
        if s_after.startswith(ct_stripped):
            return s_after[len(ct_stripped):]
        raise RuntimeError(
            'Cannot align decoded pif text with canonical chat_template output. '
            f'current_text tail: {current_text[-80:]!r}; '
            f's_after at same offset: '
            f'{s_after[max(0, len(current_text) - 80):len(current_text) + 80]!r}')

    def _append_bridge_tokens(
        self,
        pif: Dict[str, Any],
        bridge_ids: List[int],
    ) -> Dict[str, Any]:
        """Append bridge tokens with labels = -100.

        Mirrors the unroll-append-reroll pattern of
        :meth:`Template.concat_input_feature` so that ``labels`` semantics
        stay consistent with the sampler-produced pif.

        Shallow copy is deliberately used: every mutation below is a
        top-level key reassignment, never an in-place change to nested
        tensors. Multimodal payloads (``images``, ``pixel_values``,
        ``image_grid_thw`` ...) are shared by reference so we avoid
        re-copying image buffers every turn.
        """
        result = dict(pif)

        input_ids = list(result['input_ids'])
        labels = list(result.get('labels') or [])
        # labels arrive in output/shifted order (post _roll_labels). Unroll by
        # one position (shift right by 1) to get back to input order.
        if labels:
            if len(labels) != len(input_ids):
                raise RuntimeError(
                    f'labels length ({len(labels)}) != input_ids length '
                    f'({len(input_ids)}); cannot safely append bridge tokens.')
            labels = labels[-1:] + labels[:-1]
        else:
            labels = [-100] * len(input_ids)

        input_ids = input_ids + list(bridge_ids)
        labels = labels + [-100] * len(bridge_ids)

        result['input_ids'] = input_ids
        result['labels'] = labels

        if 'mm_token_type_ids' in result:
            import torch
            mm = result['mm_token_type_ids']
            if not isinstance(mm, torch.Tensor):
                mm = torch.as_tensor(mm)
            pad = torch.zeros((mm.shape[0], len(bridge_ids)),
                              dtype=mm.dtype, device=mm.device)
            result['mm_token_type_ids'] = torch.cat([mm, pad], dim=1)

        # Replay the post pipeline: refresh attention_mask / position_ids /
        # length and re-roll labels back into output/shifted order.
        refreshed = self.template._invoke_post_pipeline([result])[0]
        result.update(refreshed)
        return _to_plain(result)
