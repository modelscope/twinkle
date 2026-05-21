# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import numpy as np
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

from twinkle.data_format import Trajectory
from twinkle.data_format.sampling import SampleResponse, SamplingParams
from twinkle.infra import remote_class, remote_function
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


@remote_class()
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
        max_trajectory_tokens: Optional[int] = None,
        trace_dir: Optional[str] = None,
        trace_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
        success_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ):
        super().__init__()
        if template is None:
            raise ValueError('MultiTurnRollout requires a local Template instance')
        if tool_manager is None:
            raise ValueError('MultiTurnRollout requires a ToolManager')
        if max_turns < 1:
            raise ValueError(f'max_turns must be >= 1, got {max_turns}')
        if max_trajectory_tokens is not None and max_trajectory_tokens < 1:
            raise ValueError(f'max_trajectory_tokens must be >= 1 or None, got '
                             f'{max_trajectory_tokens}')
        self.sampler = sampler
        self.template = template
        self.tool_manager = tool_manager
        self.sampling_params = sampling_params or SamplingParams()
        self.max_turns = max_turns
        self.max_trajectory_tokens = max_trajectory_tokens
        self.trace_dir = trace_dir
        self.trace_callback = trace_callback
        self.success_callback = success_callback
        if self.trace_dir:
            os.makedirs(self.trace_dir, exist_ok=True)

        if self.sampling_params.num_samples != 1:
            raise ValueError(f'MultiTurnRollout currently supports num_samples=1 only, '
                             f'got {self.sampling_params.num_samples}')
        assert self.template.truncation_strategy != 'split', (
            "MultiTurnRollout does not support truncation_strategy='split'; "
            'use left/right/raise on the template.')

    @remote_function()
    def __call__(self, trajectories: List[Trajectory], **kwargs) -> List[Trajectory]:
        if isinstance(trajectories, dict):
            raise TypeError('MultiTurnRollout.__call__ expects a List[Trajectory]; '
                            'wrap a single trajectory as [trajectory].')
        trajectories = list(trajectories)
        n = len(trajectories)
        if n == 0:
            return []

        sampling_params = kwargs.get('sampling_params', self.sampling_params)
        tool_managers = self._resolve_tool_managers(kwargs.get('tool_manager', self.tool_manager), n)

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
            actual = len(batch_pifs)
            device_mesh = getattr(self.sampler, 'device_mesh', None)
            min_batch_size = (device_mesh.data_world_size if device_mesh is not None else 1)
            if actual < min_batch_size:
                batch_pifs = batch_pifs + ([batch_pifs[-1]] * (min_batch_size - actual))
            resps = self.sampler.sample(batch_pifs, sampling_params=sampling_params)
            resps = self._unwrap_response_list(resps, len(batch_pifs))[:actual]

            pending_bridges: List[tuple] = []  # (global_idx, tool_messages)
            for local_idx, global_idx in enumerate(active):
                turns[global_idx] += 1
                seq = resps[local_idx].sequences[0]

                if seq.new_input_feature is None or 'input_ids' not in seq.new_input_feature:
                    raise RuntimeError(f'Sampler returned a SampledSequence without '
                                       f'new_input_feature.input_ids at batch index '
                                       f'{local_idx} (trajectory {global_idx}); '
                                       f'cannot continue multi-turn.')

                pifs[global_idx] = _to_plain(dict(seq.new_input_feature))
                if seq.logprobs is not None:
                    if len(seq.logprobs) != len(seq.tokens):
                        raise RuntimeError(f'logprobs length ({len(seq.logprobs)}) does not '
                                           f'match sampled token count ({len(seq.tokens)}) '
                                           f'at turn {turns[global_idx]} '
                                           f'(trajectory {global_idx})')
                    all_logprobs[global_idx].extend(seq.logprobs)
                stop_reasons[global_idx] = seq.stop_reason

                # 3. Termination conditions
                if seq.stop_reason == 'length':
                    done[global_idx] = True
                    continue

                # 3a. Sequence-length cap.
                if (self.max_trajectory_tokens is not None
                        and len(pifs[global_idx].get('input_ids') or []) >= self.max_trajectory_tokens):
                    truncated[global_idx] = True
                    done[global_idx] = True
                    continue

                _msgs = pifs[global_idx].get('messages') or []
                _last_msg = _msgs[-1] if _msgs else None
                tool_calls = (_last_msg.get('tool_calls') if isinstance(_last_msg, dict) else None)
                if not tool_calls:
                    tool_calls = self.template.parse_tool_call(seq.decoded or '')
                if not tool_calls:
                    done[global_idx] = True
                    continue

                if turns[global_idx] >= self.max_turns:
                    truncated[global_idx] = True
                    done[global_idx] = True
                    continue

                # 4. Dispatch tools per trajectory (uses this trajectory's
                #    tool_manager, which may be a trajectory-bound clone).
                tool_messages = [{
                    'role': 'tool',
                    'content': tool_managers[global_idx](tc),
                } for tc in tool_calls]
                pending_bridges.append((global_idx, tool_messages))

            # Extend pif with bridge tokens for every trajectory that has
            # outstanding tool turns. Done serially: bridge computation is
            # a cheap decode-diff-encode on python strings / token lists.
            for global_idx, tool_messages in pending_bridges:
                pifs[global_idx] = self._extend_with_bridge(pifs[global_idx], tool_messages)

        for i in range(n):
            if not all_logprobs[i]:
                continue
            labels_i = pifs[i].get('labels') or []
            trainable_i = sum(1 for label in labels_i if label != -100)
            if len(all_logprobs[i]) != trainable_i:
                raise RuntimeError(f'logprobs/labels misaligned for trajectory {i}: '
                                   f'{len(all_logprobs[i])} logprobs vs {trainable_i} '
                                   f'trainable labels (labels != -100). This invariant is '
                                   f'required by grpo._pad_and_align_to_batch; a mismatch '
                                   f'would silently corrupt GRPO old_logps alignment.')

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

        # Per-rollout trace dump: one JSON file per selected trajectory.
        # ``trace_callback`` decides whether to store; ``success_callback``
        # decides the filename prefix. Observability only -- any failure
        # is swallowed inside ``_write_rollout_traces``.
        if self.trace_dir:
            self._write_rollout_traces(outs, global_step=kwargs.get('global_step'))
        return outs

    # ------------------------------------------------------------------ private

    @staticmethod
    def _resolve_tool_managers(arg, n: int) -> List[ToolManager]:
        """Broadcast a single ``ToolManager`` or validate a per-trajectory list."""
        if isinstance(arg, list):
            if len(arg) != n:
                raise ValueError(f'per-call tool_manager list length ({len(arg)}) does '
                                 f'not match number of trajectories ({n})')
            return list(arg)
        return [arg] * n

    _TRACE_SKIP_KEYS = (
        'input_ids',
        'labels',
        'attention_mask',
        'position_ids',
        'logprobs',
        'pixel_values',
        'image_grid_thw',
        'mm_token_type_ids',
    )

    @classmethod
    def _serialize_for_trace(cls, traj: Dict[str, Any]) -> Dict[str, Any]:
        """Drop tensor-like / oversized fields; keep messages + metadata.

        Trace files are for human forensics; raw token ids, labels and
        image buffers would bloat the file by orders of magnitude without
        adding diagnostic value (the chat-template rendering of
        ``messages`` already captures the textual content).
        """
        slim = {k: v for k, v in traj.items() if k not in cls._TRACE_SKIP_KEYS}
        return _to_plain(slim)

    @staticmethod
    def _extract_ground_truth(traj: Dict[str, Any]) -> str:
        """Pull ``ground_truth`` out of ``user_data`` (list of kv pairs)."""
        for kv in (traj.get('user_data') or []):
            if (isinstance(kv, (list, tuple)) and len(kv) >= 2 and kv[0] == 'ground_truth'):
                return kv[1] or ''
        return ''

    @staticmethod
    def _resolve_traj_id(traj: Dict[str, Any], fallback_idx: int) -> str:
        """Stable-ish trajectory id for filenames.

        Prefers an explicit ``id`` / ``prompt_id`` key in ``user_data``
        (sanitised for filesystem safety); else falls back to
        ``{timestamp_ms}-{fallback_idx}`` so concurrent rollouts do not
        overwrite each other's files.
        """
        for kv in (traj.get('user_data') or []):
            if (isinstance(kv, (list, tuple)) and len(kv) >= 2 and kv[0] in ('id', 'prompt_id')):
                val = kv[1]
                if val not in (None, ''):
                    safe = re.sub(r'[^A-Za-z0-9_\-.]+', '_', str(val))[:64]
                    if safe:
                        return safe
        return f'{int(time.time() * 1000)}-{fallback_idx}'

    def _build_trace_record(
        self,
        traj: Dict[str, Any],
        *,
        idx: int,
        success: bool,
    ) -> Dict[str, Any]:
        """Assemble one trace record. Subclasses override to add fields.

        ``idx`` is the trajectory's position in the rollout output list,
        so subclasses can correlate the record with any per-call state
        they stashed on ``self`` during ``__call__``.
        """
        return {
            'trajectory': self._serialize_for_trace(traj),
            'ground_truth': self._extract_ground_truth(traj),
            'stop_reason': traj.get('stop_reason'),
            'truncated': bool(traj.get('truncated')),
            'success': success,
        }

    def _write_rollout_traces(
        self,
        outs: List[Dict[str, Any]],
        *,
        global_step: Optional[int] = None,
    ) -> None:
        """Dump one pretty-printed JSON file per selected trajectory.

        ``trace_callback`` (if set) decides WHETHER to store;
        ``success_callback`` (if set) decides the filename prefix
        (``ok-`` vs ``fail-``). Defaults: store-all / mark-fail.

        Observability must never break training -- any I/O or encoding
        problem on a single trajectory is swallowed so the remaining
        dumps and the optimisation loop continue unaffected.
        """
        if not self.trace_dir:
            return
        for idx, traj in enumerate(outs):
            try:
                should_store = True
                if self.trace_callback is not None:
                    try:
                        should_store = bool(self.trace_callback(traj))
                    except Exception:
                        should_store = False
                if not should_store:
                    continue

                success = False
                if self.success_callback is not None:
                    try:
                        success = bool(self.success_callback(traj))
                    except Exception:
                        success = False

                record = self._build_trace_record(traj, idx=idx, success=success)
                prefix = 'ok' if success else 'fail'
                # global_step prefix lets file listings sort by training step.
                step_tag = f'step{int(global_step):06d}-' if global_step is not None else ''
                fname = f'{step_tag}{prefix}-{self._resolve_traj_id(traj, idx)}.json'
                path = os.path.join(self.trace_dir, fname)
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(record, f, ensure_ascii=False, indent=2, default=str)
            except Exception:
                # Per-trajectory failure never aborts the loop.
                pass

    @staticmethod
    def _unwrap_response_list(resps, expected: int) -> List[SampleResponse]:
        """Validate that the sampler returned ``expected`` ``SampleResponse``s,
        one per input in the batch.
        """
        if not isinstance(resps, list):
            raise TypeError(f'expected List[SampleResponse] from sampler.sample (batched '
                            f'call), got {type(resps).__name__}')
        if len(resps) != expected:
            raise RuntimeError(f'sampler returned {len(resps)} responses for a batch of '
                               f'{expected} trajectories; expected one per input.')
        for i, r in enumerate(resps):
            if not isinstance(r, SampleResponse):
                raise TypeError(f'expected SampleResponse at batch index {i}, got '
                                f'{type(r).__name__}')
            if not r.sequences:
                raise RuntimeError(f'SampleResponse at batch index {i} has no sequences')
        return resps

    def _extend_with_bridge(
        self,
        pif: Dict[str, Any],
        tool_messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Append tool messages and the next generation prompt as -100 bridge.

        Strategy: compute the bridge ENTIRELY in template space. Render
        ``messages_before`` and ``messages_before + tool_messages`` with the
        same chat template and take ``s_after[len(s_before):]`` as the delta.

        We deliberately do NOT diff against ``tokenizer.decode(pif.input_ids)``
        because raw vLLM output and canonical template rendering differ in
        whitespace (e.g. Qwen inserts ``\\n\\n`` between assistant content and
        a ``<tool_call>`` block, while the model generates only ``\\n``). Such
        cosmetic divergences would break a ``startswith`` alignment but do not
        affect training correctness: history tokens stay in ``pif.input_ids``
        verbatim; only the newly appended bridge is tokenized from the
        canonical template output.
        """
        tokenizer = self.template.tokenizer

        messages_before = list(pif.get('messages') or [])
        messages_after = messages_before + list(tool_messages)

        enable_thinking = getattr(self.template, 'enable_thinking', False)
        s_before = tokenizer.apply_chat_template(
            messages_before, tokenize=False, add_generation_prompt=False, enable_thinking=enable_thinking)
        s_after = tokenizer.apply_chat_template(
            messages_after, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)

        if not s_after.startswith(s_before):
            raise RuntimeError('Canonical chat_template output for messages_after is not a '
                               'prefix-extension of messages_before; cannot compute bridge '
                               'delta. This indicates the template is non-monotonic in the '
                               'message list (e.g. reorders / rewrites earlier turns).\n'
                               f's_before tail: {s_before[-80:]!r}\n'
                               f's_after at same offset: '
                               f'{s_after[max(0, len(s_before) - 80):len(s_before) + 80]!r}')
        bridge_text = s_after[len(s_before):]
        if not bridge_text:
            raise RuntimeError('Bridge text computation returned empty string; '
                               'tool turn would add no tokens (template misconfiguration?).')

        bridge_ids = tokenizer.encode(bridge_text, add_special_tokens=False)
        if not bridge_ids:
            raise RuntimeError(f'Bridge text tokenised to empty id list: {bridge_text!r}')

        new_pif = self._append_bridge_tokens(pif, bridge_ids)
        new_pif['messages'] = messages_after
        return new_pif

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
                raise RuntimeError(f'labels length ({len(labels)}) != input_ids length '
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
            pad = torch.zeros((mm.shape[0], len(bridge_ids)), dtype=mm.dtype, device=mm.device)
            result['mm_token_type_ids'] = torch.cat([mm, pad], dim=1)

        # Replay the post pipeline: refresh attention_mask / position_ids /
        # length and re-roll labels back into output/shifted order.
        refreshed = self.template._invoke_post_pipeline([result])[0]
        result.update(refreshed)
        return _to_plain(result)
