# Copyright (c) ModelScope Contributors. All rights reserved.
"""Model-free trajectory transforms: field write-back, row filtering, and the privileged teacher view.

This is a decoration carrier for plain-value functions, not a transport service: the field
contracts and validation live in ``twinkle.data_format.trajectory``. ``TrajectoryUpdate`` writes
columns onto copies of the trajectories; ``TrajectoryFilter`` selects rows by a caller-supplied
rule. Both return ``(new_trajs, summary)`` -- the caller replaces its old references with the new
ones; nothing mutates an already-published ObjectRef. The summary stays light (row identity,
lengths, versions, named scalars) so it rides the collect while the stashed trajectories stay
references. :class:`PrivilegedView` re-encodes rows under their ``teacher_prompt`` column for the
teacher forward and publishes the encoded rows one by one.
"""
import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from twinkle import remote_class, remote_function

from twinkle.data_format import Trajectory, ground_truth, turn_token_spans, validate_trajectory
from twinkle.signals import response_positions


def batch_summary(trajs: Sequence[Trajectory]) -> Dict[str, Any]:
    """Light batch facts only: row identity, lengths, versions, truncation -- never payloads.

    Row-level facts (per-row lengths, ``truncated`` / ``turns``) are small metadata the consumer
    folds into its rollout metrics without materialising the trajectories; the aggregates are
    kept too so a caller that only logs totals reads one key. ONE summary shape for every
    row-based producer here (the streaming driver's own delivery facts are sample-based -- its
    rows are references it never opens).
    """
    response_lengths = [len(traj.get('response_token_ids') or ()) for traj in trajs]
    prompt_lengths = [(len(traj.get('input_ids') or ()) - length if traj.get('input_ids') is not None else 0)
                      for traj, length in zip(trajs, response_lengths)]
    versions = sorted({v for v in (traj.get('policy_version') for traj in trajs) if v is not None})
    truncated = [bool(traj.get('truncated')) for traj in trajs]
    # The named ORM score column (scorer name -> one float-or-None per row): the one payload-adjacent
    # statistic a consumer legitimately needs without materialising the trajectories -- reward
    # metrics. Every other fact here is length / identity / version metadata.
    orm_scores: Dict[str, List[Any]] = {}
    for traj in trajs:
        for name, value in ((traj.get('rewards') or {}).get('orm') or {}).items():
            orm_scores.setdefault(name, []).append(value)
    return {
        'num_rows': len(trajs),
        'episode_ids': [traj.get('episode_id') for traj in trajs],
        'prompt_lengths': prompt_lengths,
        'response_lengths': response_lengths,
        'response_length_min': min(response_lengths) if response_lengths else 0,
        'response_length_max': max(response_lengths) if response_lengths else 0,
        'response_length_sum': sum(response_lengths),
        'policy_versions': versions,
        'truncated': truncated,
        'num_truncated': sum(truncated),
        'turns': [traj.get('turns') for traj in trajs],
        'orm_scores': orm_scores,
    }


@remote_class()
class TrajectoryUpdate:
    """The remote carrier for field write-back: scored rewards and identity onto the trajectories."""

    @remote_function(execute='first', collect='first', stash=(0,))
    def update(self, trajs: List[Trajectory], *,
               updates: List[Dict[str, Any]]) -> Tuple[List[Trajectory], Dict[str, Any]]:
        """Write fields onto copies of ``trajs`` and return ``(new_trajs, summary)``.

        ``updates`` aligns row for row with ``trajs``; a mismatched count or a field update that
        breaks the response-frame contract (token-level fields must match each row's response
        width) fails loudly instead of silently miswriting.
        """
        if updates is None or len(updates) != len(trajs):
            raise ValueError(f'updates ({0 if updates is None else len(updates)}) must match '
                             f'the trajectory rows ({len(trajs)})')
        new_trajs = []
        for traj, row_updates in zip(trajs, updates):
            merged = dict(traj)
            for key, value in (row_updates or {}).items():
                merged[key] = value
            validate_trajectory(merged)
            new_trajs.append(merged)
        return new_trajs, batch_summary(new_trajs)


@remote_class()
class TrajectoryFilter:
    """The remote carrier for row selection: one caller-supplied rule picks the rows that train."""

    @remote_function(execute='first', collect='first', stash=(0,))
    def select(self, trajs: List[Trajectory], *,
               indices: Optional[List[int]] = None,
               where: Optional[Callable[[Trajectory], bool]] = None
               ) -> Tuple[List[Trajectory], Dict[str, Any]]:
        """Select rows by ``indices`` or a ``where`` rule as ``(kept, summary)``.

        ``where`` is one row-level rule (``lambda traj: ...``); it runs here in the worker on the
        materialised rows, so it may read any trajectory field -- the nonempty-response rule is
        ``where=lambda traj: bool(traj.get('response_token_ids'))``. The summary describes the
        original batch as well, so a caller filtering rows can log both sides without
        materialising them; ``indices`` reports which rows were kept, so a row-aligned side
        column (the driver's score rows) can be sliced to match.
        """
        if indices is not None and where is not None:
            raise ValueError('select takes one rule: indices or where, not both.')
        if indices is None:
            if where is None:
                indices = list(range(len(trajs)))
            else:
                indices = [i for i, traj in enumerate(trajs) if where(traj)]
        seen = set()
        for index in indices:
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(trajs):
                raise ValueError(f'select index {index!r} is not a valid row of {len(trajs)}')
            if index in seen:
                raise ValueError(f'select index {index!r} repeats: selection must be unique')
            seen.add(index)
        kept = [trajs[index] for index in indices]
        summary = dict(batch_summary(kept))
        summary['num_dropped'] = len(trajs) - len(kept)
        # Which rows survived, in delivery order: a caller holding a row-aligned side column (the
        # driver's score rows) slices it with these to match the rows that train.
        summary['indices'] = list(indices)
        return kept, summary


# ------------------------------------------------------------ privileged teacher view

_MULTIMODAL_KEYS = ('images', 'videos', 'audios', 'tools', 'objects', 'chat_template_kwargs')

# What a privileged row inherits from its student trajectory: identity and the original response
# ids (the encode-and-check tail guarantees the re-encoded tokens are exactly those ids).
_PRIVILEGED_KEYS = ('episode_id', 'group_key', 'policy_version', 'role',
                    'response_token_ids', 'tokenizer_fingerprint')


def encode_privileged_view(
    template: Any,
    messages: List[dict],
    response_ids: List[int],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Encode a privileged teacher view (``messages`` carrying the shared ``response_ids``) and assert alignment.

    Shared by GRPO's RLSD/SDAR teacher view and OPSD's, which differ only in HOW ``messages`` is
    built (multi-turn reconstruction vs. a single privileged user turn) but not in the
    encode-and-check tail. Encoded in ``train`` mode with ``add_eos=False`` so the assistant turn
    is exactly ``response_ids`` (no extra end-of-sequence token), then checked: the encoded
    feature must score the SAME response tokens -- the ``response_positions`` count and every
    position's label must match ``response_ids``. A template or tokenizer difference that changes
    any of them would silently grade different tokens -- a hard error, not a truncation.
    """
    if template is None:
        raise ValueError('teacher_prompt requires a template to build the privileged teacher view.')
    extra = extra or {}
    row: Dict[str, Any] = {'messages': messages, 'add_eos': False}
    for key in _MULTIMODAL_KEYS:
        if extra.get(key) is not None:
            row[key] = extra[key]
    teacher_template = copy.copy(template)
    teacher_template.set_mode('train')
    feature = teacher_template.encode(row)
    positions = response_positions(feature)
    if len(positions) != len(response_ids):
        raise RuntimeError('teacher_prompt encoding changed the response-token count; teacher and student must '
                           'share a tokenizer and score exactly the same response tokens.')
    labels = feature.get('labels')
    if labels is not None and any(labels[pos] != token for pos, token in zip(positions, response_ids)):
        raise RuntimeError('teacher_prompt encoding changed the response tokens; the teacher view must pin the '
                           "student's exact sampled tokens.")
    return feature


def encode_privileged_feature(template: Any, prompt_messages: List[dict], teacher_prompt: Any,
                              response_ids: List[int], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Encode one single-turn privileged teacher view (RLSD / OPSD share the exact contract).

    Replaces the FIRST user turn's content with the privileged ``teacher_prompt``, appends the
    assistant turn carrying the student's response token ids verbatim, then delegates the
    encode-and-align-check tail to :func:`encode_privileged_view` -- teacher and student must
    score exactly the same response tokens.
    """
    messages = [dict(message) for message in prompt_messages]
    if messages and messages[-1].get('role') == 'assistant':
        messages = messages[:-1]
    for message in messages:
        if message.get('role') == 'user':
            message['content'] = teacher_prompt
            break
    else:
        raise ValueError('teacher_prompt requires a user turn in the prompt to carry the privileged context.')
    messages.append({'role': 'assistant', 'content': list(response_ids)})
    return encode_privileged_view(template, messages, response_ids, extra)


def _messages_with_token_ids(messages: List[dict], response_ids: List[int],
                             encoded: Dict[str, Any]) -> List[dict]:
    """Splice each assistant turn's exact sampled tokens back into the messages.

    The teacher view re-encodes under a different prompt but must score the *same* tokens the
    policy produced, so each turn carries its ids verbatim rather than its text (re-encoding text
    can drift to different ids). ``encoded`` is the policy's flat feature: ``turn_token_spans``
    splits ``response_ids`` into one contiguous slice per assistant turn, and the newest slice
    maps to the newest assistant message.
    """
    turn_token_ids = [list(response_ids[start:end]) for start, end in turn_token_spans(encoded)]
    result = [dict(message) for message in messages]
    completion_index = 0
    for message in reversed(result):
        if message.get('role') != 'assistant':
            continue
        if completion_index >= len(turn_token_ids):
            break
        ids = turn_token_ids[-1 - completion_index]
        message['content'] = {'token_ids': ids, 'loss_scale': [1] * len(ids)}
        completion_index += 1
    if completion_index != len(turn_token_ids):
        raise RuntimeError(
            f'multi-turn messages contain {completion_index} generated assistant turns, but the '
            f'trajectory has {len(turn_token_ids)} trainable token runs. A run-splitting mismatch '
            'means an assistant turn produced no trainable token (an empty generation) or a '
            'completion_mask split one turn in two.')
    return result


@remote_class(execute='first')
class PrivilegedView:
    """Re-encode each response under its privileged ``teacher_prompt`` column, for the teacher forward.

    A trajectory without a ``teacher_prompt`` is returned unchanged; with one, the SAME response
    tokens are re-encoded from the privileged view (the first user turn carries the hint), so the
    teacher grades the policy's answer hint-augmented. Single-turn rows go through
    :func:`encode_privileged_feature`, multi-turn rows pin every assistant turn to its sampled ids.
    The encoded rows keep each student row's identity and original response ids; the policy's own
    trajectories are never touched. ``stash=True`` publishes the rows one by one so the teacher
    forward consumes them as references.
    """

    def __init__(self, *, template: Any):
        self.template = template

    @remote_function(dispatch='all', collect='first', lazy_collect=True, stash=True)
    def __call__(self, trajs: List[Trajectory], **kwargs) -> List[Dict[str, Any]]:
        features = []
        for traj in trajs:
            teacher_prompt = ground_truth(traj, 'teacher_prompt') or traj.get('teacher_prompt')
            if not teacher_prompt:
                features.append(traj)
                continue
            response_ids = [int(t) for t in traj['response_token_ids']]
            messages = list(traj.get('messages') or [])
            assistant = [i for i, message in enumerate(messages) if message.get('role') == 'assistant']
            if len(assistant) == 1:
                feature = encode_privileged_feature(
                    self.template, messages[:assistant[0]], teacher_prompt, response_ids, traj)
            else:
                pinned = [dict(message) for message in messages]
                for message in pinned:
                    if message.get('role') == 'user':
                        message['content'] = teacher_prompt
                        break
                else:
                    raise ValueError('teacher_prompt requires a user turn in the prompt to carry the '
                                     'privileged context.')
                feature = encode_privileged_view(
                    self.template, _messages_with_token_ids(pinned, response_ids, traj), response_ids, traj)
            for key in _PRIVILEGED_KEYS:
                if key in traj:
                    feature[key] = traj[key]
            features.append(feature)
        return features
