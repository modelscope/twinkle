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

import numpy as np
import torch

from twinkle import remote_class, remote_function

from twinkle.data_format import Trajectory, ground_truth, turn_token_spans, validate_trajectory


def response_positions(feature: Dict[str, Any]) -> List[int]:
    """Indices of the trainable (response) tokens in ``feature``, i.e. where ``labels != -100``.

    This is the exact frame the loss mask uses, so scores extracted at these positions (a teacher's
    log-probs, a critic's values) align one-to-one with the policy's per-token loss.
    """
    labels = feature.get('labels')
    if labels is None:
        return []
    if isinstance(labels, torch.Tensor):
        return torch.nonzero(labels.reshape(-1) != -100, as_tuple=True)[0].tolist()
    return np.flatnonzero(np.asarray(labels) != -100).tolist()


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

    @remote_function(execute='first', collect='first', lazy_collect=False, stash=True)
    def expand_prompts(self, batch, *, num_generations: int = 1, sampling_params=None):
        """Expand prompt groups at the data boundary, with distinct identities for every episode."""
        import uuid
        from twinkle.data_format import attach_user_data

        rows = batch.get('rows', []) if isinstance(batch, dict) else batch
        if not isinstance(rows, list) or not rows:
            raise ValueError('Generation requires a non-empty list of prompt rows.')
        if num_generations < 1:
            raise ValueError('num_generations must be positive.')
        expanded = []
        for prompt in rows:
            group_key = uuid.uuid4().int
            for sample_index in range(num_generations):
                row = dict(prompt)
                row.update(episode_id=uuid.uuid4().hex, group_key=group_key, sample_index=sample_index)
                if sampling_params:
                    row = attach_user_data(row, sampling_params=sampling_params)
                expanded.append(row)
        return expanded

    @remote_function(execute='first', collect='first', lazy_collect=False, stash=True)
    def pad_rows(self, template: Trajectory, *, pad_count: int) -> List[Trajectory]:
        """Inert DP pad rows: marked copies of one real row for the remainder slots.

        A rollout batch's row count carries no divisibility invariant, while the fixed
        micro-batch forward lanes require equal per-rank counts. Each pad row copies a real
        row's encoded features so the lanes can process it; the marker fields let the
        advantage (excluded from group stats) and the training-side strip drop these rows
        without inspecting content. Stashed element-wise, the pad rows stay object
        references end to end.
        """
        if pad_count < 1:
            raise ValueError('pad_rows requires a positive pad count.')
        rows = []
        for index in range(pad_count):
            row = dict(template)
            row['episode_id'] = f'__dp_pad_{index}'
            row['group_key'] = f'__dp_pad_{index}'
            row['_dp_pad'] = True
            row.pop('advantages', None)
            row.pop('returns', None)
            rows.append(row)
        return rows

    @remote_function(execute='first', collect='first', lazy_collect=False, stash=(0, 1))
    def score(self, trajs: List[Trajectory], *, scorers: Sequence[Tuple[str, Callable]]):
        """Score named columns once and publish trajectories, score rows, and light metrics."""
        from concurrent.futures import Future

        if not trajs:
            raise ValueError('Scoring requires a non-empty trajectory batch.')
        if not scorers:
            raise ValueError('Scoring requires at least one named scorer.')
        rows = [dict(traj) for traj in trajs]
        values = {}
        invalid = 0
        names = [str(name) for name, _ in scorers]
        if len(names) != len(set(names)):
            raise ValueError('Scorer column names must be unique.')
        for name, (_, scorer) in zip(names, scorers):
            if not callable(scorer):
                raise TypeError(f'Scorer {name!r} is not callable.')
            has_column = [name in ((traj.get('rewards') or {}).get('orm') or {}) for traj in trajs]
            if all(has_column):
                column = [traj['rewards']['orm'][name] for traj in trajs]
            elif any(has_column):
                raise ValueError(f'Scorer {name!r} has scored only part of the batch.')
            else:
                result = scorer(trajs)
                if isinstance(result, Future):
                    result = result.result()
                column = list(result)
                if len(column) != len(trajs):
                    raise ValueError(f'Scorer {name!r} returned {len(column)} scores for {len(trajs)} rows.')
            for row, value in zip(rows, column):
                rewards = dict(row.get('rewards') or {})
                orm = dict(rewards.get('orm') or {})
                orm[name] = value
                rewards['orm'] = orm
                row['rewards'] = rewards
            finite = [float(value) for value in column if value is not None and value == value]
            invalid += len(column) - len(finite)
            values[name] = sum(finite) / len(finite) if finite else None
        scores = self.score_rows(rows)
        return rows, scores, {'values': values, 'invalid_count': invalid}

    @staticmethod
    def score_rows(trajs: Sequence[Trajectory]) -> List[Dict[str, Any]]:
        """Extract the estimator's score-row contract at the data compute boundary."""
        return [{'rewards': dict(traj.get('rewards') or {}), 'prm_segments': traj.get('prm_segments')}
                for traj in trajs]

    @remote_function(execute='first', collect='first', lazy_collect=False, stash=(0,))
    def update(self, trajs: List[Trajectory], *,
               updates: List[Dict[str, Any]], merge_fields: Sequence[str] = ()
               ) -> Tuple[List[Trajectory], Dict[str, Any]]:
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
                if key in merge_fields:
                    value = {**(merged.get(key) or {}), **(value or {})}
                merged[key] = value
            validate_trajectory(merged)
            new_trajs.append(merged)
        return new_trajs, batch_summary(new_trajs)


@remote_class()
class TrajectoryFilter:
    """The remote carrier for row selection: one caller-supplied rule picks the rows that train."""

    @remote_function(execute='first', collect='first', lazy_collect=False, stash=(0, 1, 2))
    def select_scored(self, trajs: List[Trajectory], *, mode: str = 'threshold',
                      num_generations: int = 8, top_k: int = 1, threshold: float = 0.0,
                      signals: Sequence[Tuple[str, Any]] = (), scores: Optional[List[Dict[str, Any]]] = None):
        """Apply one score-based selection to trajectories, model channels, and score rows."""
        from twinkle.data_format.output import select_model_output

        if not trajs:
            raise ValueError('Selection requires a non-empty trajectory batch.')
        column_sets = {tuple(sorted((traj.get('rewards') or {}).get('orm') or {})) for traj in trajs}
        if len(column_sets) != 1 or len(next(iter(column_sets))) != 1:
            raise ValueError('Selection requires exactly one consistent named ORM column.')
        name = next(iter(column_sets))[0]
        rewards = [traj['rewards']['orm'][name] for traj in trajs]
        # Collected segments of one episode are selected together, never ranked as independent samples.
        episodes = {}
        for index, traj in enumerate(trajs):
            episode_id = traj.get('episode_id')
            key = ('episode', episode_id) if episode_id is not None else ('row', index)
            episodes.setdefault(key, []).append(index)
        units = list(episodes.values())
        unit_rewards = []
        for indices in units:
            values = [rewards[index] for index in indices]
            finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
            if finite and (len(finite) != len(values) or any(value != finite[0] for value in finite)):
                raise ValueError('Collected segments of an episode must carry the same outcome score.')
            unit_rewards.append(finite[0] if finite else None)
        if mode in ('best_of_n', 'top_k'):
            count = 1 if mode == 'best_of_n' else top_k
            if num_generations < 1 or count < 1:
                raise ValueError('Selection group width and top_k must be positive.')
            groups = {}
            for index, indices in enumerate(units):
                group_key = trajs[indices[0]].get('group_key')
                key = ('group', group_key) if group_key is not None else ('block', index // num_generations)
                groups.setdefault(key, []).append(index)
            selected = []
            for group in groups.values():
                valid = [index for index in group if unit_rewards[index] is not None]
                selected.extend(sorted(valid, key=lambda index: -unit_rewards[index])[:count])
        elif mode == 'threshold':
            selected = [index for index, value in enumerate(unit_rewards) if value is not None and value >= threshold]
        else:
            raise ValueError(f'Unknown selection mode {mode!r}.')
        keep = sorted(index for unit in selected for index in units[unit])
        rows = [trajs[index] for index in keep]
        if scores is None:
            scores = TrajectoryUpdate.score_rows(trajs)
        if len(scores) != len(trajs):
            raise ValueError('Score rows must align with the trajectory batch before selection.')
        lanes = [select_model_output(value, keep, len(trajs)) for _, value in signals]
        summary = {'values': {'kept': len(keep), 'dropped': len(trajs) - len(keep)}, 'invalid_count': 0}
        return rows, lanes, [scores[index] for index in keep], summary

    @remote_function(execute='first', collect='first', lazy_collect=False, stash=(0,))
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
