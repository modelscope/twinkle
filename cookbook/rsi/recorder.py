# Copyright (c) ModelScope Contributors. All rights reserved.
"""What a collection pass writes, shared by both halves of the loop.

The agentic and the code half invent completely different problems, but a
trajectory is a trajectory: token fields to ``.npz``, everything a reader needs
to interpret them to ``trajs/index.jsonl``, and train.py reads that one index
without caring which half produced a line. Keeping one writer is what makes
``side`` a plain field rather than two file formats to reconcile.

In cookbook rather than in :mod:`twinkle_agentic` on purpose: this is the
on-disk contract between a collection pass and the step that trains on it, and
that contract is still moving -- fields get added as questions come up about
runs. A library version would freeze it, and the freezing is the expensive part,
not the code.
"""
import json
import os
import threading
from typing import Any, Dict, List

import numpy as np


def logprob_column(logprobs: Any) -> List[float]:
    """One float per generated token: the logprob of the token that was chosen.

    The sampler hands these over as ``List[List[Tuple[int, float]]]`` -- per
    generated token, a list of top-k ``(token_id, logprob)`` pairs with the chosen
    token first (``SampledSequence.logprobs``, data_format/sampling.py:185).
    Passing that to ``np.asarray`` directly would store an ``(N, k, 2)`` array and
    the loader would hand GRPO nested lists where it wants one float per trainable
    token -- which is a crash inside the step, or worse a silent reshape.

    A plain list of floats is accepted too, for a sampler that already flattened.
    Anything else raises rather than being coerced: a wrong ``old_logps`` makes the
    GRPO ratio wrong on the first step, and nothing downstream would say so.
    """
    out: List[float] = []
    for step in logprobs:
        if isinstance(step, (int, float)):
            out.append(float(step))
            continue
        if isinstance(step, (list, tuple)) and step:
            head = step[0]
            if isinstance(head, (list, tuple)) and len(head) >= 2:
                out.append(float(head[1]))
                continue
        raise TypeError(f'cannot read a logprob out of {step!r}; expected a float '
                        f'or a list of (token_id, logprob) pairs')
    return out


class Recorder:
    """Everything a run writes, behind one lock.

    Trajectories go to ``.npz`` for the token fields and to ``index.jsonl`` for
    everything a reader needs to interpret them. The text is written in full and
    never truncated: these files are read to check whether a reward was deserved,
    which a shortened statement cannot answer.

    Both halves of an iteration share one instance, so the numbering is global
    and the index interleaves them. That is also why every handle is opened up
    front even when the half in front of it has nothing to put in some of them:
    a file that appears only sometimes is a file every reader has to guard.
    """

    def __init__(self, out_dir: str):
        self.dir = out_dir
        self.traj_dir = os.path.join(out_dir, 'trajs')
        os.makedirs(self.traj_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._n = 0
        self._index = open(os.path.join(self.traj_dir, 'index.jsonl'), 'w', encoding='utf-8')
        self._groups = open(os.path.join(out_dir, 'groups.jsonl'), 'w', encoding='utf-8')
        self._tasks = open(os.path.join(out_dir, 'tasks.jsonl'), 'w', encoding='utf-8')
        # Why a build produced no task. The reason alone is not diagnosable: nine
        # empty_workspace rejections in one run all looked like the model refusing
        # to act, and the question of whether it had run out of tokens or simply
        # emitted no call could not be answered from the record, because the fields
        # that answered it were on the trajectory and were dropped.
        self._rejected = open(os.path.join(out_dir, 'rejected.jsonl'), 'w', encoding='utf-8')
        # Keyword replies, both sides in full. The one question this file exists to
        # answer -- did the model disobey the format, or does the parser reject what
        # it produced -- cannot be answered from a count. Keyword generation was
        # silently broken for whole runs when the prompt asked for one per line and
        # the parser wanted a JSON array.
        self._keywords = open(os.path.join(out_dir, 'keyword_gen.jsonl'), 'w', encoding='utf-8')
        # Every solver attempt, passed or not, with the state it left and what the
        # check said about it. A task measured at 0 of 8 has three explanations --
        # the check is wrong, the statement withholds something the check demands,
        # or the solver gave up -- and only the attempt and the workspace it left
        # tell them apart. Written for every attempt, not only for the ones that
        # end up trained on: the failures are what this file is for.
        self._attempts = open(os.path.join(out_dir, 'solver_attempts.jsonl'), 'w',
                              encoding='utf-8')
        # The rubric, all three of its dimensions. Only novelty reaches a reward;
        # usefulness and complexity are recorded so the question of whether they
        # should count can be answered from a run instead of argued.
        self._novelty = open(os.path.join(out_dir, 'novelty_scores.jsonl'), 'w',
                             encoding='utf-8')

    def trajectory(self, traj: Dict[str, Any], **fields: Any) -> None:
        """One training sample: token fields to npz, everything else to the index.

        A trajectory with no ``logprobs`` is written anyway, with the field left
        null. It is not trainable and the loader will say so -- which is the point:
        a sample silently dropped here would make the group it belongs to look like
        a different size than it was.
        """
        input_ids = np.asarray(traj.get('input_ids') or [], dtype=np.int32)
        labels = np.asarray(traj.get('labels') or [], dtype=np.int32)
        logprobs = traj.get('logprobs')
        with self._lock:
            self._n += 1
            name = f'{self._n:06d}.npz'
        arrays = {'input_ids': input_ids, 'labels': labels}
        if logprobs is not None:
            # float64, and the chosen token's column only. These are the old_logps a
            # GRPO step divides by; float32 would round them to about 7 digits, so
            # the ratio exp(logp - old_logp) would be off by roughly 1e-7 for
            # reasons that have nothing to do with the policy having changed.
            arrays['logprobs'] = np.asarray(logprob_column(logprobs), dtype=np.float64)
        # Compressed: a 24-turn agentic episode is tens of thousands of token ids,
        # and 128 of them per iteration adds up on disk.
        np.savez_compressed(os.path.join(self.traj_dir, name), **arrays)
        record = dict(fields)
        record.update({
            'npz': name,
            'n_tokens': int(input_ids.size),
            'n_trainable': int((labels != -100).sum()) if labels.size else 0,
            'has_logprobs': logprobs is not None,
            # The rollout guarantees one logprob per trainable label; recorded so a
            # loader can check it rather than trust it.
            'n_logprobs': int(arrays['logprobs'].size) if logprobs is not None else 0,
            'turns': traj.get('turns'),
            'stop_reason': traj.get('stop_reason'),
            'truncated': bool(traj.get('truncated')),
            'tool_stop': traj.get('tool_stop'),
            'messages': traj.get('messages') or [],
        })
        self._write(self._index, record)

    def group(self, record: Dict[str, Any]) -> None:
        self._write(self._groups, record)

    def task(self, record: Dict[str, Any]) -> None:
        self._write(self._tasks, record)

    def rejected(self, record: Dict[str, Any]) -> None:
        self._write(self._rejected, record)

    def keywords(self, record: Dict[str, Any]) -> None:
        self._write(self._keywords, record)

    def attempt(self, record: Dict[str, Any]) -> None:
        self._write(self._attempts, record)

    def novelty(self, record: Dict[str, Any]) -> None:
        self._write(self._novelty, record)

    def close(self) -> None:
        for handle in (self._index, self._groups, self._tasks, self._rejected,
                       self._keywords, self._attempts, self._novelty):
            handle.close()

    def _write(self, handle, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            handle.write(line + '\n')
            handle.flush()
