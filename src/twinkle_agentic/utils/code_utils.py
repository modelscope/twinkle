# Copyright (c) ModelScope Contributors. All rights reserved.
"""Reading code back out of a model's reply.

A model asked for a python snippet -- a check script, a solution, a repro --
fences it. Taking that fence back off is the same work whatever the snippet is
*for*, so it lives here rather than in one challenger.

A reply that fenced nothing is rejected, not read some other way. Reading bare
text and tool-call arguments was tried, and both come down to guessing where the
code starts and ends and then asking a parser whether the guess was plausible; a
guess that parses but is short a few lines is indistinguishable from a good one,
and it becomes a task. Requiring the fence trades those silent losses for a loud
one -- the reply is refused and the model is asked again.

Nothing here is pinned to a model family, and what *is* knowledge gets passed in
rather than assumed. The caller knows which language it asked for and says so with
``language_tags``. The caller does not know which model answered, so this module
absorbs that: reasoning is cut by a list of markers rather than the one tag a
given model emits. Handing that up to a challenger only moves the ignorance -- it
would then guess ``</think>`` and be right for one model family.

There are two ways out, and they differ on the replies that fenced no code -- no
fence at all, or one left empty. :func:`parse_fenced_code` answers None to both,
:func:`unwrap_code` hands the reply back whole for the first and ``''`` for the
second, where the model did say the code went here and put nothing there.
"""
import re
from functools import lru_cache
from typing import Optional, Pattern, Tuple

__all__ = [
    'PYTHON_TAGS',
    'parse_fenced_code',
    'strip_reasoning',
    'unwrap_code',
]

_REASONING_END_MARKERS = ('</think>', '</thinking>', '</reasoning>', '<|end_of_thought|>')
PYTHON_TAGS: Tuple[str, ...] = ('python', 'py')


@lru_cache(maxsize=None)
def _fence_re(language_tags: Optional[Tuple[str, ...]]) -> Pattern:
    """Match a fenced block, optionally restricting its language label.

    ``None`` accepts any label. Otherwise, listed tags match case-insensitively,
    with any version suffix; an unlabelled fence is accepted as well.
    """
    if language_tags is None:
        label = r'[^\r\n]*'
    else:
        alts = '|'.join(re.escape(tag) for tag in language_tags)
        label = r'(?:(?:%s)[\d.]*)?' % alts if alts else ''
    return re.compile(r'```[ \t]*%s[ \t]*\r?\n(.*?)```' % label, re.S | re.I)


def strip_reasoning(text: str) -> str:
    """``text`` with everything up to the end of the model's thinking removed.

    The last marker anywhere in the reply wins: reasoning precedes the answer, and
    a model that opens a second thought after answering is still answering last.
    Text with no marker is returned unchanged.
    """
    body = text or ''
    cut = 0
    for marker in _REASONING_END_MARKERS:
        idx = body.rfind(marker)
        if idx >= 0:
            cut = max(cut, idx + len(marker))
    return body[cut:]


def parse_fenced_code(
    text: str,
    language_tags: Optional[Tuple[str, ...]] = PYTHON_TAGS,
) -> Optional[str]:
    """Return the last matching fenced block, or None if there is none.

    Pass ``language_tags=None`` to accept any language label. The last block, not
    the first, is returned because a model often drafts a version before the final
    one, and the block it ends on is its answer.

    What is inside is taken as given -- a fence is the model saying which part is
    the code, so second-guessing it would throw away the one piece of the reply
    that was unambiguous. Whether it runs is the sandbox's answer to give.

    A fence the model opened and left empty answers None too, on the grounds that
    a caller who cannot use a missing script cannot use an empty one either. Use
    this when nothing downstream will judge the result and a wrong guess becomes a
    task.
    """
    blocks = _fence_re(language_tags).findall(strip_reasoning(text))
    return (blocks[-1].strip() if blocks else '') or None


def unwrap_code(text: str, language_tags: Tuple[str, ...] = PYTHON_TAGS) -> str:
    """``text`` with the model's packaging taken off, always a string.

    Takes the fence off if there is one and hands the reply back whole if there is
    not, on the reading that a reply to "write the code" *is* the code however it
    was dressed. An empty fence answers ``''``, because the model did mark where
    the code went and put nothing there.

    Those two are the whole difference from :func:`parse_fenced_code`, which
    answers None to both. Use this on an answer that is about to be run -- the
    sandbox is the better judge of whether that was code, and it says so with an
    exit status.
    """
    body = strip_reasoning(text)
    blocks = _fence_re(language_tags).findall(body)
    return blocks[-1].strip() if blocks else body.strip()
