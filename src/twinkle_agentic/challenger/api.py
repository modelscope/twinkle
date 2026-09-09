# Copyright (c) ModelScope Contributors. All rights reserved.
"""An OpenAI-compatible API, reached as if it were one more explorer.

Both halves of this loop hand some rounds to a stronger model over an API -- the
ones that are answers rather than actions: writing a check script, describing a
task, brainstorming keywords. The rule that decides what may go over the API is
that the reply must not enter a trainable trajectory, and these do not.

The call itself was written twice, here and in ``cookbook/rsi``, in the same
twenty lines each time: append a user message, send, keep the text, survive a
raised exception. :class:`ApiModel` is that call, once. :class:`ApiExplorer` wraps
it in the :data:`~.base.Explorer` signature, so anything here that takes an
explorer -- the keyword bank above all -- can be pointed at the API without
knowing it is one, and can be handed a local explorer to fall back on when the
API is unreachable.
"""
from typing import Any, Dict, List, Optional, Sequence

from twinkle.data_format import SamplingParams, Trajectory
from twinkle.utils import get_logger
from .base import Explorer

logger = get_logger()

__all__ = ['ApiExplorer', 'ApiModel']


class ApiModel:
    """One OpenAI-compatible client, the extra body it always sends, and one call.

    Args:
        api: the client, called as ``api(request, params)``, or with
            ``extra_body=`` when there is one. ``twinkle_agentic.protocol.openai``
            provides one; anything with that signature will do.
        extra_body: sent on every call (e.g. ``{'thinking_budget': N}`` to cap a
            reasoning model). ``None`` sends the request unmodified.
        name: what log lines call this model, normally the caller's class name.
    """

    def __init__(self, api: Any, *, extra_body: Optional[Dict[str, Any]] = None,
                 name: str = 'api'):
        self.api = api
        self.extra_body = dict(extra_body) if extra_body else None
        self.name = name

    def generate(self, messages: Sequence[Dict[str, Any]],
                 params: Optional[SamplingParams] = None) -> Optional[str]:
        """One reply to ``messages``, as text. ``None`` means the call raised.

        Returning ``None`` rather than raising is what keeps one unreachable call
        from ending a run that has hours of sandbox work behind it; every caller
        here either rejects that one item or falls back.

        Tools are withdrawn for these rounds on purpose -- they are answers, not
        actions -- so only the text is kept and any structured ``tool_calls`` the
        API returned are dropped.
        """
        request: Trajectory = {'messages': list(messages)}
        try:
            if self.extra_body:
                reply = self.api(request, params, extra_body=self.extra_body)
            else:
                reply = self.api(request, params)
        except Exception as exc:  # noqa: BLE001 -- one bad call must not kill the run
            logger.warning(f'[{self.name}] API call failed: {type(exc).__name__}: {exc}')
            return None
        if isinstance(reply, list):
            reply = reply[0] if reply else {}
        return (reply.get('content') if isinstance(reply, dict) else None) or ''

    def reply(self, messages: List[Dict[str, Any]], user_text: str,
              params: Optional[SamplingParams] = None) -> Optional[str]:
        """Append ``user_text`` and one reply to ``messages``; return the reply.

        For the staged conversations: a check script asked for over the end state,
        then a statement asked for over the check. ``messages`` is the caller's
        private copy, never a trainable trajectory, so mutating it in place costs
        the model nothing. A failed call leaves the user message appended and no
        assistant message, which is what the caller would have to write out by
        hand to retry.
        """
        messages.append({'role': 'user', 'content': user_text})
        content = self.generate(messages, params)
        if content is None:
            return None
        messages.append({'role': 'assistant', 'content': content})
        return content


class ApiExplorer:
    """An :data:`~.base.Explorer` that answers single text rounds over an API model.

    For the keyword bank, whose calls are one round each and whose replies are
    parsed into a list and thrown away: no tokens of them are ever trained on, so
    a stronger model may write them. That matters more than it sounds. The bank is
    the single input every task downstream is built from, and a 4B policy at the
    temperature diversity needs is the wrong instrument for a category rule list
    this long -- measured over 1344 locally generated keywords, 31% of one
    category named an activity where the rules asked for a computation, and 24% of
    another needed hardware the sandbox does not have.

    Args:
        model: the :class:`ApiModel` to ask.
        params: sampling params for these calls. A per-call ``sampling_params``
            overrides them, so a caller that already sizes its own calls keeps
            doing so.
        fallback: local explorer for whichever prompts the API could not answer.
            Without one, a failed call comes back as a trajectory with no
            assistant message, which every parser here reads as an empty reply.
            With one, an unreachable API cannot leave a keyword category dry --
            and dry means keyword-less prompts and a run that looks healthy while
            producing one prompt over and over, the exact failure the bank's
            refill logic exists to prevent.

    Every returned trajectory carries ``via``: ``'api'`` or ``'local-fallback'``.
    It is the one thing a reader of the keyword dump cannot reconstruct afterwards,
    and the two halves answer at measurably different quality.
    """

    def __init__(self, model: ApiModel, *, params: Optional[SamplingParams] = None,
                 fallback: Optional[Explorer] = None):
        self.model = model
        self.params = params
        self.fallback = fallback

    def __call__(self, prompts: Sequence[Trajectory],
                 sampling_params: Optional[SamplingParams] = None) -> List[Trajectory]:
        """One API call per prompt, in order, then the failures in one local batch.

        Serially, where a local explorer would take the whole batch at once:
        nothing here knows the API's rate limit, and firing a 32-call expansion at
        it is how that gets discovered. The failures are gathered and handed to the
        fallback together, because a local sampler shards a batch over its workers
        and one prompt at a time would leave most of them idle.
        """
        params = sampling_params or self.params
        out: List[Optional[Trajectory]] = []
        failed: List[int] = []
        for prompt in prompts:
            messages = [dict(m) for m in prompt.get('messages') or []]
            content = self.model.generate(messages, params)
            if content is None:
                failed.append(len(out))
                out.append(None)
                continue
            messages.append({'role': 'assistant', 'content': content})
            out.append({'messages': messages, 'via': 'api'})
        if failed and self.fallback is not None:
            local = self.fallback([prompts[i] for i in failed])
            for i, trajectory in zip(failed, local):
                answered = dict(trajectory)
                answered['via'] = 'local-fallback'
                out[i] = answered
        return [t if t is not None else {'messages': [], 'via': None} for t in out]
