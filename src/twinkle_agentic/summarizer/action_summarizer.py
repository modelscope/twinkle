# Copyright (c) ModelScope Contributors. All rights reserved.
"""Action-level summarizer: compress ONE agent turn into a single-line
``(action, goal, result-state)`` gist for downstream trajectory segmentation.

Unlike :class:`FactSummarizer` (which preserves *facts* for retrieval), this
summarizer preserves *intent* — what the agent tried to do this turn and
whether it worked — because that is what a segmenter needs to group turns into
sub-goals. The gist is deliberately tiny (one line) so a whole long trajectory
can be laid out and segmented in a single LLM pass.

Same machinery as every other component: the shared ``_sample`` is decorated
with ``@llm_backup`` so student/teacher routing + progressive distillation
happen transparently.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from twinkle_agentic.summarizer.base import Summarizer

if TYPE_CHECKING:
    from twinkle.data_format import SamplingParams  # noqa: F401
    from twinkle.sampler.base import Sampler  # noqa: F401


_ACTION_SCHEMA = """\
You compress ONE turn of an AI agent's trajectory into a single short line \
that captures INTENT, not facts. A downstream segmenter reads these lines to \
group consecutive turns into sub-goals, so keep the action's PURPOSE and \
OUTCOME, and drop retrieved content / long results.

Output EXACTLY one line in this shape (<= 140 chars):
    <ACTION> | goal: <what this turn is trying to achieve> | result: <ok|fail|partial|pending|answer>

Where <ACTION> is a terse verb-phrase for the turn, e.g. one of:
    search, read, compute, call-tool:<name>, plan, reason, ask-user, final-answer, other

Rules:
- Focus on WHY the turn happened and WHETHER it advanced the task.
- result=ok (tool/step succeeded), fail (error/empty), partial (some progress),
  pending (awaiting more), answer (produced a user-facing final answer).
- Do NOT copy retrieved passages, numbers or long tool output — only the gist.
- One line only. No markdown, no extra commentary.
"""

_ACTION_USER_TEMPLATE = """\
Summarize this single agent turn into ONE intent line (see the required shape). \
Capture the action, its goal, and the result-state. Do not exceed {budget} chars. \
Ignore long retrieved content; keep only what the turn was DOING.

## Task / query (context)
{query}

## Turn
{text}"""


class ActionSummarizer(Summarizer):
    """Compress a single rendered turn into a one-line intent gist.

    Defaults target a tiny fixed budget (one line) regardless of turn length,
    which is what makes a whole-trajectory single-pass segmentation cheap.
    """

    def __init__(
        self,
        sampler: 'Sampler',
        compression_ratio: float = 6.0,
        *,
        model_path: str = '',
        sampling_params: 'SamplingParams | None' = None,
        min_budget_chars: int = 140,
        template: Any | None = None,
        lora_path: str | None = None,
    ):
        super().__init__(
            sampler,
            compression_ratio,
            model_path=model_path,
            sampling_params=sampling_params,
            system_prompt=_ACTION_SCHEMA,
            user_prompt_template=_ACTION_USER_TEMPLATE,
            min_budget_chars=min_budget_chars,
            template=template,
            lora_path=lora_path,
        )
