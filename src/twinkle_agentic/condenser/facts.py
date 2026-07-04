# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from twinkle_agentic.condenser.base import Condenser

if TYPE_CHECKING:
    from twinkle.data_format import SamplingParams  # noqa: F401
    from twinkle.sampler.base import Sampler  # noqa: F401

_SECTION_SCHEMA = """You are a text compression assistant. A downstream model will read your compressed output to decide whether the detail it needs is inside this block; if yes, it will fetch and read the original passage.

Downstream model workflow:
Read your compressed output -> Decide whether needed info is in this block -> If yes -> Fetch original.

Therefore your compression MUST NOT lose major information from the source.

Output format:

```text
## Summary
Overview plus facts STRONGLY RELATED to the Query, stated explicitly.

## More
A collapsed index; expansion required to see specific information.
```

Rules:
1. Telegraphic style — drop function words ("the", "a", "is", "are", "of", ...); colons and commas mean "is" / "has".
    * Exception: KEEP role-tagging verb+preposition phrases verbatim ("published by X", "written by X", "directed by X", "starring X", "founded by X", "created by X", "composed by X", "produced by X", "based on X", "adapted from X"). Collapsing these to a bare name loses the relation role (author vs publisher vs director) that the downstream question may hinge on.
2. Summary MUST contain the passage's primary topic + 2–4 concrete core facts drawn from the source (entities, numbers, dates, relations). If a Query is given, order Query-relevant facts first, but STILL include other core facts within the budget. A Query is an ORDERING HINT, NOT a filter.
3. Summary MUST NOT be meta-commentary about the Query. Forbidden patterns: "no X mention", "Query info: absent", "passage covers Y only", "does not contain ...", "no relevant info", or summaries that are only abstract category words like "structure/order/usage" with no facts. If the passage is unrelated to the Query, you still summarize the passage normally.
4. More is an INDEX of category keywords, NOT inline data. Enumerate what CAN be recovered from the source (e.g. "birthplace, death place, age"); do NOT paste dates/numbers/names inline. Make sure all category of useful facts are introduced here.
5. Output language MUST match the source language.
6. Do NOT fabricate. Do NOT omit major information. Any fact not in the source MUST NOT appear in your output.

Now begin.
""" # noqa

_SECTION_USER_TEMPLATE = """\
Downstream model will read your compressed block to decide whether to \
expand it. Compress faithfully: preserve the passage topic + core facts. \
Do NOT invent facts. Do NOT drop major facts. Do NOT write meta-commentary \
about the Query (never write "Query info: absent", "no X mention", etc.); \
if the passage does not address the Query, still summarize the passage.

## Query (ordering hint only — still summarize the whole passage)
{query}

## Target length
Compress AS MUCH AS faithfully possible. HARD CEILING: {budget} chars. \
If core facts fit in far fewer chars, output fewer. \
Never exceed the ceiling.

## Passage
{text}"""


class FactsCondenser(Condenser):

    def __init__(
        self,
        sampler: Sampler,
        compression_ratio: float = 2.0,
        *,
        model_path: str = '',
        sampling_params: SamplingParams | None = None,
        min_budget_chars: int = 250,
        template: Any | None = None,
        lora_path: str | None = None,
    ):
        super().__init__(
            sampler,
            compression_ratio,
            model_path=model_path,
            sampling_params=sampling_params,
            system_prompt=_SECTION_SCHEMA,
            user_prompt_template=_SECTION_USER_TEMPLATE,
            min_budget_chars=min_budget_chars,
            template=template,
            lora_path=lora_path,
        )
