# Copyright (c) ModelScope Contributors. All rights reserved.
"""Where the challenger's variety comes from.

The challenger knows how to work in a sandbox, verify what it built and state a
task; it does not know what the task should be about. That choice is a seeder:
called once per round, it returns the sentence or two appended to the
challenger's own opening instruction, or nothing at all.

Keeping it out here is what lets a run add a kind of variety the challenger has
never heard of -- a keyword pool, earlier trajectories, a difficulty ladder --
without the challenger growing a parameter per kind.
"""
from typing import List, Optional, Sequence

__all__ = ['Seeder', 'ChainSeeder']


class Seeder:
    """A source of opening variety, asked once per proposing round.

    Returns the text to hand the challenger, or ``None`` when this round has
    nothing to offer -- an exhausted pool, an empty seed set -- which the
    challenger reads as "propose from scratch". So an unhelpful round costs a
    plainer prompt, not a failed round.

    The text is *appended* to the challenger's instruction, so it says what to
    build around and nothing about the mechanics of doing it: the challenger
    already told the model it has tools and must not describe the task yet.
    """

    def __call__(self) -> Optional[str]:
        raise NotImplementedError(f'{type(self).__name__} does not produce seeds')


class ChainSeeder(Seeder):
    """Several seeders as one, their texts joined in the order given.

    A round takes what each member offers and skips the ones offering nothing,
    so a pool running dry narrows the prompt instead of ending the run; ``None``
    only when every member declined. Order is the caller's, and it matters: the
    members are read as one paragraph after another.
    """

    def __init__(self, seeders: Sequence[Seeder], separator: str = '\n\n'):
        self.seeders = list(seeders)
        if not self.seeders:
            raise ValueError('ChainSeeder needs at least one seeder')
        self._separator = separator

    def __call__(self) -> Optional[str]:
        parts: List[str] = []
        for seeder in self.seeders:
            text = seeder()
            if text and text.strip():
                parts.append(text.strip())
        return self._separator.join(parts) if parts else None
