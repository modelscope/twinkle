# Copyright (c) ModelScope Contributors. All rights reserved.
"""Read advisory knowledge-card/profile affinities from TWINKLE_TOPIC_AFFINITIES.

Each JSONL row names card_id, topic_path and eligible_profiles. Unknown or
missing entries produce no switch. PREFERENCE orders eligible profiles;
unreachable services are excluded. rotation() lists every supported profile
on this host, independently of the table's per-card entries.
"""
from __future__ import annotations

import json
import logging
import os

from . import world
from .profiles import SandboxProfile, default_profile_registry

logger = logging.getLogger(__name__)

PREFERENCE: tuple = (
    'lean-proof@1',
    'world-forensics@1',
    'db-analyst@1',
    'web-research@1',
    'sql-analytics@1',
    'docs-retrieval@1',
    'log-forensics@1',
    'repo-patch@1',
    'data-wrangling@1',
    'python-compute@1',
    'code-python@1',
    'knowledge-answer@1',
)

BASELINE = 'knowledge-answer@1'

SWITCHABLE = frozenset(p for p in PREFERENCE if p not in ('knowledge-answer@1', 'lean-proof@1'))

# Profiles named here but not switchable, kept visible so the reason is in one
# place rather than spread across the callers.
UNAVAILABLE = (
    ('knowledge-answer@1', 'session kind is no_tool: not a sandbox to switch into'),
    ('lean-proof@1', 'runtime lean4-mathlib-pinned has not been built'),
)

START_HOST = 'host'
START_SEEDED = 'seeded'
START_EMPTY = 'empty'


def _profile_or_none(profile_id: str | None, registry=None) -> SandboxProfile | None:
    """Resolve a registered profile, returning None for absent or unknown IDs."""
    if not profile_id:
        return None
    reg = registry or default_profile_registry()
    try:
        return reg.get(profile_id)
    except ValueError:
        return None


def starting_state(profile_id: str | None, registry=None) -> str:
    """Derive host, seeded or empty starting data from the profile manifest."""
    profile = _profile_or_none(profile_id, registry)
    if profile is None:
        return START_EMPTY
    if set(profile.allowed_tools or ()) & set(world.TOOL_NAMES):
        return START_HOST
    if any(str(step).startswith('seed-') for step in (profile.setup_manifest or ())):
        return START_SEEDED
    return START_EMPTY


def host_tools_of(profile_id: str | None, registry=None) -> tuple[str, ...]:
    """The host-answered tools a profile grants, in the registry's own order."""
    profile = _profile_or_none(profile_id, registry)
    if profile is None:
        return ()
    return tuple(name for name in (profile.allowed_tools or ()) if name in world.TOOL_NAMES)


def seed_steps_of(profile_id: str | None, registry=None) -> tuple[str, ...]:
    """The `seeds` programs a profile runs before the agent sees anything."""
    profile = _profile_or_none(profile_id, registry)
    if profile is None:
        return ()
    return tuple(step for step in (profile.setup_manifest or ()) if str(step).startswith('seed-'))


_SERVICE_BACKED = frozenset({'db-analyst@1'})


def _service_reachable(profile_id: str, timeout: float = 2.0) -> bool:
    """Probe database module configuration and TCP reachability without provisioning a role."""
    if profile_id not in _SERVICE_BACKED:
        return True
    if profile_id != 'db-analyst@1':
        return True
    if not os.environ.get('TWINKLE_WORLD_DB_MODULE', ''):
        return False
    # The same two variables `world` connects with. Reading a different pair here
    # would probe one host and then query another.
    host = os.environ.get('TWINKLE_WORLD_DB_HOST', '127.0.0.1')
    port = int(os.environ.get('TWINKLE_WORLD_DB_PORT', '5432'))
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class TopicProfiles:
    """A card id -> eligible profile list lookup, loaded from the affinity table."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or os.environ.get('TWINKLE_TOPIC_AFFINITIES', '')
        self.by_card: dict[str, list[str]] = {}
        self.by_topic: dict[str, list[str]] = {}
        self.loaded = False
        # Computed before _load, which filters candidates through it.
        self.unreachable = tuple(sorted(p for p in SWITCHABLE if not _service_reachable(p)))
        self.switchable = SWITCHABLE - set(self.unreachable)
        if self.unreachable:
            logger.warning(
                '[topic-profile] service unreachable, not switchable here: %s '
                '(their cards fall through to the next eligible profile)', ', '.join(self.unreachable))
        self._load()

    def _load(self) -> None:
        if not self.path:
            return
        if not os.path.isfile(self.path):
            # Not "every topic uses the baseline": for_card returns None, so every
            # slot stays on whatever it was built with. Saying otherwise in the log
            # would send whoever reads it looking for a switch that never happens.
            logger.warning(
                '[topic-profile] affinity table missing at %s; '
                'every card will stay on its slot\'s own profile', self.path)
            return
        n = 0
        with open(self.path, encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                chosen = [p for p in record.get('eligible_profiles', ()) if p in self.switchable]
                # Kept even when empty. `for_card` answers None for it either way,
                # but describe() counts what is here, and "N cards offered nothing
                # this machine can run" is the number a run's log most needs --
                # dropping the record hides exactly the case worth noticing.
                card_id = record.get('card_id')
                if card_id:
                    self.by_card[str(card_id)] = chosen
                topic = record.get('topic_path') or {}
                if topic.get('root') and topic.get('leaf'):
                    self.by_topic[f"{topic['root']} > {topic['leaf']}"] = chosen
                n += 1
        self.loaded = True
        logger.info('[topic-profile] loaded %d affinity record(s) from %s: %s', n, self.path, self.describe())

    def rotation(self) -> list[str]:
        """Return supported profiles in preference order, excluding unreachable services."""
        return [profile for profile in PREFERENCE if profile in self.switchable]

    def for_card(self, card) -> str | None:
        """Choose an eligible profile by card ID or source cue; None means retain the current slot."""
        candidates: list[str] = []
        card_id = getattr(card, 'card_id', None)
        if card_id and str(card_id) in self.by_card:
            candidates = self.by_card[str(card_id)]
        if not candidates:
            cue = getattr(card, 'source_cue', None)
            if cue and cue in self.by_topic:
                candidates = self.by_topic[cue]
        for profile in PREFERENCE:
            if profile in candidates and profile in self.switchable:
                return profile
        return None

    def describe(self) -> dict[str, int]:
        """How many cards route to each profile, for the run's log.

        ``stay-on-slot`` counts the cards that offered nothing switchable -- only
        the baseline, or a profile this machine cannot run. Reported rather than
        omitted: it is the number that says an affinity table has stopped matching
        the registry.
        """
        counts: dict[str, int] = {}
        for profiles in self.by_card.values():
            chosen = next((p for p in PREFERENCE if p in profiles and p in self.switchable), 'stay-on-slot')
            counts[chosen] = counts.get(chosen, 0) + 1
        return counts
