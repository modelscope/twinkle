# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import re
import threading
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from twinkle_agentic.condenser.base import Condenser
from twinkle_agentic.data_format import Chunk, Chunks

# ---------------------------------------------------------------------------
# spaCy lazy loader (one model per process, thread-safe)
# ---------------------------------------------------------------------------
_SPACY_MODELS: dict[str, Any] = {}
_SPACY_LOCK = threading.Lock()


def _load_spacy(name: str):
    nlp = _SPACY_MODELS.get(name)
    if nlp is not None:
        return nlp
    with _SPACY_LOCK:
        nlp = _SPACY_MODELS.get(name)
        if nlp is not None:
            return nlp
        try:
            import spacy
        except ImportError as e:
            raise ImportError('KeywordCondenser requires spaCy. Install with: '
                              '`pip install spacy && python -m spacy download en_core_web_sm`') from e
        try:
            nlp = spacy.load(name)
        except OSError as e:
            raise OSError(f'spaCy model {name!r} not found. Download with: '
                          f'`python -m spacy download {name}`') from e
        _SPACY_MODELS[name] = nlp
        return nlp


# ---------------------------------------------------------------------------
# configuration-free constants
# ---------------------------------------------------------------------------
# Entity labels dropped from keyword candidates (low recall value).
_DROP_ENT_LABELS: frozenset[str] = frozenset({'CARDINAL', 'ORDINAL', 'PERCENT', 'QUANTITY'})

# Dependency labels that introduce sub-clauses / conjuncts we do NOT want
# to pull into a single noun-phrase span.
_DROP_NP_DEPS: frozenset[str] = frozenset(
    {'relcl', 'acl', 'advcl', 'ccomp', 'xcomp', 'conj', 'cc', 'appos', 'parataxis'})

# Tokens stripped from NP boundaries.
_LEADING_STRIP_POS: frozenset[str] = frozenset({'DET', 'PUNCT'})

# Tuple-slot separator. ``|`` avoids confusion when a slot itself
# contains a comma (e.g. ``"London, England"``).
_SLOT_SEP = ' | '
_TRIPLE_SEP = '; '

_WORD_RE = re.compile(r'\w+', flags=re.UNICODE)


# ---------------------------------------------------------------------------
# NP / verb surface helpers
# ---------------------------------------------------------------------------
def _np_text(head) -> str:
    """Return the noun-phrase text headed by ``head``.

    Keeps the contiguous span from the leftmost to the rightmost kept
    token so internal punctuation (hyphens, apostrophes, slashes) is
    preserved verbatim. Drops clausal / conjunct sub-trees and trims
    leading determiners / possessive pronouns.
    """
    # Collect subtree tokens, cutting off whole clausal children.
    collected: list = []

    def _walk(tok):
        if tok is not head and tok.dep_ in _DROP_NP_DEPS:
            return
        collected.append(tok)
        for child in tok.children:
            _walk(child)

    _walk(head)
    if not collected:
        return head.text
    collected.sort(key=lambda t: t.i)

    # Strip leading det/punct and possessive pronouns.
    while collected and (collected[0].pos_ in _LEADING_STRIP_POS or
                         (collected[0].pos_ == 'PRON' and collected[0].dep_ == 'poss')):
        collected.pop(0)
    while collected and collected[-1].pos_ == 'PUNCT':
        collected.pop()
    if not collected:
        return head.text

    start, end = collected[0].i, collected[-1].i + 1
    # If the kept tokens form a contiguous span, use the original text
    # (preserves hyphens etc.). Otherwise fall back to text_with_ws.
    if end - start == len(collected):
        return head.doc[start:end].text.strip()
    return ''.join(t.text_with_ws for t in collected).strip()


def _verb_surface(verb_tok) -> str:
    """Verb text including auxiliaries (``was born``, ``has been released``)."""
    aux = [c for c in verb_tok.children if c.dep_ in ('aux', 'auxpass')]
    if not aux:
        return verb_tok.text
    tokens = sorted(aux + [verb_tok], key=lambda t: t.i)
    return ' '.join(t.text for t in tokens)


def _first_child(token, deps: Sequence[str]):
    if token is None:
        return None
    for c in token.children:
        if c.dep_ in deps:
            return c
    return None


def _strip_leading_nc(noun_chunk) -> str:
    toks = list(noun_chunk)
    while toks and (toks[0].pos_ in _LEADING_STRIP_POS or toks[0].pos_ == 'NUM' or
                    (toks[0].pos_ == 'PRON' and toks[0].tag_ in ('PRP$', 'WP$'))):
        toks.pop(0)
    while toks and toks[-1].pos_ == 'PUNCT':
        toks.pop()
    if not toks:
        return ''
    start, end = toks[0].i, toks[-1].i + 1
    if end - start == len(toks):
        return noun_chunk.doc[start:end].text.strip()
    return ''.join(t.text_with_ws for t in toks).strip()


def _word_tokens_lower(text: str) -> frozenset[str]:
    return frozenset(m.group(0).lower() for m in _WORD_RE.finditer(text))


def _word_boundary_truncate(text: str, limit: int) -> str:
    """Truncate ``text`` to ``limit`` chars at the nearest space."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    sp = cut.rfind(' ')
    trimmed = cut[:sp] if sp >= limit // 2 else cut
    return trimmed.rstrip() or cut


# ---------------------------------------------------------------------------
# extraction (pure functions on spaCy Doc)
# ---------------------------------------------------------------------------
def _extract_opening(doc, max_chars: int) -> str:
    """First non-empty sentence, word-boundary-truncated to ``max_chars``."""
    if max_chars <= 0:
        return ''
    for sent in doc.sents:
        text = sent.text.strip()
        if text:
            return _word_boundary_truncate(text, max_chars)
    return ''


def _extract_triples(doc, n: int) -> list[tuple[str, ...]]:
    """Subject-verb-object (+ optional prep-obj) triples.

    - Skips pronoun subjects (unresolved coreference is noise).
    - Preserves verb surface form (``was born`` rather than ``bear``).
    - Deduplicates on lemmas.
    """
    if n <= 0:
        return []
    out: list[tuple[str, ...]] = []
    seen: set = set()
    for sent in doc.sents:
        for verb in sent:
            if verb.pos_ not in ('VERB', 'AUX'):
                continue
            subj = _first_child(verb, ('nsubj', 'nsubjpass', 'csubj'))
            if subj is None or subj.pos_ == 'PRON':
                continue
            obj = _first_child(verb, ('dobj', 'attr', 'oprd'))
            prep = _first_child(verb, ('prep', ))
            prep_obj = _first_child(prep, ('pobj', 'pcomp')) if prep is not None else None

            subj_txt = _np_text(subj)
            verb_txt = _verb_surface(verb)

            if obj is not None and prep_obj is not None:
                triple = (subj_txt, verb_txt, _np_text(obj), f'{prep.text} {_np_text(prep_obj)}')
                key = (subj.lemma_.lower(), verb.lemma_.lower(), obj.lemma_.lower(),
                       f'{prep.text.lower()} {prep_obj.lemma_.lower()}')
            elif obj is not None:
                triple = (subj_txt, verb_txt, _np_text(obj))
                key = (subj.lemma_.lower(), verb.lemma_.lower(), obj.lemma_.lower())
            elif prep_obj is not None:
                triple = (subj_txt, f'{verb_txt} {prep.text}', _np_text(prep_obj))
                key = (subj.lemma_.lower(), f'{verb.lemma_.lower()} {prep.text.lower()}', prep_obj.lemma_.lower())
            else:
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(triple)
            if len(out) >= n:
                return out
    return out


def _extract_keywords(doc, k: int, excluded_tokens: frozenset[str]) -> list[str]:
    """Rank keyword candidates by (entity-weighted) frequency.

    - Drops pure-numeric entities (CARDINAL / ORDINAL / PERCENT / QUANTITY).
    - Skips any term whose words are all already in ``excluded_tokens``
      (so we don't repeat what the opening already says).
    - Subsumption dedup: drops a shorter form if a longer form
      containing it is already kept (``"Nolan"`` dropped when
      ``"Christopher Nolan"`` is present).
    """
    if k <= 0:
        return []
    counts: dict[str, float] = {}
    order: dict[str, int] = {}
    idx = 0

    def _add(term: str, weight: float) -> None:
        nonlocal idx
        t = term.strip()
        if len(t) < 2:
            return
        words = [w.lower() for w in _WORD_RE.findall(t)]
        if not words:
            return
        if all(w in excluded_tokens for w in words):
            return
        if t not in order:
            order[t] = idx
            idx += 1
        counts[t] = counts.get(t, 0.0) + weight

    for ent in doc.ents:
        if ent.label_ in _DROP_ENT_LABELS:
            continue
        _add(ent.text, weight=10.0)
    for nc in doc.noun_chunks:
        _add(_strip_leading_nc(nc), weight=1.0)
    for tok in doc:
        if tok.pos_ == 'PROPN' and not tok.is_stop:
            _add(tok.text, weight=2.0)

    ranked = sorted(counts.keys(), key=lambda t: (-counts[t], order[t]))

    kept: list[str] = []
    kept_word_sets: list[frozenset[str]] = []
    for term in ranked:
        words = frozenset(_WORD_RE.findall(term.lower()))
        # Subsumed by any already-kept term (identical or proper subset).
        if any(words == ws or words < ws for ws in kept_word_sets):
            continue
        # Also drop earlier-kept strict subsets of the current term.
        to_remove = [i for i, ws in enumerate(kept_word_sets) if ws < words]
        for i in reversed(to_remove):
            kept.pop(i)
            kept_word_sets.pop(i)
        kept.append(term)
        kept_word_sets.append(words)
        if len(kept) >= k:
            break
    return kept


# ---------------------------------------------------------------------------
# budget-aware formatting (pure strings)
# ---------------------------------------------------------------------------
def _format_triple(triple: tuple[str, ...]) -> str:
    return '(' + _SLOT_SEP.join(triple) + ')'


def _compose(opening: str, rel: str, kw: str) -> str:
    parts: list[str] = []
    if opening:
        parts.append(f'Open: {opening}')
    if rel:
        parts.append(f'Rel: {rel}')
    if kw:
        parts.append(f'More: {kw}')
    return '\n'.join(parts)


def _fit_under_budget(
    opening: str,
    triples: list[tuple[str, ...]],
    keywords: list[str],
    budget: int,
    *,
    fallback_text: str = '',
) -> str:
    """Pack as many triples + keywords as possible under ``budget``.

    Strategy:
      1. If opening alone is already too long, word-boundary truncate it.
      2. Greedily append triples one-by-one, keeping a running string.
      3. Greedily append keywords one-by-one on top of whatever fits.
      4. Never exceed ``budget`` — final safety clamp applies.
    """
    # ----- opening -----
    if opening and len(f'Open: {opening}') > budget:
        max_open = max(0, budget - len('Open: '))
        opening = _word_boundary_truncate(opening, max_open) if max_open else ''

    if not opening and not triples and not keywords:
        # Nothing extractable — fall back to raw text, strict-truncated.
        base = fallback_text[:budget] if fallback_text else ''
        return _word_boundary_truncate(base, budget) if base else base

    current = _compose(opening, '', '')
    if len(current) > budget:
        return current[:budget]

    # ----- triples -----
    kept_triples: list[tuple[str, ...]] = []
    for t in triples:
        trial_rel = _TRIPLE_SEP.join(_format_triple(x) for x in kept_triples + [t])
        trial = _compose(opening, trial_rel, '')
        if len(trial) <= budget:
            kept_triples.append(t)
        else:
            break

    rel_str = _TRIPLE_SEP.join(_format_triple(x) for x in kept_triples)

    # ----- keywords -----
    kept_kws: list[str] = []
    for k in keywords:
        trial_kw = ', '.join(kept_kws + [k])
        trial = _compose(opening, rel_str, trial_kw)
        if len(trial) <= budget:
            kept_kws.append(k)
        else:
            break

    kw_str = ', '.join(kept_kws)
    result = _compose(opening, rel_str, kw_str)
    if not result:
        # Budget too tight for any extracted slot — fall back to raw
        # text truncated at a word boundary.
        base = fallback_text[:budget] if fallback_text else ''
        return _word_boundary_truncate(base, budget) if base else base
    # Belt-and-braces: budget is strict.
    return result if len(result) <= budget else result[:budget]


# ---------------------------------------------------------------------------
# KeywordCondenser
# ---------------------------------------------------------------------------
class KeywordCondenser(Condenser):
    """Extractive, spaCy-driven passage condenser.
    TODO: Experimental feature, wait for testing

    Args:
        num_relations: Max number of
            ``(subject, verb, object[, prep-obj])`` tuples per chunk.
            Set to ``0`` to disable the ``Rel:`` slot.
        max_first_sentence_chars: Hard cap for the opening slot, applied
            before the global compression budget.
        num_keywords: Max keyword items per chunk. ``0`` disables ``More:``.
        compression_ratio: Target compression factor. Must be ``> 1``.
            ``len(output) <= ceil(len(input) / compression_ratio)`` is
            strictly enforced for every chunk that passes ``min_chars``.
        spacy_model: spaCy pipeline name (default ``en_core_web_sm``).
        min_chars: Pre-filter. Chunks shorter than this are passed
            through **unchanged**; the ratio contract does not apply to
            them. Set to ``0`` to always compress.
        skip_roles: Roles whose chunks are never compressed.
        rounds: Optional set/list of conversation-turn numbers to
            compress. ``None`` (default) = no round-based filtering;
            when provided, chunks whose ``round`` is not in this set
            are passed through unchanged. Chunks that lack a ``round``
            field are also skipped when this filter is active.

    Every produced chunk is marked with ``raw.condensed=True`` so
    :meth:`Chunks.to_trajectory` wraps it in ``<block_N>...</block_N>``.

    Example:
        >>> from twinkle_agentic.chunker import NativeChunker
        >>> from twinkle_agentic.condenser.keyword import KeywordCondenser
        >>> chunker = NativeChunker(chunk_size=1024)
        >>> cond = KeywordCondenser(
        ...     num_relations=3, max_first_sentence_chars=160,
        ...     num_keywords=8, compression_ratio=4.0)
        >>> traj = {'messages': [{'role': 'user', 'content': long_passage}]}
        >>> chunks = cond(chunker(traj))
        >>> traj_compressed = chunks.to_trajectory()
    """

    def __init__(
            self,
            num_relations: int = 3,
            max_first_sentence_chars: int = 160,
            num_keywords: int = 8,
            compression_ratio: float = 4.0,
            spacy_model: str = 'en_core_web_sm',
            min_chars: int = 200,
            skip_roles: Sequence[str] = ('system', 'tool', 'assistant'),
            rounds: Sequence[int] | None = None,
    ):
        if num_relations < 0:
            raise ValueError(f'num_relations must be >= 0, got {num_relations}')
        if num_keywords < 0:
            raise ValueError(f'num_keywords must be >= 0, got {num_keywords}')
        if max_first_sentence_chars < 0:
            raise ValueError(f'max_first_sentence_chars must be >= 0, got {max_first_sentence_chars}')
        if compression_ratio <= 1.0:
            raise ValueError(f'compression_ratio must be > 1, got {compression_ratio}')
        if min_chars < 0:
            raise ValueError(f'min_chars must be >= 0, got {min_chars}')

        self.num_relations = num_relations
        self.max_first_sentence_chars = max_first_sentence_chars
        self.num_keywords = num_keywords
        self.compression_ratio = float(compression_ratio)
        self.spacy_model = spacy_model
        self.min_chars = min_chars
        self.skip_roles = tuple(skip_roles)
        self.rounds = set(rounds) if rounds is not None else None

    # ------------------------------------------------------------------
    def __call__(self, chunks: Chunks, **kwargs) -> Chunks:
        nlp = _load_spacy(self.spacy_model)
        out: list[Chunk] = []
        for c in chunks.chunks:
            if not self._should_condense(c):
                out.append(c)
                continue
            compressed = self._condense(c['content'], nlp)
            out.append(self._mark_condensed(c, compressed))
        return Chunks(chunks=out)

    # ------------------------------------------------------------------
    # selection policy
    # ------------------------------------------------------------------
    def _should_condense(self, chunk: Chunk) -> bool:
        if chunk.get('type') != 'text':
            return False
        if chunk.get('role') in self.skip_roles:
            return False
        if self.rounds is not None and chunk.get('round') not in self.rounds:
            return False
        content = chunk.get('content')
        if not isinstance(content, str) or not content:
            return False
        if len(content) < self.min_chars:
            return False
        raw = chunk.get('raw') or {}
        if isinstance(raw, dict):
            # Chunker-emitted reasoning / tool-call text chunks carry a
            # non-empty ``kind`` marker; leave them alone.
            if raw.get('kind'):
                return False
            # Idempotency — don't re-condense already condensed chunks.
            if raw.get('condensed'):
                return False
        return True

    @staticmethod
    def _mark_condensed(chunk: Chunk, content: str) -> Chunk:
        new: dict[str, Any] = dict(chunk)
        raw = dict(new.get('raw') or {})
        raw.setdefault('original', new.get('content', ''))
        new['content'] = content
        raw['condensed'] = True
        new['raw'] = raw
        return new  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # core extractive compression
    # ------------------------------------------------------------------
    def _condense(self, text: str, nlp) -> str:
        budget = max(1, math.ceil(len(text) / self.compression_ratio))
        doc = nlp(text)
        opening = _extract_opening(doc, self.max_first_sentence_chars)
        excluded = _word_tokens_lower(opening)
        triples = _extract_triples(doc, self.num_relations)
        keywords = _extract_keywords(doc, self.num_keywords, excluded)
        return _fit_under_budget(opening, triples, keywords, budget, fallback_text=text)
