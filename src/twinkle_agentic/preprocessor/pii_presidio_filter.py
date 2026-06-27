# Copyright (c) ModelScope Contributors. All rights reserved.
"""Multi-language, multi-country PII rewriter via Presidio + spaCy NER + Faker.

Coverage:
  Names/Locations/Orgs:  PERSON, LOCATION, ORGANIZATION (NER, en + zh)
  Network/contact:       EMAIL_ADDRESS, IP_ADDRESS, URL
  Finance:               CREDIT_CARD (Luhn), IBAN_CODE, CRYPTO, US_BANK_NUMBER, CN_BANK
  Government IDs:        US_SSN, US_ITIN, US_PASSPORT, US_DRIVER_LICENSE,
                         UK_NHS, UK_NINO, IN_AADHAAR, IN_PAN, AU_ABN, SG_NRIC,
                         IT_FISCAL_CODE, ES_NIF, ES_NIE, CN_ID
  Phones:                PHONE_NUMBER (libphonenumber), CN_PHONE, CN_LANDLINE
  Other:                 DATE_TIME, MEDICAL_LICENSE, NRP

Strategies (per entity, configurable via ``entity_strategy``):
  ``mask``    -> keep edges, mask middle (numeric IDs/cards)
  ``replace`` -> Faker fake value (names/emails — preserves text fluency)
  ``redact``  -> drop the span entirely
  ``hash``    -> sha256 prefix (deterministic, deidentified, joinable)

Consistency: same source value → same fake value within a batch (and optionally
across batches via ``persistent_consistency``), so dialogues stay coherent.
"""
import hashlib
import threading
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple  # noqa: F401

from twinkle.preprocessor import Preprocessor

# ─── Validators ─────────────────────────────────────────────────────────────────

_ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_ID_CHECKS = '10X98765432'


def _is_valid_cn_id(s: str) -> bool:
    if len(s) != 18 or not s[:17].isdigit():
        return False
    total = sum(int(s[i]) * _ID_WEIGHTS[i] for i in range(17))
    return _ID_CHECKS[total % 11] == s[17].upper()


def _is_valid_luhn(s: str) -> bool:
    digits = [int(c) for c in s if c.isdigit()]
    if len(digits) < 13:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        checksum += d
    return checksum % 10 == 0


# ─── Replacement primitives ─────────────────────────────────────────────────────


class Strategy(str, Enum):
    MASK = 'mask'
    REPLACE = 'replace'
    REDACT = 'redact'
    HASH = 'hash'

    @classmethod
    def coerce(cls, value: 'str | Strategy') -> 'Strategy':
        try:
            return cls(value) if not isinstance(value, cls) else value
        except ValueError as e:
            allowed = ', '.join(s.value for s in cls)
            raise ValueError(f'Unknown strategy {value!r}. Allowed: {allowed}') from e


def _mask_keep_edges(s: str, head: int = 3, tail: int = 4, ch: str = '*') -> str:
    if len(s) <= head + tail:
        return ch * len(s)
    return s[:head] + ch * (len(s) - head - tail) + s[-tail:]


def _hash_short(s: str, salt: str = '') -> str:
    return hashlib.sha256((salt + s).encode('utf-8')).hexdigest()[:12]


# ─── Faker dispatcher (per-instance, thread-safe) ───────────────────────────────


class FakerProvider:
    """Maps Presidio entity_type → Faker provider call, with lang-locale cache."""

    _PROVIDER: Dict[str, Any] = {
        'PERSON': lambda f: f.name(),
        'LOCATION': lambda f: f.city(),
        'ORGANIZATION': lambda f: f.company(),
        'EMAIL_ADDRESS': lambda f: f.email(),
        'PHONE_NUMBER': lambda f: f.phone_number(),
        'CN_PHONE': lambda f: f.phone_number(),
        'CN_LANDLINE': lambda f: f.phone_number(),
        'IP_ADDRESS': lambda f: f.ipv4(),
        'URL': lambda f: f.url(),
        'IBAN_CODE': lambda f: f.iban(),
        'CREDIT_CARD': lambda f: f.credit_card_number(),
        'US_BANK_NUMBER': lambda f: f.credit_card_number(),
        'CN_BANK': lambda f: f.credit_card_number(),
        'CRYPTO': lambda f: f.sha256()[:34],
        'DATE_TIME': lambda f: str(f.date()),
    }
    _LOCALE: Dict[str, str] = {'zh': 'zh_CN', 'en': 'en_US'}

    def __init__(self) -> None:
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def faker(self, lang: str):
        if lang not in self._cache:
            with self._lock:
                if lang not in self._cache:
                    from faker import Faker
                    self._cache[lang] = Faker(self._LOCALE.get(lang, 'en_US'))
        return self._cache[lang]

    def fake_for(self, entity: str, original: str, lang: str) -> str:
        f = self.faker(lang)
        provider = self._PROVIDER.get(entity.upper())
        if provider is not None:
            return provider(f)
        # Same-length opaque alnum for unknown entities; downstream length checks survive.
        return f.bothify('?' * 2 + '#' * max(2, len(original) - 2)).upper()


# ─── CN recognizers (module-level so they introspect/pickle cleanly) ────────────


def _cn_recognizer_classes():
    """Lazy-imported once; PatternRecognizer requires presidio_analyzer at import time."""
    from presidio_analyzer import Pattern, PatternRecognizer

    class CNIDRecognizer(PatternRecognizer):

        def validate_result(self, pattern_text: str) -> bool:
            return _is_valid_cn_id(pattern_text)

    class CNBankRecognizer(PatternRecognizer):

        def validate_result(self, pattern_text: str) -> bool:
            return _is_valid_luhn(pattern_text)

    return Pattern, PatternRecognizer, CNIDRecognizer, CNBankRecognizer


def _build_cn_recognizers(languages: Sequence[str]) -> List[Any]:
    Pattern, PatternRecognizer, CNIDRecognizer, CNBankRecognizer = _cn_recognizer_classes()
    specs = [
        ('CN_ID', r'(?<![\dA-Za-z])\d{17}[\dXx](?![\dA-Za-z])', 0.85, CNIDRecognizer),
        ('CN_PHONE', r'(?<!\d)1[3-9]\d{9}(?!\d)', 0.85, PatternRecognizer),
        ('CN_LANDLINE', r'(?<!\d)0\d{2,3}[-\s]?\d{7,8}(?!\d)', 0.70, PatternRecognizer),
        ('CN_BANK', r'(?<!\d)\d{13,19}(?!\d)', 0.40, CNBankRecognizer),
    ]
    out: List[Any] = []
    for entity, regex, score, cls in specs:
        pat = Pattern(name=entity.lower(), regex=regex, score=score)
        for lang in languages:
            out.append(cls(supported_entity=entity, patterns=[pat], supported_language=lang))
    return out


# ─── Filter ─────────────────────────────────────────────────────────────────────


class PIIPresidioFilter(Preprocessor):
    """Multi-language, multi-country PII rewriter (Presidio + spaCy + Faker)."""

    DEFAULT_ENTITY_STRATEGY: Dict[str, Strategy] = {
        'EMAIL_ADDRESS': Strategy.REPLACE,
        'PHONE_NUMBER': Strategy.MASK,
        'IP_ADDRESS': Strategy.MASK,
        'CREDIT_CARD': Strategy.MASK,
        'IBAN_CODE': Strategy.MASK,
        'CRYPTO': Strategy.MASK,
        'US_BANK_NUMBER': Strategy.MASK,
        'US_SSN': Strategy.MASK,
        'US_ITIN': Strategy.MASK,
        'US_PASSPORT': Strategy.MASK,
        'US_DRIVER_LICENSE': Strategy.MASK,
        'UK_NHS': Strategy.MASK,
        'UK_NINO': Strategy.MASK,
        'IN_AADHAAR': Strategy.MASK,
        'IN_PAN': Strategy.MASK,
        'AU_ABN': Strategy.MASK,
        'SG_NRIC': Strategy.MASK,
        'IT_FISCAL_CODE': Strategy.MASK,
        'ES_NIF': Strategy.MASK,
        'ES_NIE': Strategy.MASK,
        'MEDICAL_LICENSE': Strategy.MASK,
        'CN_ID': Strategy.MASK,
        'CN_PHONE': Strategy.MASK,
        'CN_LANDLINE': Strategy.MASK,
        'CN_BANK': Strategy.MASK,
    }
    DEFAULT_SPACY_MODELS: Dict[str, str] = {'en': 'en_core_web_sm', 'zh': 'zh_core_web_sm'}
    CJK_LANG_THRESHOLD: float = 0.15
    # Per-entity minimum span length to suppress short-token false positives.
    DEFAULT_MIN_LENGTH: Dict[str, int] = {
        'EMAIL_ADDRESS': 5,
    }
    MIN_LENGTH_FALLBACK: int = 3
    # NER-driven entities (spaCy hardcoded score 0.85) are too noisy on technical text; only regex-based
    # identifiers (phone/email/IDs/bank/cards) reliably indicate real PII. URL is also dropped—redacting
    # links in technical/instruction text changes semantics without privacy benefit.
    IGNORED_ENTITIES: Tuple[str, ...] = ('PERSON', 'LOCATION', 'ORGANIZATION', 'NRP', 'DATE_TIME', 'URL')
    INSTALL_HINT = ('PIIPresidioFilter requires: pip install presidio-analyzer presidio-anonymizer '
                    'faker spacy && python -m spacy download en_core_web_sm && '
                    'python -m spacy download zh_core_web_sm')

    def __init__(
        self,
        languages: Sequence[str] = ('en', 'zh'),
        spacy_models: Optional[Dict[str, str]] = None,
        entity_strategy: Optional[Dict[str, str]] = None,
        default_strategy: str = Strategy.MASK.value,
        score_threshold: float = 0.5,
        roles: Sequence[str] = ('user', 'assistant', 'system'),
        consistency: bool = True,
        persistent_consistency: bool = False,
        hash_salt: str = '',
        record_counts: bool = False,
    ) -> None:
        super().__init__()
        self._require_deps()

        self._languages: List[str] = list(languages)
        self._spacy_models = dict(self.DEFAULT_SPACY_MODELS)
        if spacy_models:
            self._spacy_models.update(spacy_models)
        for lang in self._languages:
            if lang not in self._spacy_models:
                raise ValueError(f'No spaCy model configured for language {lang!r}')

        self._strategy = {k: Strategy.coerce(v) for k, v in self.DEFAULT_ENTITY_STRATEGY.items()}
        if entity_strategy:
            self._strategy.update({k.upper(): Strategy.coerce(v) for k, v in entity_strategy.items()})
        self._default_strategy = Strategy.coerce(default_strategy)

        self._score_threshold = score_threshold
        self._roles = set(roles)
        self._consistency = consistency
        self._persistent_consistency = persistent_consistency
        self._hash_salt = hash_salt
        self._record_counts = record_counts

        self._faker = FakerProvider()
        self._persistent_map: Dict[Tuple[str, str], str] = {}
        self._analyzer = self._build_analyzer()
        # Restrict analyze() to entities we act on AND that the registry actually supports per language;
        # avoids 'Entity X doesn't have the corresponding recognizer in language : Y' warnings.
        wanted = {e for e in self._strategy if e not in self.IGNORED_ENTITIES}
        registry = self._analyzer.registry
        self._allowed_entities: Dict[str, List[str]] = {
            lang: sorted(wanted & set(registry.get_supported_entities(languages=[lang])))
            for lang in self._languages
        }

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def _require_deps(cls) -> None:
        try:
            import faker  # noqa: F401
            import presidio_analyzer  # noqa: F401
            import presidio_anonymizer  # noqa: F401
            import spacy  # noqa: F401
        except ImportError as e:
            raise ImportError(f'{e}. {cls.INSTALL_HINT}') from e

    def _build_analyzer(self):
        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        nlp_conf = {
            'nlp_engine_name': 'spacy',
            'models': [{
                'lang_code': lang,
                'model_name': self._spacy_models[lang]
            } for lang in self._languages],
        }
        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_conf).create_engine()
        # NER pipe is the heaviest spaCy component and we discard all NER entities; disable to save 2-4x latency.
        for nlp in getattr(nlp_engine, 'nlp', {}).values():
            for pipe in ('ner', 'parser', 'attribute_ruler', 'lemmatizer'):
                if pipe in nlp.pipe_names:
                    nlp.disable_pipe(pipe)
        registry = RecognizerRegistry(supported_languages=self._languages)
        registry.load_predefined_recognizers(languages=self._languages, nlp_engine=nlp_engine)
        for r in _build_cn_recognizers(self._languages):
            registry.add_recognizer(r)
        return AnalyzerEngine(registry=registry, nlp_engine=nlp_engine, supported_languages=self._languages)

    # ── language routing ────────────────────────────────────────────────────

    def _resolve_language(self, text: str) -> str:
        cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        guess = 'zh' if cjk / max(1, len(text)) > self.CJK_LANG_THRESHOLD else 'en'
        return guess if guess in self._languages else self._languages[0]

    # ── replacement ─────────────────────────────────────────────────────────

    def _replacement_for(
        self,
        entity: str,
        original: str,
        lang: str,
        local_map: Dict[Tuple[str, str], str],
    ) -> str:
        strategy = self._strategy.get(entity.upper(), self._default_strategy)
        if strategy is Strategy.REDACT:
            return ''
        if strategy is Strategy.HASH:
            return f'<{entity}:{_hash_short(original, self._hash_salt)}>'
        if strategy is Strategy.MASK:
            return _mask_keep_edges(original)
        # Strategy.REPLACE — Faker with optional consistency cache.
        if not self._consistency:
            return self._faker.fake_for(entity, original, lang)
        cache = self._persistent_map if self._persistent_consistency else local_map
        key = (entity.upper(), original)
        if key not in cache:
            cache[key] = self._faker.fake_for(entity, original, lang)
        return cache[key]

    @classmethod
    def _min_length(cls, entity: str) -> int:
        return cls.DEFAULT_MIN_LENGTH.get(entity.upper(), cls.MIN_LENGTH_FALLBACK)

    # ── span dedup ──────────────────────────────────────────────────────────

    @staticmethod
    def _dedupe_overlaps(results: List[Any]) -> List[Any]:
        """Greedy interval scheduling: keep highest-score span per overlapping region."""
        ordered = sorted(results, key=lambda r: (-r.score, -(r.end - r.start), r.start))
        kept: List[Any] = []
        for r in ordered:
            if any(r.start < k.end and r.end > k.start for k in kept):
                continue
            kept.append(r)
        return kept

    # ── core scrubbing ──────────────────────────────────────────────────────

    def _scrub_text(
        self,
        text: str,
        local_map: Dict[Tuple[str, str], str],
    ) -> Tuple[str, Dict[str, int]]:
        if not text:
            return text, {}
        lang = self._resolve_language(text)
        results = self._analyzer.analyze(
            text=text, language=lang, entities=self._allowed_entities.get(lang), score_threshold=self._score_threshold)
        if not results:
            return text, {}

        spans = self._dedupe_overlaps(results)
        spans = [r for r in spans if r.entity_type.upper() not in self.IGNORED_ENTITIES]
        spans = [r for r in spans if (r.end - r.start) >= self._min_length(r.entity_type)]
        if not spans:
            return text, {}
        # Reverse-sort so in-place index slicing stays valid.
        spans.sort(key=lambda r: r.start, reverse=True)
        out = text
        hits: Dict[str, int] = {}
        for r in spans:
            original = out[r.start:r.end]
            replacement = self._replacement_for(r.entity_type, original, lang, local_map)
            out = out[:r.start] + replacement + out[r.end:]
            hits[r.entity_type] = hits.get(r.entity_type, 0) + 1
        return out, hits

    def _scrub_row(
        self,
        row: Dict[str, Any],
        local_map: Dict[Tuple[str, str], str],
    ) -> Dict[str, int]:
        row_hits: Dict[str, int] = {}
        for m in row.get('messages') or []:
            if not isinstance(m, dict) or m.get('role') not in self._roles:
                continue
            content = m.get('content')
            if not isinstance(content, str) or not content:
                continue
            new_content, hits = self._scrub_text(content, local_map)
            if hits:
                m['content'] = new_content
                for k, v in hits.items():
                    row_hits[k] = row_hits.get(k, 0) + v
        return row_hits

    def __call__(self, rows) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        local_map: Dict[Tuple[str, str], str] = {}
        for row in rows:
            row_hits = self._scrub_row(row, local_map)
            if self._record_counts:
                if row_hits:
                    row['_pii_hits'] = row_hits
                else:
                    row.pop('_pii_hits', None)
        return rows, []
