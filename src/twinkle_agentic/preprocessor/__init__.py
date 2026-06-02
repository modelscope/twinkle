# Copyright (c) ModelScope Contributors. All rights reserved.
import json
from functools import partial
from typing import Any, Callable, Dict, List, Optional

from twinkle.preprocessor import Preprocessor
from twinkle.template import Template
from twinkle.utils import get_logger
from twinkle.utils.parallel import PosixFileLock
from .consistency_filter import ConsistencyFilter
from .data_juicer import DataJuicerPreprocessor
from .dead_loop_filter import DeadLoopFilter
from .hard_filter import HardFilter
from .ifd_filter import IFDFilter
from .intent_classifier import IntentClassifier
from .llm_backend import LLMBackend, OpenAIBackend, SamplerBackend  # noqa: F401
from .majority_vote import MajorityVoteFilter
from .message_sanity import MessageSanityFilter
from .perplexity import PerplexityFilter
from .refuse_filter import RefuseFilter
from .response_refiner import ResponseRefiner
from .token_soup import TokenSoupFilter

logger = get_logger(only_local_master=False)


class QualityPreprocessor(Preprocessor):
    """End-to-end trajectory quality pipeline.

    Stages run in order; each stage operates only on rows that survived all
    previous stages.  Set a flag to False or leave optional resources as None /
    empty-string to skip that stage.

    Phase 1  Text normalisation    fix_unicode, remove_repeat_sentences
    Phase 1.5 Message sanity        role order, trim-to-assistant, sensitive words
    Phase 2  Structural rules      hard_filter, refuse_filter, dead_loop_filter
    Phase 3  Character quality     token_soup, word/char repeat, special chars, alnum
    Phase 4  Token length          token_num_filter (HF tokenizer)
    Phase 5  Vocabulary quality    stopwords, flagged_words
    Phase 6  Language ID           language_filter (FastText)
    Phase 7  KenLM PPL             kenlm_perplexity_filter (N-gram, CPU)
    Phase 8  MinHash dedup         minhash_dedup (off by default)
    Phase 9  Neural PPL            PerplexityFilter (vLLM sampler, off by default)
    Phase 9.5 2D Consistency       ConsistencyFilter (rollout + embed, off by default)
    Phase 10 LLM API filters       quality/difficulty/condition (off by default)
    Phase 11 Intent classification  annotate intent label (off by default)
    """

    def __init__(
        self,
        # ── Shared LLM backend (alternative to per-phase endpoints) ───────────
        backend: Optional[LLMBackend] = None,
        embed_backend: Optional[LLMBackend] = None,
        # ── Phase 1: text normalisation ───────────────────────────────────────
        fix_unicode: bool = True,
        remove_repeat_sentences: bool = True,
        # ── Phase 1.5: message sanity ──────────────────────────────────────────
        message_sanity_filter: bool = True,
        sensitive_words_file: str = '',  # '' = use built-in defaults; path to .json/.txt
        extra_sensitive_words: Optional[List[str]] = None,
        # ── Phase 2: structural rule filters ──────────────────────────────────
        hard_filter: bool = True,
        refuse_filter: bool = True,
        dead_loop_filter: bool = True,
        # Pass-through for passage-only rows (no user turn) so HardFilter does not
        # drop them outright.
        allow_incomplete_role: bool = False,
        # ── Phase 3: character-level quality ──────────────────────────────────
        token_soup_filter: bool = True,
        word_repeat_max_ratio: float = 0.4,
        char_repeat_max_ratio: float = 0.4,
        # special_chars_filter is structurally incompatible with markdown-formatted
        # responses (tables/bold/dividers push ratio above any usable threshold);
        # opt-in only.
        special_chars_filter: bool = False,
        special_chars_max_ratio: float = 0.5,
        alphanumeric_min_ratio: float = 0.25,
        # ── Phase 4: token length bounds ──────────────────────────────────────
        token_num_filter: bool = True,
        token_num_min: int = 10,
        token_num_max: int = 8192,
        hf_tokenizer: str = 'Qwen/Qwen3.5-4B',
        # ── Phase 5: vocabulary quality ───────────────────────────────────────
        content_lang: str = 'all',          # language code for vocab filters ('all' covers multilingual data)
        stopwords_min_ratio: float = 0.0,
        # 'all' merges low-resource lists where 2-letter math vars (BF/AF/...) collide as profanity
        flagged_words_lang: str = 'en',
        # raised from 0.045 to tolerate proper nouns like "Dick"/"Cock"/"Wang" in narratives
        flagged_words_max_ratio: float = 0.10,
        # ── Phase 6: language identification ──────────────────────────────────
        language: str = '',                  # '' = skip; 'en'/'zh'/... = enforce
        language_min_score: float = 0.7,
        # ── Phase 7: KenLM n-gram perplexity ──────────────────────────────────
        kenlm_lang: str = '',                # '' = skip
        kenlm_max_ppl: float = 1500.0,
        # ── Phase 8: near-duplicate removal ───────────────────────────────────
        minhash_dedup: bool = False,
        jaccard_threshold: float = 0.7,
        # ── Phase 9: neural PPL via OpenAI-compatible API (optional) ────────────────
        ppl_api_endpoint: str = '',      # '' = skip
        ppl_model: str = 'default',
        ppl_tokenizer: str = '',         # HF tokenizer for chat-template rendering
        ppl_min: float = 2.0,
        ppl_max: float = 100.0,
        ppl_max_workers: int = 8,
        # ── Phase 9.5: 2D consistency filter (optional) ───────────────────────
        consistency_sampler_endpoint: str = '',  # '' = skip
        consistency_embed_endpoint: str = '',
        consistency_sampler_model: str = 'default',
        consistency_embed_model: str = 'bge-m3',
        consistency_n_rollouts: int = 8,
        consistency_c_thresh: float = 0.7,
        consistency_d_thresh: float = 0.3,
        consistency_source: str = 'auto',    # 'teacher'|'self'|'auto'
        consistency_annotate: bool = False,
        consistency_max_workers: int = 4,
        # ── Phase 9.7: majority vote filter (optional) ────────────────────────
        majority_vote_sources: Optional[List[Dict[str, Any]]] = None,
        majority_vote_system_prompt: str = '',
        majority_vote_threshold: float = 0.5,
        majority_vote_temperature: float = 0.0,
        majority_vote_max_workers: int = 8,
        # ── Phase 10: LLM API filters (optional) ──────────────────────────────
        llm_api_endpoint: str = '',          # '' = skip all LLM filters
        llm_model: str = 'default',
        llm_quality_min_score: float = 0.5,
        llm_difficulty_min_score: float = 0.0,  # 0.0 = skip
        llm_condition: str = '',             # '' = skip
        llm_task_desc: str = '',             # '' = skip
        # ── Phase 11: intent classification (annotation, not filter; pure heuristic) ────────────
        # ── Phase 12: IFD hard-example filter (requires Phase 11) ───────────
        ifd_api_endpoint: str = '',          # '' = skip
        ifd_model: str = 'default',
        ifd_template: Optional[Template] = None,
        ifd_threshold: float = 0.8,
        # Diagnostic re-sampling: which intents to re-answer; [] disables (no extra inference cost).
        ifd_diagnostic_sample_intents: Optional[List[str]] = None,
        ifd_diagnostic_sample_n: int = 4,
        ifd_diagnostic_sample_temperature: float = 0.7,
        ifd_diagnostic_sample_max_tokens: int = 4096,
        # Paraphrase mode: 'both' dumps GT+paraphrase, True=paraphrase only, False=GT only.
        ifd_paraphrase_mode='both',
        ifd_paraphrase_intents: Optional[List[str]] = None,
        ifd_paraphrase_temperature: float = 0.7,
        ifd_paraphrase_max_tokens: int = 4096,
        ifd_paraphrase_prompt_budget: int = 4096,
        # ── Phase 13: response refinement (requires key_rounds) ─────────────
        refine_api_endpoint: str = '',       # '' = skip
        refine_model: str = 'default',
        refine_api_key: str = '',
        refine_temperature: float = 0.6,
        refine_max_tokens: int = 4096,
        refine_max_workers: int = 8,
        # ── Diagnostics ───────────────────────────────────────────────────────
        dropped_log_path: str = '',          # '' = skip; otherwise JSONL append
    ) -> None:
        super().__init__()

        dj = DataJuicerPreprocessor()
        pipeline: List[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = []

        # Phase 1: normalisation
        if fix_unicode:
            pipeline.append(dj.fix_unicode)
        if remove_repeat_sentences:
            pipeline.append(dj.remove_repeat_sentences)

        # Phase 1.5: message sanity
        if message_sanity_filter:
            pipeline.append(MessageSanityFilter(
                sensitive_words_file=sensitive_words_file or None,
                extra_sensitive_words=extra_sensitive_words,
            ).message_sanity_filter)

        # Phase 2: structural rules
        if hard_filter:
            pipeline.append(HardFilter(allow_incomplete_role=allow_incomplete_role).hard_filter)
        if refuse_filter:
            pipeline.append(RefuseFilter().refuse_filter)
        if dead_loop_filter:
            pipeline.append(DeadLoopFilter().dead_loop_filter)

        # Phase 3: character-level quality
        if token_soup_filter:
            pipeline.append(TokenSoupFilter().token_soup_filter)
        pipeline.append(partial(dj.word_repeat_filter, max_ratio=word_repeat_max_ratio))
        pipeline.append(partial(dj.char_repeat_filter, max_ratio=char_repeat_max_ratio))
        if special_chars_filter:
            pipeline.append(partial(dj.special_chars_filter, max_ratio=special_chars_max_ratio))
        pipeline.append(partial(dj.alphanumeric_filter, min_ratio=alphanumeric_min_ratio))

        # Phase 4: token length
        if token_num_filter:
            pipeline.append(partial(dj.token_num_filter,
                                    hf_tokenizer=hf_tokenizer,
                                    min_num=token_num_min,
                                    max_num=token_num_max))

        # Phase 5: vocabulary quality
        pipeline.append(partial(dj.stopwords_filter,
                                lang=content_lang,
                                min_ratio=stopwords_min_ratio))
        pipeline.append(partial(dj.flagged_words_filter,
                                lang=flagged_words_lang,
                                max_ratio=flagged_words_max_ratio))

        # Phase 6: language identification
        if language:
            pipeline.append(partial(dj.language_filter,
                                    lang=language,
                                    min_score=language_min_score))

        # Phase 7: KenLM perplexity
        if kenlm_lang:
            pipeline.append(partial(dj.kenlm_perplexity_filter,
                                    lang=kenlm_lang,
                                    max_ppl=kenlm_max_ppl))

        # Phase 8: near-duplicate removal
        if minhash_dedup:
            pipeline.append(partial(dj.minhash_dedup, jaccard_threshold=jaccard_threshold))

        # Phase 9: neural PPL
        if (backend or ppl_api_endpoint) and ppl_tokenizer:
            pf = PerplexityFilter(
                backend=backend,
                api_endpoint=ppl_api_endpoint,
                model=ppl_model,
                tokenizer_name_or_path=ppl_tokenizer,
                ppl_min=ppl_min,
                ppl_max=ppl_max,
                max_workers=ppl_max_workers,
            )
            pipeline.append(pf.ppl_filter)

        # Phase 9.5: 2D consistency filter
        if (backend or consistency_sampler_endpoint) and (embed_backend or consistency_embed_endpoint):
            cf = ConsistencyFilter(
                backend=backend,
                embed_backend=embed_backend,
                sampler_endpoint=consistency_sampler_endpoint,
                embed_endpoint=consistency_embed_endpoint,
                sampler_model=consistency_sampler_model,
                embed_model=consistency_embed_model,
                n_rollouts=consistency_n_rollouts,
                c_thresh=consistency_c_thresh,
                d_thresh=consistency_d_thresh,
                source=consistency_source,
                annotate=consistency_annotate,
                max_workers=consistency_max_workers,
            )
            pipeline.append(cf.consistency_filter)

        # Phase 9.7: majority vote
        if majority_vote_sources:
            mv_kwargs: Dict[str, Any] = {
                'sources': majority_vote_sources,
                'pass_threshold': majority_vote_threshold,
                'temperature': majority_vote_temperature,
                'max_workers': majority_vote_max_workers,
            }
            if majority_vote_system_prompt:
                mv_kwargs['system_prompt'] = majority_vote_system_prompt
            pipeline.append(MajorityVoteFilter(**mv_kwargs).majority_vote_filter)

        # Phase 10: LLM API filters
        if llm_api_endpoint:
            pipeline.append(partial(dj.llm_quality_filter,
                                    api_endpoint=llm_api_endpoint,
                                    model=llm_model,
                                    min_score=llm_quality_min_score))
            if llm_difficulty_min_score > 0.0:
                pipeline.append(partial(dj.llm_difficulty_filter,
                                        api_endpoint=llm_api_endpoint,
                                        model=llm_model,
                                        min_score=llm_difficulty_min_score))
            if llm_condition:
                pipeline.append(partial(dj.llm_condition_filter,
                                        condition=llm_condition,
                                        api_endpoint=llm_api_endpoint,
                                        model=llm_model))
            if llm_task_desc:
                pipeline.append(partial(dj.llm_task_relevance_filter,
                                        api_endpoint=llm_api_endpoint,
                                        task_desc=llm_task_desc,
                                        model=llm_model))

        # Phase 11: intent classification (pure heuristic, no LLM)
        ic = IntentClassifier()
        pipeline.append(ic.classify_intent)

        # Phase 12: IFD hard-example filter
        if (backend or ifd_api_endpoint) and ifd_template is not None:
            ifd = IFDFilter(
                backend=backend,
                api_endpoint=ifd_api_endpoint,
                model=ifd_model,
                template=ifd_template,
                ifd_threshold=ifd_threshold,
                diagnostic_sample_intents=ifd_diagnostic_sample_intents,
                diagnostic_sample_n=ifd_diagnostic_sample_n,
                diagnostic_sample_temperature=ifd_diagnostic_sample_temperature,
                diagnostic_sample_max_tokens=ifd_diagnostic_sample_max_tokens,
                paraphrase_mode=ifd_paraphrase_mode,
                paraphrase_intents=ifd_paraphrase_intents,
                paraphrase_temperature=ifd_paraphrase_temperature,
                paraphrase_max_tokens=ifd_paraphrase_max_tokens,
                paraphrase_prompt_budget=ifd_paraphrase_prompt_budget,
            )
            pipeline.append(ifd.ifd_filter)

        # Phase 13: response refinement
        if backend or refine_api_endpoint:
            refiner = ResponseRefiner(
                backend=backend,
                api_endpoint=refine_api_endpoint,
                model=refine_model,
                api_key=refine_api_key,
                temperature=refine_temperature,
                max_tokens=refine_max_tokens,
                max_workers=refine_max_workers,
            )
            pipeline.append(refiner.refine)

        self._pipelines = pipeline
        self._dropped_log_path = dropped_log_path
        self._lock: Optional[PosixFileLock] = (
            PosixFileLock(dropped_log_path + '.lock') if dropped_log_path else None)

    def _log_dropped(self, step_name: str, prev: List[Dict[str, Any]],
                     kept: List[Dict[str, Any]]) -> None:
        if not self._lock or len(kept) == len(prev):
            return
        kept_ids = {id(r) for r in kept}
        dropped = [r for r in prev if id(r) not in kept_ids]
        if not dropped:
            return
        with self._lock:
            with open(self._dropped_log_path, 'a', encoding='utf-8') as f:
                for r in dropped:
                    f.write(json.dumps({'step': step_name, 'row': r},
                                       ensure_ascii=False, default=str) + '\n')

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        for step in self._pipelines:
            if not rows:
                break
            step_name = getattr(step, '__name__', str(step))
            before = len(rows)
            prev = rows
            rows = step(rows)
            after = len(rows)
            logger.info(f'[QualityPreprocessor] {step_name}: {before} -> {after} (dropped {before - after})')
            self._log_dropped(step_name, prev, rows)
        return self.map_row_to_col(rows)

