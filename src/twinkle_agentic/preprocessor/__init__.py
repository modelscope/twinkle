# Copyright (c) ModelScope Contributors. All rights reserved.
from functools import partial
from typing import Any, Callable, Dict, List, Optional

from twinkle.preprocessor import Preprocessor

from .consistency_filter import ConsistencyFilter
from .data_juicer import DataJuicerPreprocessor
from .dead_loop_filter import DeadLoopFilter
from .hard_filter import HardFilter
from .majority_vote import MajorityVoteFilter
from .perplexity import PerplexityFilter
from .refuse_filter import RefuseFilter
from .token_soup import TokenSoupFilter


class QualityPreprocessor(Preprocessor):
    """End-to-end trajectory quality pipeline.

    Stages run in order; each stage operates only on rows that survived all
    previous stages.  Set a flag to False or leave optional resources as None /
    empty-string to skip that stage.

    Phase 1  Text normalisation    fix_unicode, remove_repeat_sentences
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
    """

    def __init__(
        self,
        # ── Phase 1: text normalisation ───────────────────────────────────────
        fix_unicode: bool = True,
        remove_repeat_sentences: bool = True,
        # ── Phase 2: structural rule filters ──────────────────────────────────
        hard_filter: bool = True,
        refuse_filter: bool = True,
        dead_loop_filter: bool = True,
        # ── Phase 3: character-level quality ──────────────────────────────────
        token_soup_filter: bool = True,
        word_repeat_max_ratio: float = 0.4,
        char_repeat_max_ratio: float = 0.4,
        special_chars_max_ratio: float = 0.25,
        alphanumeric_min_ratio: float = 0.25,
        # ── Phase 4: token length bounds ──────────────────────────────────────
        token_num_filter: bool = True,
        token_num_min: int = 10,
        token_num_max: int = 8192,
        hf_tokenizer: str = 'Qwen/Qwen2.5-0.5B',
        # ── Phase 5: vocabulary quality ───────────────────────────────────────
        content_lang: str = 'en',           # language code for vocab filters
        stopwords_min_ratio: float = 0.1,
        flagged_words_max_ratio: float = 0.045,
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
    ) -> None:
        super().__init__()

        dj = DataJuicerPreprocessor()
        pipeline: List[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = []

        # Phase 1: normalisation
        if fix_unicode:
            pipeline.append(dj.fix_unicode)
        if remove_repeat_sentences:
            pipeline.append(dj.remove_repeat_sentences)

        # Phase 2: structural rules
        if hard_filter:
            pipeline.append(HardFilter().hard_filter)
        if refuse_filter:
            pipeline.append(RefuseFilter().refuse_filter)
        if dead_loop_filter:
            pipeline.append(DeadLoopFilter().dead_loop_filter)

        # Phase 3: character-level quality
        if token_soup_filter:
            pipeline.append(TokenSoupFilter().token_soup_filter)
        pipeline.append(partial(dj.word_repeat_filter, max_ratio=word_repeat_max_ratio))
        pipeline.append(partial(dj.char_repeat_filter, max_ratio=char_repeat_max_ratio))
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
                                lang=content_lang,
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
        if ppl_api_endpoint:
            pf = PerplexityFilter(
                api_endpoint=ppl_api_endpoint,
                model=ppl_model,
                tokenizer_name_or_path=ppl_tokenizer,
                ppl_min=ppl_min,
                ppl_max=ppl_max,
                max_workers=ppl_max_workers,
            )
            pipeline.append(pf.ppl_filter)

        # Phase 9.5: 2D consistency filter
        if consistency_sampler_endpoint and consistency_embed_endpoint:
            cf = ConsistencyFilter(
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

        self._pipelines = pipeline

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        for step in self._pipelines:
            if not rows:
                break
            rows = step(rows)
        return self.map_row_to_col(rows)

