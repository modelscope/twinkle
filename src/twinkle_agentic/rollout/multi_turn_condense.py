# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, Callable, Dict, List, Optional

from twinkle.data_format import Trajectory
from twinkle.data_format.sampling import SamplingParams
from twinkle.infra import remote_class, remote_function
from twinkle.template.base import Template
from twinkle_agentic.chunker.base import Chunker
from twinkle_agentic.condenser.base import Condenser
from twinkle_agentic.data_format import Chunks
from twinkle_agentic.tools.extract_condensed import TOOL_NAME as EXTRACT_TOOL_NAME
from twinkle_agentic.tools.extract_condensed import ExtractCondensed
from twinkle_agentic.tools.tool_manager import ToolManager
from .multi_turn import MultiTurnRollout


@remote_class()
class MultiTurnCondenseRollout(MultiTurnRollout):
    """Multi-turn rollout with trajectory compression + on-demand recovery.
    TODO: Experimental feature, wait for testing

    Pipeline for a batch of trajectories:
        1. ``chunker(trajectory)`` splits each incoming trajectory into chunks.
        2. All per-trajectory :class:`Chunks` are concatenated into a single
           :class:`Chunks` and passed through ``condenser`` in ONE call, so
           the underlying sampler (e.g. vLLM) sees a maximally-packed batch
           spanning the whole rollout batch instead of a per-trajectory
           sequence. Remembered trajectory boundaries are used to slice the
           condensed chunks back into per-trajectory :class:`Chunks`.
        3. ``chunks.to_trajectory()`` rebuilds each trajectory, wrapping every
           condensed chunk in ``<block_N>...</block_N>`` markers.
        4. A trajectory-scoped :class:`ExtractCondensed` tool is registered on
           a per-trajectory clone of :attr:`tool_manager`, so the model can
           recover the original text of any block by its number.
        5. The batch of compressed trajectories + a parallel list of
           per-trajectory tool managers are handed to
           :meth:`MultiTurnRollout.__call__`, which drives the sample/tool
           loop (one batched ``sampler.sample`` per turn).

    The per-call tool manager is cloned via :meth:`ToolManager.copy`; the
    shared ``self.tool_manager`` is never mutated, so concurrent rollouts on
    the same instance are safe.

    Constructor accepts any :class:`Chunker` / :class:`Condenser` pair, so
    plug-in chunkers (e.g. ``NativeChunker``) and condensers (e.g.
    ``KeywordCondenser``, ``ModelCondenser``) compose freely.
    """

    def __init__(
        self,
        sampler,
        template: Template,
        tool_manager: ToolManager,
        chunker: Chunker,
        condenser: Condenser,
        sampling_params: Optional[SamplingParams] = None,
        max_turns: int = 6,
        max_trajectory_tokens: Optional[int] = None,
        condenser_kwargs: Optional[Dict[str, Any]] = None,
        trace_dir: Optional[str] = None,
        trace_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
        success_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
        post_compress_callback: Optional[Callable] = None,
    ):
        super().__init__(
            sampler=sampler,
            template=template,
            tool_manager=tool_manager,
            sampling_params=sampling_params,
            max_turns=max_turns,
            max_trajectory_tokens=max_trajectory_tokens,
            trace_dir=trace_dir,
            trace_callback=trace_callback,
            success_callback=success_callback,
        )
        if chunker is None:
            raise ValueError('MultiTurnCondenseRollout requires a Chunker instance')
        if condenser is None:
            raise ValueError('MultiTurnCondenseRollout requires a Condenser instance')
        if EXTRACT_TOOL_NAME in tool_manager.names():
            raise ValueError(f'tool_manager already registers {EXTRACT_TOOL_NAME!r}; '
                             f'MultiTurnCondenseRollout registers a trajectory-bound '
                             f'ExtractCondensed per call and would shadow the existing '
                             f'one. Remove it from the shared manager or rename it.')
        self.chunker = chunker
        self.condenser = condenser
        if getattr(self.condenser, 'template', None) is None:
            self.condenser.template = template
        self.condenser_kwargs = dict(condenser_kwargs or {})
        self.post_compress_callback = post_compress_callback
        self._trace_block_chunks: Optional[List[Optional[Chunks]]] = None

    @remote_function()
    def __call__(self, trajectories: List[Trajectory], **kwargs) -> List[Trajectory]:
        if isinstance(trajectories, dict):
            raise TypeError('MultiTurnCondenseRollout.__call__ expects a '
                            'List[Trajectory]; wrap a single trajectory as [trajectory].')
        trajectories = list(trajectories)
        if not trajectories:
            return []

        per_traj_chunks: List[Chunks] = [self.chunker(t) for t in trajectories]
        signatures = [self._chunk_signature(ck) for ck in per_traj_chunks]
        group_first: Dict[int, int] = {}
        for i, sig in enumerate(signatures):
            group_first.setdefault(sig, i)
        unique_indices: List[int] = list(group_first.values())

        merged_list = []
        boundaries: List[int] = []
        for idx in unique_indices:
            merged_list.extend(per_traj_chunks[idx].chunks)
            boundaries.append(len(merged_list))
        merged = Chunks(chunks=merged_list)
        merged = self.condenser(merged, **self.condenser_kwargs)

        # Split the merged result back into per-unique-trajectory Chunks.
        canonical: Dict[int, Chunks] = {}
        start = 0
        for uidx, end in zip(unique_indices, boundaries):
            canonical[uidx] = Chunks(chunks=merged.chunks[start:end])
            start = end

        compressed_list: List[Trajectory] = []
        tool_managers: List[ToolManager] = []
        for i, traj in enumerate(trajectories):
            traj_chunks = canonical[group_first[signatures[i]]]
            compressed = traj_chunks.to_trajectory()
            for k, v in traj.items():
                compressed.setdefault(k, v)
            if self.post_compress_callback is not None:
                compressed = self.post_compress_callback(compressed, traj_chunks, **kwargs)
            compressed_list.append(compressed)

            call_tm = self.tool_manager.copy()
            call_tm.register(ExtractCondensed(traj_chunks))
            tool_managers.append(call_tm)

        # 5. Delegate to the parent batch loop. A caller-supplied
        #    ``tool_manager`` would be surprising here (we already built
        #    the list) -- drop it to avoid ambiguity.
        kwargs.pop('tool_manager', None)
        if self.trace_dir:
            self._trace_block_chunks = [canonical[group_first[signatures[i]]] for i in range(len(trajectories))]
        else:
            self._trace_block_chunks = None
        try:
            return super().__call__(compressed_list, tool_manager=tool_managers, **kwargs)
        finally:
            self._trace_block_chunks = None

    @staticmethod
    def _chunk_signature(chunks: Chunks) -> int:
        """Cheap content-based signature of a :class:`Chunks` for dedup.

        Walks the chunk list once, dispatches on content type:

        * ``str`` / ``bytes``: hash with Python's built-in ``hash`` --
          SipHash, ~1 GB/s in C, and CPython caches the result on the
          string object so GRPO duplicates that share the same string
          are re-hashed for free.
        * Multimodal (PIL image, numpy array, tensor, dict, ...): if
          the object exposes ``tobytes``, hash its byte payload (stable
          across identity-distinct but pixel-identical images); else
          fall back to ``id(content)`` so duplicates referencing the
          SAME object still dedup, while distinct-but-equal payloads
          safely under-dedup (never over-dedup).

        Avoids ``json.dumps`` / ``repr``: both are 10-100x slower on
        long text, and either crash on non-serializable multimodal
        payloads or produce unstable output (e.g. PIL ``repr`` embeds
        a memory address).
        """
        parts: List[Any] = []
        for c in chunks.chunks:
            content = c.get('content')
            if isinstance(content, (str, bytes)):
                chash = hash(content)
            elif content is None:
                chash = 0
            else:
                tobytes = getattr(content, 'tobytes', None)
                if callable(tobytes):
                    try:
                        chash = hash(tobytes())
                    except Exception:
                        chash = id(content)
                else:
                    chash = id(content)
            parts.append((
                c.get('type'),
                c.get('role'),
                c.get('round'),
                chash,
            ))
        return hash(tuple(parts))

    def _build_trace_record(
        self,
        traj: Dict[str, Any],
        *,
        idx: int,
        success: bool,
    ) -> Dict[str, Any]:
        """Attach per-block and per-passthrough-passage maps to the record.

        Two complementary maps are dumped so the trace alone is enough
        to audit compression quality and compression coverage:

        * ``blocks`` — numbered ``block_N`` entries mirror
          :meth:`Chunks.to_trajectory` and :class:`ExtractCondensed`:
          text chunks with ``raw.condensed=True``, non-empty content
          and ``role != 'tool'``, numbered from 1. Each entry carries
          the pre-compression text (``original``, from
          ``raw.original``) and the post-compression text
          (``compressed``, the chunk content the model saw inside
          ``<block_N>...</block_N>``).
        * ``passages`` — numbered ``passage_M`` entries for text chunks
          from the first user message (role neither ``'system'`` nor
          ``'tool'``) that were NOT compressed — either because they
          failed the eligibility filter (too short, wrong role,
          ``skip_pattern`` matched, ...) or because the condenser's
          output was not strictly shorter than the original and fell
          back to passthrough. This lets the trace show the compressed
          vs. passthrough ratio per rollout.
        """
        record = super()._build_trace_record(traj, idx=idx, success=success)

        all_chunks = self._trace_block_chunks
        if all_chunks is None or idx >= len(all_chunks):
            return record
        chunks = all_chunks[idx]
        if chunks is None:
            return record
        blocks, passages = self._enumerate_blocks(chunks)
        record['blocks'] = blocks
        record['passages'] = passages
        return record

    @staticmethod
    def _enumerate_blocks(chunks: Chunks, ) -> 'tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]':
        """Walk ``chunks`` and emit ``(blocks, passages)`` maps.

        * ``blocks`` → ``{block_N: {original, compressed}}`` for every
          text chunk flagged ``raw.condensed=True`` (``role != 'tool'``).
          ``original`` is ``None`` when the condenser did not attach a
          ``raw.original`` snapshot; ``compressed`` is always present
          since it is simply the chunk's post-compression content.
        * ``passages`` → ``{passage_M: {content}}`` for every text chunk
          from the first user message (``role not in {'system', 'tool'}``)
          that was NOT flagged ``raw.condensed`` — i.e. chunks that
          were either filtered out before compression or fell back to
          passthrough because the model output was not strictly shorter.
          Lets a reader of the trace see the compressed / passthrough
          split without having to diff the raw trajectory.
        """
        blocks: Dict[str, Dict[str, Any]] = {}
        passages: Dict[str, Dict[str, Any]] = {}
        block_counter = 0
        passage_counter = 0
        for c in chunks.chunks:
            if c.get('type') != 'text':
                continue
            content = c.get('content')
            if not isinstance(content, str) or not content:
                continue
            role = c.get('role')
            if role == 'tool':
                continue
            raw = c.get('raw')
            is_condensed = (isinstance(raw, dict) and bool(raw.get('condensed')))
            if is_condensed:
                block_counter += 1
                original = raw.get('original') if isinstance(raw, dict) else None
                blocks[f'block_{block_counter}'] = {
                    'original': (original if isinstance(original, str) and original else None),
                    'compressed': content,
                }
            elif role == 'user':
                passage_counter += 1
                passages[f'passage_{passage_counter}'] = {
                    'content': content,
                }
        return blocks, passages
