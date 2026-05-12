from typing import Any, Callable, Dict, List, Optional

from twinkle.data_format import Trajectory
from twinkle.data_format.sampling import SamplingParams
from twinkle.template.base import Template

from twinkle_agentic.chunker.base import Chunker
from twinkle_agentic.condenser.base import Condenser
from twinkle_agentic.data_format import Chunks
from twinkle.infra import remote_class, remote_function
from twinkle_agentic.tools.extract_condensed import ExtractCondensed, TOOL_NAME as EXTRACT_TOOL_NAME
from twinkle_agentic.tools.tool_manager import ToolManager
from .multi_turn import MultiTurnRollout


@remote_class()
class MultiTurnCondenseRollout(MultiTurnRollout):
    """Multi-turn rollout with trajectory compression + on-demand recovery.

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
            raise ValueError(
                'MultiTurnCondenseRollout requires a Chunker instance')
        if condenser is None:
            raise ValueError(
                'MultiTurnCondenseRollout requires a Condenser instance')
        if EXTRACT_TOOL_NAME in tool_manager.names():
            raise ValueError(
                f'tool_manager already registers {EXTRACT_TOOL_NAME!r}; '
                f'MultiTurnCondenseRollout registers a trajectory-bound '
                f'ExtractCondensed per call and would shadow the existing '
                f'one. Remove it from the shared manager or rename it.')
        self.chunker = chunker
        self.condenser = condenser
        if getattr(self.condenser, 'template', None) is None:
            self.condenser.template = template
        self.condenser_kwargs = dict(condenser_kwargs or {})

    @remote_function()
    def __call__(self, trajectories: List[Trajectory], **kwargs) -> List[Trajectory]:
        if isinstance(trajectories, dict):
            raise TypeError(
                'MultiTurnCondenseRollout.__call__ expects a '
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
            compressed_list.append(compressed)

            call_tm = self.tool_manager.copy()
            call_tm.register(ExtractCondensed(traj_chunks))
            tool_managers.append(call_tm)

        # 5. Delegate to the parent batch loop. A caller-supplied
        #    ``tool_manager`` would be surprising here (we already built
        #    the list) -- drop it to avoid ambiguity.
        kwargs.pop('tool_manager', None)
        return super().__call__(
            compressed_list, tool_manager=tool_managers, **kwargs)

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
