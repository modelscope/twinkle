from typing import Any, Dict, List, Optional

from twinkle.data_format import Trajectory
from twinkle.data_format.sampling import SamplingParams
from twinkle.template.base import Template

from twinkle_agentic.chunker.base import Chunker
from twinkle_agentic.condenser.base import Condenser
from twinkle_agentic.tools.extract_condensed import ExtractCondensed, TOOL_NAME as EXTRACT_TOOL_NAME
from twinkle_agentic.tools.tool_manager import ToolManager
from .multi_turn import MultiTurnRollout


class MultiTurnCondenseRollout(MultiTurnRollout):
    """Multi-turn rollout with trajectory compression + on-demand recovery.

    Pipeline per trajectory in the batch:
        1. ``chunker(trajectory)`` splits the incoming trajectory into chunks.
        2. ``condenser(chunks, **condenser_kwargs)`` rewrites selected text
           chunks with compressed stand-ins, marking them ``raw.condensed=True``
           and stashing the original under ``raw.original``.
        3. ``chunks.to_trajectory()`` rebuilds a trajectory where every
           condensed chunk is wrapped in ``<block_N>...</block_N>`` markers.
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
        condenser_kwargs: Optional[Dict[str, Any]] = None,
        trace_path: Optional[str] = None,
    ):
        super().__init__(
            sampler=sampler,
            template=template,
            tool_manager=tool_manager,
            sampling_params=sampling_params,
            max_turns=max_turns,
            trace_path=trace_path,
        )
        if chunker is None:
            raise ValueError(
                'MultiTurnCondenseRollout requires a Chunker instance')
        if condenser is None:
            raise ValueError(
                'MultiTurnCondenseRollout requires a Condenser instance')
        if EXTRACT_TOOL_NAME in tool_manager.names():
            # We reserve the name because we register a trajectory-bound
            # ExtractCondensed per trajectory; a pre-existing registration
            # would be silently overwritten on the clone, which is confusing.
            raise ValueError(
                f'tool_manager already registers {EXTRACT_TOOL_NAME!r}; '
                f'MultiTurnCondenseRollout registers a trajectory-bound '
                f'ExtractCondensed per call and would shadow the existing '
                f'one. Remove it from the shared manager or rename it.')
        self.chunker = chunker
        self.condenser = condenser
        self.condenser_kwargs = dict(condenser_kwargs or {})

    def __call__(self, trajectories: List[Trajectory], **kwargs) -> List[Trajectory]:
        if isinstance(trajectories, dict):
            raise TypeError(
                'MultiTurnCondenseRollout.__call__ expects a '
                'List[Trajectory]; wrap a single trajectory as [trajectory].')
        trajectories = list(trajectories)
        if not trajectories:
            return []

        compressed_list: List[Trajectory] = []
        tool_managers: List[ToolManager] = []
        for traj in trajectories:
            # 1-2. Chunk + condense this trajectory.
            chunks = self.chunker(traj)
            chunks = self.condenser(chunks, **self.condenser_kwargs)
            compressed = chunks.to_trajectory()
            for k, v in traj.items():
                compressed.setdefault(k, v)
            compressed_list.append(compressed)

            # 4. Per-trajectory tool manager: clone + inject ExtractCondensed
            #    bound to THIS trajectory's chunks. Never mutate
            #    self.tool_manager.
            call_tm = self.tool_manager.copy()
            call_tm.register(ExtractCondensed(chunks))
            tool_managers.append(call_tm)

        # 5. Delegate to the parent batch loop. A caller-supplied
        #    ``tool_manager`` would be surprising here (we already built
        #    the list) -- drop it to avoid ambiguity.
        kwargs.pop('tool_manager', None)
        return super().__call__(
            compressed_list, tool_manager=tool_managers, **kwargs)
