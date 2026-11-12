# Copyright (c) ModelScope Contributors. All rights reserved.
"""Carry a recorded MoE routing into the trajectories the training forward replays."""
from typing import Any, Dict, List, Sequence, Tuple

from twinkle import remote_class, remote_function


@remote_class(execute='first')
class RoutingReplay:
    """Stitch a RECORD forward's MoE routing (``router_replay_mode='R2'``) into new trajectories.

    ``recorded`` is the record pass's whole-batch return; its ``routed_experts`` rows follow the
    batch order. Each trajectory is copied with ``routed_experts`` cut to its own input length and
    ``replay_mask`` mirroring its ``completion_mask``. The copies ride the ``stash`` channel as
    per-row references, so a caller swaps them in positionally without materialising them.
    """

    @remote_function(dispatch='all', collect='first', lazy_collect=True, stash=(0,))
    def __call__(self, trajs: Sequence[Dict[str, Any]], recorded: Any, **kwargs) -> Tuple[List[Dict[str, Any]]]:
        routed_rows = recorded.get('routed_experts') if isinstance(recorded, dict) else None
        if routed_rows is None:
            raise RuntimeError(
                'router_replay_mode="R2" asked the training model to RECORD its MoE routing, but the '
                'forward returned none. The policy must be an MoE model built with routing replay '
                'enabled; a dense policy has no expert routing to record (use "disabled").')
        tensors = routed_rows if isinstance(routed_rows, (list, tuple)) else [routed_rows]
        rows = [tensor[index] for tensor in tensors for index in range(tensor.shape[0])]
        if len(rows) != len(trajs):
            raise RuntimeError(f'R2 routing RECORD returned {len(rows)} rows for {len(trajs)} trajectories; '
                               'per-row routing alignment is impossible.')
        payloads = []
        for traj, routed in zip(trajs, rows):
            encoded = dict(traj)
            length = len(encoded['input_ids'])
            if routed.shape[0] < length:
                raise RuntimeError(f'R2 recorded routing covers {routed.shape[0]} tokens for a {length}-token '
                                   'sequence; the RECORD forward must route the whole sequence.')
            encoded['routed_experts'] = routed[:length]
            encoded['replay_mask'] = encoded['completion_mask']
            payloads.append(encoded)
        return (payloads, )
