from ray.serve.request_router import (FIFOMixin, MultiplexMixin, PendingRequest, ReplicaID, ReplicaResult,
                                      RequestRouter, RunningReplica)
from typing import Dict, List, Optional

from twinkle.server.utils.state import ServerStateProxy, get_server_state
from twinkle.utils.logger import get_logger

logger = get_logger()


class StickyLoraRequestRouter(FIFOMixin, MultiplexMixin, RequestRouter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.state: ServerStateProxy = get_server_state()

    async def choose_replicas(
        self,
        candidate_replicas: List[RunningReplica],
        pending_request: Optional[PendingRequest] = None,
    ) -> List[List[RunningReplica]]:
        """
        This method chooses the best replica for the request based on
        multiplexed and avaliable lora count. The algorithm
        works as follows:

        1. Populate top_ranked_replicas based on available replicas based on
          multiplex_id (only one replica is chosen)
        2. Populate and override top_ranked_replicas info based on avalible lora
          slots of the replica.
        """

        # Take the best set of replicas for the multiplexed model
        if (pending_request is not None and pending_request.metadata.multiplexed_model_id):
            ranked_replicas_multiplex: List[RunningReplica] = (self.rank_replicas_via_multiplex(
                replicas=candidate_replicas,
                multiplexed_model_id=pending_request.metadata.multiplexed_model_id,
            ))[0]

            # If found any replica, return it
            if ranked_replicas_multiplex:
                logger.debug('[Router] Found replica for multiplexed model !!!')
                return [ranked_replicas_multiplex]

        # Dictionary to hold the top-ranked replicas
        top_ranked_replicas: Dict[ReplicaID, RunningReplica] = {}

        # Filter out replicas that are not available (queue length exceed max ongoing request)
        ranked_replicas_locality = self.select_available_replicas(candidates=candidate_replicas)

        for replica in ranked_replicas_locality:
            top_ranked_replicas[replica.replica_id] = replica

        # Filter out replicas that exceed max lora count (query from server state)
        candidate_ids = [r.replica_id.unique_id for r in top_ranked_replicas.values()]
        available_ids = set(self.state.get_available_replica_ids(candidate_ids))
        if available_ids:
            top_ranked_replicas = {
                rid: r
                for rid, r in top_ranked_replicas.items() if r.replica_id.unique_id in available_ids
            }

        if not top_ranked_replicas:
            # No replica has remaining LoRA capacity – fall back to all candidates
            logger.debug('[Router] No replica has remaining LoRA capacity')
            return [candidate_replicas]

        logger.debug('[Router] StickyLoraRequestRouter choosing replica for request')

        # Take the replica with minimum throughput.
        min_throughput_replicas = min(
            [replica for replica in top_ranked_replicas.values()],
            key=lambda r: r.routing_stats.get('throughput', 0),
        )
        return [[min_throughput_replicas]]
