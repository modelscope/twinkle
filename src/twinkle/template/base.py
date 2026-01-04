from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer, PreTrainedTokenizer
import numpy as np
from twinkle.hub import HubOperation
from twinkle.data_format import Trajectory, InputFeature
from .utils import get_assistant_labels


class Template:

    PLACEHOLDER = "<<<ASSISTANT_PLACEHOLDER_7f3d2a1b>>>"

    def __init__(self, model_id: str, **kwargs):
        model_id = HubOperation.download_model(model_id)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id)

    def encode(self, trajectory: Trajectory) -> InputFeature:
        conversation = [dict(message) for message in trajectory['messages']]
        tools = [dict(tool) for tool in trajectory['tools']]
        encoded = self.tokenizer.apply_chat_template(conversation=conversation, tools=tools)
        labels = get_assistant_labels(self.tokenizer, conversation)
        labels = np.roll(labels, -1, axis=-1)
        return InputFeature(
            input_ids=np.array(encoded),
            attention_mask=np.ones_like(encoded),
            position_ids=np.arange(0, len(encoded)),
            labels=labels,
        )

    @staticmethod
    def find_subsequence(seq: List[int], subseq: List[int], start: int = 0) -> int:
        subseq_len = len(subseq)
        for i in range(start, len(seq) - subseq_len + 1):
            if seq[i:i + subseq_len] == subseq:
                return i
        return -1

    @staticmethod
    def map_col_to_row(trajectories: Dict[str, Any]):
        if not trajectories:
            return []
        rows = []
        total_count = len(trajectories[next(iter(list(trajectories.keys())))])
        for i in range(total_count):
            row = {}
            for key in trajectories:
                row[key] = trajectories[key][i]
            rows.append(row)
        return rows

    @staticmethod
    def map_row_to_col(rows: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        if not rows:
            return {}

        columns: Dict[str, List[Any]] = {}
        keys = rows[0].keys()

        for key in keys:
            columns[key] = [row[key] for row in rows]

        return columns

    def batch_encode(self, trajectories: Dict[str, Any]) -> List[InputFeature]:
        output = []
        _transfer = False
        if isinstance(trajectories, dict):
            _transfer = True
            trajectories = self.map_row_to_col(trajectories)
        for trajectory in trajectories:
            output.append(self.encode(trajectory))
        if _transfer:
            output = self.map_row_to_col(output)
        return output

    def check(self, trajectory: Trajectory) -> Optional[Trajectory]:
        encoded = None
        try:
            encoded = self.encode(trajectory)
            if not encoded:
                return None
            else:
                return trajectory
        except Exception as e:  # noqa
            return None
        finally:
            if encoded:
                del encoded

    def batch_check(self, trajectories: List[Trajectory]) -> List[Optional[Trajectory]]:
        output = []
        for trajectory in trajectories:
            output.append(self.check(trajectory))
        return output

    def decode(self, token_ids: List[int], **kwargs) -> str:
        return self.tokenizer.decode(token_ids, **kwargs)

    def batch_decode(self, token_ids: List[List[int]], **kwargs) -> List[str]:
        return [self.tokenizer.decode(_ids, **kwargs) for _ids in token_ids]