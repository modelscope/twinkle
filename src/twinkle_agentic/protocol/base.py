from abc import ABC, abstractmethod
from typing import List, Union

from twinkle.data_format import Trajectory
from twinkle.data_format.message import Message
from twinkle.data_format.sampling import SamplingParams


class API(ABC):
    """Abstract LLM API client: Trajectory + SamplingParams -> assistant Message(s)."""

    @abstractmethod
    def __call__(
        self,
        trajectory: Trajectory,
        sampling_params: SamplingParams,
        **kwargs,
    ) -> Union[Message, List[Message]]:
        raise NotImplementedError()
