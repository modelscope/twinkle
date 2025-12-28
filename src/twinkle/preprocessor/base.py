from typing import List

from twinkle.data_format import Trajectory


class Preprocessor:

    def __call__(self, row) -> Trajectory:
        ...


class DataFilter:

    def __call__(self, row) -> bool:
        ...