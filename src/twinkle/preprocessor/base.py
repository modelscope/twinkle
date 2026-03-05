# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, Dict, List

from twinkle.data_format import Trajectory


class Preprocessor:

    @staticmethod
    def map_col_to_row(rows: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        if not rows:
            return []
        _new_rows = []
        total_count = len(rows[next(iter(list(rows.keys())))])
        for i in range(total_count):
            row = {}
            for key in rows:
                row[key] = rows[key][i]
            _new_rows.append(row)
        return _new_rows

    @staticmethod
    def map_row_to_col(rows: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        if not rows:
            return {}

        columns: Dict[str, List[Any]] = {}
        keys = rows[0].keys()

        for key in keys:
            columns[key] = [row[key] for row in rows]

        return columns

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        ...


class DataFilter:

    def __call__(self, row) -> bool:
        ...
