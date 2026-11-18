"""A2 探针用的 Python 3.10 兼容层。

项目正式要求 Python >=3.11；A2 的 NPU 运行时是 3.10。这里只补探针导入所需的
``enum.StrEnum``，不改变生产代码，也不用于证明 3.10 是受支持的部署环境。
"""

from __future__ import annotations
import enum

if not hasattr(enum, "StrEnum"):

    class StrEnum(str, enum.Enum):
        """Python 3.11 ``StrEnum`` 的最小兼容实现。"""

        def __str__(self) -> str:
            return str.__str__(self)

    enum.StrEnum = StrEnum
