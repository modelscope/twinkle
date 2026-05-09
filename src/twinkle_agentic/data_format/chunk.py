import sys
from dataclasses import dataclass
from itertools import groupby
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

if sys.version_info[:2] <= (3, 11):
    # Pydantic requirements.
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

