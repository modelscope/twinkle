import json
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Type, Union
from dataclasses import dataclass, field, fields
from PIL import Image
from copy import deepcopy

from twinkle import remote_class, requires
from twinkle.template import Template

Tool = Dict[str, Union[str, Dict]]
History = List[Union[Tuple[str, str], List[str]]]
Message = Dict[str, Union[str, List[Dict[str, Any]], List[int], None]]
Messages = List[Message]
Prompt = List[Union[str, List[int], List[str]]]
Word = Union[str, List[int]]
Context = Word

@remote_class()
class Gemma4Template(Template):
    """Processor for Google Gemma4 series."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # use original Template