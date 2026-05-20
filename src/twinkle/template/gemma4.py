
from typing import List
from twinkle import remote_class
from twinkle.template import Template
from twinkle.data_format import Trajectory

@remote_class()
class Gemma4Template(Template):
    """Processor for Google Gemma4 series."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # use original Template

    
