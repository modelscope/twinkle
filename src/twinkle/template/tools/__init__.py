# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tool-call parser registry.

Importing this package auto-registers every parser. Order matters:
narrower / stronger formats first so round-robin detection prefers them
over weaker fallbacks.
"""
from .base import ToolCallParser, ToolCallRegistry
from .cline import ClineParser
from .qwen import HermesQwenParser
from .react import ReActParser
from .vcp import VCPParser

# Order: strongest/most-specific markers first. Hermes owns ``<tool_call>``
# (also denied by Cline), so its detection wins for shared-XML inputs.
ToolCallRegistry.register(HermesQwenParser())
ToolCallRegistry.register(ClineParser())
ToolCallRegistry.register(VCPParser())
ToolCallRegistry.register(ReActParser())

__all__ = [
    'ToolCallParser',
    'ToolCallRegistry',
    'HermesQwenParser',
    'ClineParser',
    'VCPParser',
    'ReActParser',
]
