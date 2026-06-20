# Copyright (c) Twinkle Contributors. All rights reserved.
"""Skills loading framework - extensible providers for agent skill injection."""

from twinkle_client.tui.skills.base import SkillProvider
from twinkle_client.tui.skills.local_provider import LocalSkillProvider
from twinkle_client.tui.skills.manager import SkillManager
from twinkle_client.tui.skills.modelscope_provider import ModelScopeSkillProvider

__all__ = ['SkillProvider', 'SkillManager', 'LocalSkillProvider', 'ModelScopeSkillProvider']
