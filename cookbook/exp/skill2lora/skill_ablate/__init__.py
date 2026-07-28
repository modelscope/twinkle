# Copyright (c) ModelScope Contributors. All rights reserved.
"""Skill-generation ablation package (view A / view B × think × style × training method).

Design: reuse train_skill_v2.py primitives verbatim (import, never edit); this package only
adds the experiment matrix, the sample pool, the rubric double-cache, and the pluggable
training methods on top. See cookbook/exp/skill2lora/skill_quality_analysis.md sections
"AI 最终清单" and "AI 接口方案" for the frozen design decisions this code implements.
"""
