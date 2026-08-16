# Copyright (c) ModelScope Contributors. All rights reserved.
"""RSI (recursive self-improvement) pipeline scripts.

Stages (each a standalone entry script):
    rsi_prepare.py  - step 1: read a raw source, parallel-preprocess, dump a subset.
    rsi_refine.py   - step 2: re-analyze/strengthen trajectories into a standard flow.
    rsi_rl.py       - step 3: multi-LoRA RL, one training query per round.
    rsi_distill.py  - step 4: dump llm_backup data, SFT the auxiliary-role LoRAs.
"""
