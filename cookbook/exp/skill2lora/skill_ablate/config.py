# Copyright (c) ModelScope Contributors. All rights reserved.
"""Declarative ablation matrix (E1-E12) + run order.

Frozen decisions (skill_quality_analysis.md):
- 12 experiments; run nothink before think, the SFT method (E12) LAST and manually gated.
- Unified knobs: executor frozen T=0; skill-model train T=1.0 × 8 rollouts; eval T=0.5 × 4
  rollouts (query-only, no rubric); skill-max-tokens = 8192 (think) / 4096 (nothink).
- view B = query-only; view A = rubric line, eval still query-only (knowledge-transfer probe).

This module is intentionally dependency-free (pure stdlib) so it can be imported and unit-
smoke-tested without torch / twinkle / a GPU.
"""
from dataclasses import dataclass
from typing import Dict, List

# --- training methods (internal keys) --------------------------------------------------
# bnpo        view B query-only GRPO/BNPO main loop (reuses v2 process_chunk verbatim).
# rl_ab       view A RL, AB split: first bare-problem greedy solve; WRONG problems -> A line
#             (skill sampled under query+rubric), RIGHT problems -> B line (query-only);
#             both go through executor greedy -> reward -> in-group BNPO, trained together.
# rl_err      view A RL, error-only: same as rl_ab but the B line is NOT trained
#             (single-variable contrast vs rl_ab on "does training the right-answer B line help").
# opsd        view A On-Policy Self-Distillation: student(query-only) logps pulled toward
#             teacher(query+rubric) logps per token (loss='opsd'); error problems only.
# improve_sft view A improve-skill + SFT: first-pass 1 skill; correct -> positive pool
#             (no leak, <=4096 chars); wrong -> rubric regen (2-in-8 pick 1) -> negative pool;
#             1:1 accumulate -> SFT.
# sft         view A plain SFT: bare-problem wrong -> rubric -> regen (2-in-8) -> accumulate SFT.
METHODS = ('bnpo', 'rl_ab', 'rl_err', 'opsd', 'improve_sft', 'sft')
VIEW_OF_METHOD = {'bnpo': 'B', 'rl_ab': 'A', 'rl_err': 'A',
                  'opsd': 'A', 'improve_sft': 'A', 'sft': 'A'}
STYLES = ('narrative', 'pitfall')
THINKINGS = ('on', 'off')


@dataclass(frozen=True)
class ExpSpec:
    name: str            # E1..E13
    method: str          # one of METHODS
    thinking: str        # 'on' | 'off'
    style: str           # 'narrative' | 'pitfall' (ignored by align='seam': SEAM prompts bypass style)
    optional: bool = False  # E12(sft): manually gated (RUN_SFT=1), runs last
    align: str = 'v2'    # 'v2' | 'seam' — sets v2._ALIGN_MODE (prompt/判分/executor 嵌套全开关)

    @property
    def view(self) -> str:
        return VIEW_OF_METHOD[self.method]

    @property
    def needs_rubric(self) -> bool:
        return self.view == 'A'

    @property
    def skill_max_tokens(self) -> int:
        # think must have room for <think> + <skills> (4096 truncates to an empty block).
        # seam align: 人工拍板用 8192（不复刻 SEAM 原版 4096：think 模式下 4096 会把大量候选截断在
        # <think> 里、压低 parseable，与“think 模式 skill-max-tokens 必须 8192”的矩阵规范保持一致）。
        if self.align == 'seam':
            return 8192
        return 8192 if self.thinking == 'on' else 4096

    @property
    def loss(self) -> str:
        return 'opsd' if self.method == 'opsd' else 'bnpo'

    @property
    def exp_dir(self) -> str:
        # output.ablate12/E{n}_{method}_{think}_{style}/  (seam align 加后缀区分)
        suffix = '_seam' if self.align == 'seam' else ''
        return f'{self.name}_{self.method}_{self.thinking}_{self.style}{suffix}'

    @property
    def swanlab_exp(self) -> str:
        return f'ablate12_{self.exp_dir}'


# --- the 12-experiment matrix (declarative; order field below drives execution) --------
MATRIX: List[ExpSpec] = [
    # group 1 — view B BNPO: think × style, no-rubric baseline
    ExpSpec('E1', 'bnpo', 'off', 'pitfall'),
    ExpSpec('E2', 'bnpo', 'off', 'narrative'),
    ExpSpec('E3', 'bnpo', 'on', 'pitfall'),
    ExpSpec('E4', 'bnpo', 'on', 'narrative'),
    # group 2 — view A RL-AB-mix: same grid as E1-E4, isolates "rubric rescues zero-grad groups"
    ExpSpec('E5', 'rl_ab', 'off', 'pitfall'),
    ExpSpec('E6', 'rl_ab', 'off', 'narrative'),
    ExpSpec('E7', 'rl_ab', 'on', 'pitfall'),
    ExpSpec('E8', 'rl_ab', 'on', 'narrative'),
    # group 3 — view A training-method comparison (fixed think+narrative), sft last & optional
    ExpSpec('E9', 'rl_err', 'on', 'narrative'),
    ExpSpec('E10', 'opsd', 'on', 'narrative'),
    ExpSpec('E11', 'improve_sft', 'on', 'narrative'),
    ExpSpec('E12', 'sft', 'on', 'narrative', optional=True),
    # group 4 — SEAM-align ablation: same data pipeline as the rest of the matrix, but ALL
    # prompt/parsing/executor-nesting rules follow SEAM (align='seam' -> v2._ALIGN_MODE):
    # actor uses SEAM EXPERIENCE_PROMPT (<memory_item>), executor sees the nested
    # prompt_text+response_text(+think), lpem-parity greedy scoring, actor budget 4096.
    # Query-only BNPO main loop (= SEAM's training form); eval stays the matrix-unified
    # query-only readout so E13 is directly comparable with E1-E12.
    ExpSpec('E13', 'bnpo', 'on', 'narrative', align='seam'),
]

# execution order: all nothink first, then think; E13 (seam-align baseline) right after E6;
# the data-hungry SFT method dead last.
RUN_ORDER: List[str] = ['E1', 'E2', 'E5', 'E6', 'E13', 'E3', 'E7', 'E8', 'E9', 'E10', 'E11', 'E4', 'E12']

BY_NAME: Dict[str, ExpSpec] = {e.name: e for e in MATRIX}


def get_spec(name: str) -> ExpSpec:
    key = name.strip().upper()
    if key not in BY_NAME:
        raise KeyError(f'unknown experiment {name!r}; valid: {sorted(BY_NAME)}')
    return BY_NAME[key]


def ordered_specs(include_optional: bool = True) -> List[ExpSpec]:
    specs = [BY_NAME[n] for n in RUN_ORDER]
    return specs if include_optional else [s for s in specs if not s.optional]


def _self_check() -> None:
    """Invariants that guard against typos when editing the matrix."""
    assert set(BY_NAME) == set(RUN_ORDER), 'RUN_ORDER must cover every matrix entry exactly once'
    assert len(RUN_ORDER) == len(set(RUN_ORDER)) == len(MATRIX), 'duplicate / missing names'
    for e in MATRIX:
        assert e.method in METHODS, f'{e.name}: bad method {e.method}'
        assert e.thinking in THINKINGS and e.style in STYLES, f'{e.name}: bad think/style'
        assert e.align in ('v2', 'seam'), f'{e.name}: bad align {e.align}'
    # nothink-before-think ordering within contiguous runs is a soft convention, not asserted.


if __name__ == '__main__':
    import sys
    _self_check()
    if '--plan' in sys.argv:
        # machine-readable run plan for the launcher: name<TAB>exp_dir<TAB>think<TAB>smt<TAB>optional
        for e in ordered_specs():
            print(f'{e.name}\t{e.exp_dir}\t{e.thinking}\t{e.skill_max_tokens}\t{int(e.optional)}')
        sys.exit(0)
    print(f'{len(MATRIX)} experiments; run order: {" -> ".join(RUN_ORDER)}')
    hdr = f'{"name":<4} {"view":<4} {"method":<12} {"think":<6} {"style":<10} {"align":<5} {"smt":<5} {"loss":<5} opt'
    print(hdr)
    print('-' * len(hdr))
    for e in ordered_specs():
        print(f'{e.name:<4} {e.view:<4} {e.method:<12} {e.thinking:<6} {e.style:<10} '
              f'{e.align:<5} {e.skill_max_tokens:<5} {e.loss:<5} {"Y" if e.optional else ""}')
