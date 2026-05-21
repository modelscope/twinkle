# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for Qwen3ChatTemplate chat_template patch.

Strategy:
  - Unit tests drive the patch directly against a mock tokenizer; no model
    download required, runs in CI without network.
  - Functional tests use ``jinja2`` to render a minimal assistant-branch
    template with the OLD vs NEW parse block on the exact orphan-``</think>``
    scenario that breaks multi-turn rollout bridge, asserting the patched
    template is byte-level round-trippable.
"""
import pytest
import warnings
from types import SimpleNamespace

from twinkle.patch import apply_patch
from twinkle.patch.qwen3_chat_template import _NEW, _OLD, Qwen3ChatTemplate

# ---------------------------------------------------------------------------
# Fixtures: minimal jinja harness reproducing the assistant-branch parse path
# ---------------------------------------------------------------------------

# Skeleton mimicking Qwen3.5 jinja assistant branch. The ``{block}`` placeholder
# receives either _OLD or _NEW verbatim, preserving their 12/16-space
# indentation so the patch's string-replace can locate OLD without drift.
# Only the last message is rendered (index-0 assistant), sufficient to expose
# the parse defect.
_SKELETON = '''\
{{%- for message in messages %}}
    {{%- set content = message.content %}}
    {{%- if message.role == "assistant" %}}
        {{%- set reasoning_content = '' %}}
        {{%- if message.reasoning_content is string %}}
            {{%- set reasoning_content = message.reasoning_content %}}
        {{%- else %}}
{block}
        {{%- endif %}}
        {{%- set reasoning_content = reasoning_content|trim %}}
        {{{{ '<|im_start|>assistant\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content + '<|im_end|>' }}}}
    {{%- endif %}}
{{%- endfor %}}
''' # noqa


def _render(block: str, content: str) -> str:
    """Render the minimal skeleton with given parse block and assistant content."""
    from jinja2 import Environment
    env = Environment()
    tmpl = env.from_string(_SKELETON.format(block=block))
    msg = SimpleNamespace(role='assistant', content=content, reasoning_content=None)
    return tmpl.render(messages=[msg])


# ---------------------------------------------------------------------------
# Unit tests: patch string-replacement mechanics
# ---------------------------------------------------------------------------


class TestPatchMechanics:

    def test_patch_replaces_old_with_new(self):
        fake = SimpleNamespace(chat_template=f'prefix\n{_OLD}\nsuffix')
        patched = apply_patch(fake, Qwen3ChatTemplate)
        assert patched is True
        assert _NEW in fake.chat_template
        assert _OLD not in fake.chat_template

    def test_patch_is_idempotent_on_second_call(self):
        fake = SimpleNamespace(chat_template=f'prefix\n{_OLD}\nsuffix')
        first = apply_patch(fake, Qwen3ChatTemplate)
        snapshot = fake.chat_template
        second = apply_patch(fake, Qwen3ChatTemplate)
        assert first is True
        assert second is False
        assert fake.chat_template == snapshot  # no double-patching

    def test_patch_warns_and_noops_on_unknown_template(self):
        fake = SimpleNamespace(chat_template='<some unrelated template>')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = apply_patch(fake, Qwen3ChatTemplate)
        assert result is False
        assert fake.chat_template == '<some unrelated template>'
        assert any('Qwen3ChatTemplate patch' in str(w.message) for w in caught)

    def test_patch_noops_on_none_or_missing_template(self):
        fake_none = SimpleNamespace(chat_template=None)
        fake_missing = SimpleNamespace()
        assert apply_patch(fake_none, Qwen3ChatTemplate) is False
        assert apply_patch(fake_missing, Qwen3ChatTemplate) is False

    def test_patch_replaces_only_first_occurrence(self):
        # Safety: replace(..., 1) prevents accidental double substitution.
        fake = SimpleNamespace(chat_template=f'{_OLD}\n---\n{_OLD}')
        apply_patch(fake, Qwen3ChatTemplate)
        assert fake.chat_template.count(_NEW) == 1
        assert fake.chat_template.count(_OLD) == 1  # second one left intact


# ---------------------------------------------------------------------------
# Functional tests: jinja rendering behavior on the real failure scenario
# ---------------------------------------------------------------------------


class TestRenderBehavior:
    """Render the minimal Qwen3 assistant branch and verify the patch fixes
    the orphan-</think> bug that breaks multi-turn rollout bridge."""

    # Content stored in messages[-1]['content'] by concat_input_feature after
    # sampler produces CoT ending in an orphan </think>. The generation_prompt
    # injected the opening <think>\n\n</think>\n\n into prompt_ids (not into
    # content), so content here has no opening <think>.
    CONTENT_WITH_ORPHAN = ('Step 1: Review blocks.\nStep 2: Decide.\n</think>\n\n'
                           '<tool_call>\n<function=extract>\n<parameter=ids>\n[1, 2]\n</parameter>\n'
                           '</function>\n</tool_call>')

    # Clean content (no </think> at all) — normal policy-compliant output.
    CONTENT_CLEAN = 'Step 1: Just answer.\n\n<tool_call>\n<function=a>\n</function>\n</tool_call>'

    # ---- OLD template: demonstrates the pre-patch byte-level mismatch ----

    def test_old_template_drops_empty_think_wrapper_on_orphan(self):
        rendered = _render(_OLD, self.CONTENT_WITH_ORPHAN)
        # Bug symptom: OLD template's parse branch hoists the CoT (Step 1/2)
        # into <think> block, so the assistant segment no longer begins with
        # the empty '<think>\n\n</think>\n\n' wrapper that the sampler's
        # generation_prompt actually injected into input_ids. This is the
        # 11-byte discrepancy that breaks multi-turn bridge text alignment.
        assert '<|im_start|>assistant\n<think>\n\n</think>\n\n' not in rendered
        # Confirm CoT was (incorrectly) moved into the reasoning block.
        assert '<think>\nStep 1: Review blocks.' in rendered

    # ---- NEW template (post-patch): content preserved intact ----

    def test_new_template_preserves_orphan_content(self):
        rendered = _render(_NEW, self.CONTENT_WITH_ORPHAN)
        # Post-patch: content.startswith('<think>') is False, parse branch
        # is skipped, reasoning_content stays empty, content stays verbatim.
        # Rendered output must contain the original content byte-for-byte.
        assert self.CONTENT_WITH_ORPHAN in rendered
        # And reasoning block must be empty (matches generation_prompt injection).
        assert '<think>\n\n</think>\n\n' + self.CONTENT_WITH_ORPHAN in rendered

    # ---- NEW template on clean content: behavior unchanged ----

    def test_new_template_clean_content_unchanged(self):
        rendered_old = _render(_OLD, self.CONTENT_CLEAN)
        rendered_new = _render(_NEW, self.CONTENT_CLEAN)
        # On clean content the two templates must produce identical output:
        # patch is strictly a bug-fix, no behavior change on happy path.
        assert rendered_old == rendered_new
        assert self.CONTENT_CLEAN in rendered_new

    # ---- NEW template on legitimate thinking content: still parsed ----

    def test_new_template_parses_proper_thinking_block(self):
        # Content produced when enable_thinking=True and model emits a proper
        # <think>...</think> wrapper (not our current case, but template must
        # keep supporting it).
        proper = '<think>\nLet me think.\n</think>\n\nHere is the answer.'
        rendered = _render(_NEW, proper)
        # reasoning_content should be extracted, content should be the tail.
        assert '<think>\nLet me think.\n</think>\n\nHere is the answer.<|im_end|>' in rendered

    # ---- Byte-level round-trip: current_text vs re-rendered s_after ----

    def test_bridge_roundtrip_orphan_case(self):
        """Simulate the multi-turn bridge check: current_text (decoded from
        input_ids, includes generation_prompt's empty think block) must be a
        strict prefix of s_after (re-rendered from messages). Pre-patch this
        fails by 11 bytes; post-patch it holds."""
        # What the decoded input_ids look like for this assistant turn:
        current_text = ('<|im_start|>assistant\n<think>\n\n</think>\n\n' + self.CONTENT_WITH_ORPHAN + '<|im_end|>')
        # What the chat_template renders the same assistant message as:
        rendered_old = _render(_OLD, self.CONTENT_WITH_ORPHAN).strip()
        rendered_new = _render(_NEW, self.CONTENT_WITH_ORPHAN).strip()
        # Pre-patch: rendered text diverges from current_text (bridge breaks).
        assert current_text not in rendered_old
        # Post-patch: current_text is reproduced byte-for-byte (bridge works).
        assert current_text in rendered_new
