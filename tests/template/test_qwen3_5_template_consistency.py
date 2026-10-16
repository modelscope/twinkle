# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for the Qwen3.5-family template registration and encode consistency.

Two layers:

  - ``TestModelTemplateMapping`` is a pure unit test over
    ``MODEL_TEMPLATE_MAPPING``; no model files needed, always runs.
  - ``TestEncodeMatchesApplyChatTemplate`` asserts that
    ``Template.encode()`` reproduces ``processor.apply_chat_template()``
    byte-for-byte. It is parametrized over whichever Qwen3.5-family models
    are present locally (Qwen3.5 / Qwen3.6 / Qwen3.8 all report
    ``model_type: qwen3_5*``), and skips when none are available.

Why the consistency layer matters: ``encode`` routes through
``apply_chat_template`` but first applies ``_to_standard_reasoning_content``
(inline ``<think>`` -> ``reasoning_content`` field) and the Qwen3 template
patches. Any of those can silently desynchronize the rendered text from the
token stream, which breaks multi-turn rollout bridge alignment. Qwen3.8 makes
this concrete: it dropped the template's inline ``</think>`` parsing and
flipped the ``preserve_thinking`` default to true, so feeding raw inline
thinking straight to ``apply_chat_template`` yields a doubly-nested
``<think>`` block. ``test_inline_thinking_is_normalized_not_double_wrapped``
pins down that twinkle's normalization prevents exactly that.

Point the tests at specific checkouts with
``TWINKLE_TEST_QWEN35_MODEL_DIRS=/path/a:/path/b``.
"""
import json
import os
import pytest
import re
from pathlib import Path
from typing import List, Tuple

import twinkle
from twinkle.data_format import Message, Trajectory
from twinkle.server.utils.template_utils import DEFAULT_TEMPLATE, get_template_for_model

twinkle.initialize(mode='local')

_QWEN3_5_TEMPLATE = 'Qwen3_5Template'

# ---------------------------------------------------------------------------
# Part A: model name -> template mapping (no model files required)
# ---------------------------------------------------------------------------


class TestModelTemplateMapping:
    """Qwen3.5 / 3.6 / 3.8 share one architecture, hence one template."""

    @pytest.mark.parametrize(
        'model_name',
        [
            'Qwen3.5-4B',
            'Qwen3.5-35B-A3B',
            'ms://Qwen/Qwen3.5-9B',
            'Qwen3.6-27B',
            'Qwen3.6-35B-A3B',
            'ms://Qwen/Qwen3.6-27B',
            # Qwen3.8 ships under a preview org and carries a date suffix;
            # both must still resolve through the 'Qwen3.8' substring.
            'Qwen3.8-27B',
            'Qwen3.8-27B-0811',
            'QM-preview/Qwen3.8-27B-0811',
            'ms://QM-preview/Qwen3.8-27B-0811',
        ],
    )
    def test_qwen3_5_family_maps_to_qwen3_5_template(self, model_name):
        assert get_template_for_model(model_name) == _QWEN3_5_TEMPLATE

    @pytest.mark.parametrize('model_name', [
        'Qwen2.5-7B-Instruct',
        'Qwen3-8B',
        'llama-3-8b',
        'DeepSeek-V4-Flash',
        '',
    ])
    def test_non_family_models_fall_back_to_default(self, model_name):
        assert get_template_for_model(model_name) == DEFAULT_TEMPLATE

    def test_qwen3_8_is_registered(self):
        """Regression guard: Qwen3.8 must not silently fall back to Template.

        Without the mapping entry the default template renders a different
        prompt, which is a near-invisible training bug.
        """
        assert get_template_for_model('Qwen3.8-27B-0811') != DEFAULT_TEMPLATE


# ---------------------------------------------------------------------------
# Local model discovery
# ---------------------------------------------------------------------------


def _config_dir(root: Path) -> Path:
    """Resolve a model root to the directory holding config.json."""
    if (root / 'config.json').is_file():
        return root
    for candidate in sorted(root.glob('snapshots/*')):
        if (candidate / 'config.json').is_file():
            return candidate
    return root


def _is_usable(cfg_dir: Path) -> bool:
    """Require a Qwen3.5-family config *and* a chat template to render with."""
    cfg_path = cfg_dir / 'config.json'
    if not cfg_path.is_file():
        return False
    try:
        model_type = json.loads(cfg_path.read_text()).get('model_type', '')
    except (json.JSONDecodeError, OSError):
        return False
    if not str(model_type).startswith('qwen3_5'):
        return False
    if (cfg_dir / 'chat_template.jinja').is_file():
        return True
    tok_cfg = cfg_dir / 'tokenizer_config.json'
    try:
        return bool(json.loads(tok_cfg.read_text()).get('chat_template'))
    except (json.JSONDecodeError, OSError):
        return False


def _cache_roots() -> List[Path]:
    """Candidate ModelScope cache roots, covering both on-disk layouts.

    ``MODELSCOPE_CACHE`` may or may not already include the ``hub`` segment,
    and models live under ``<root>/models``, so probe each combination.
    """
    bases = []
    env_cache = os.getenv('MODELSCOPE_CACHE')
    if env_cache:
        bases.append(Path(env_cache))
    bases += [Path.home() / '.cache' / 'modelscope', Path('/root/.cache/modelscope')]

    roots, seen = [], set()
    for base in bases:
        for models_dir in (base / 'models', base / 'hub' / 'models'):
            resolved = str(models_dir)
            if resolved not in seen and models_dir.is_dir():
                seen.add(resolved)
                roots.append(models_dir)
    return roots


# Official releases only: finetuned derivatives share the base template and
# would just slow the suite down. The 3.8 preview carries a date suffix, so
# match on the family prefix rather than pinning exact names.
_OFFICIAL_DIR = re.compile(r'^(Qwen|QM-preview)--Qwen3\.[568](-|$)')


def _discover_models() -> List[Tuple[str, str]]:
    """Return ``(label, path)`` for every local Qwen3.5-family model."""
    override = os.getenv('TWINKLE_TEST_QWEN35_MODEL_DIRS')
    if override:
        roots = [Path(p) for p in override.split(':') if p]
    else:
        roots = [p for models_dir in _cache_roots() for p in models_dir.iterdir() if _OFFICIAL_DIR.match(p.name)]

    found, seen = [], set()
    for root in sorted(roots, key=lambda p: p.name):
        cfg_dir = _config_dir(root)
        if root.name not in seen and _is_usable(cfg_dir):
            seen.add(root.name)
            found.append((root.name, str(cfg_dir)))
    return found


_MODELS = _discover_models()
_MODEL_IDS = [label for label, _ in _MODELS]

requires_model = pytest.mark.skipif(
    not _MODELS,
    reason='No local Qwen3.5-family model found; set TWINKLE_TEST_QWEN35_MODEL_DIRS to run',
)


@pytest.fixture(scope='module', params=[path for _, path in _MODELS], ids=_MODEL_IDS)
def template(request):
    """A Qwen3_5Template bound to a local checkout (tokenizer/processor only)."""
    from twinkle.template import Qwen3_5Template
    try:
        return Qwen3_5Template(model_id=request.param, max_length=8192)
    except Exception as e:  # noqa: BLE001 - environment issue, not a test failure
        pytest.skip(f'Failed to build template for {request.param}: {e}')


# ---------------------------------------------------------------------------
# Part B: encode() must reproduce apply_chat_template() byte-for-byte
# ---------------------------------------------------------------------------

_PLAIN_CASES = {
    'single_turn': [
        {
            'role': 'user',
            'content': 'What is 1+1?'
        },
        {
            'role': 'assistant',
            'content': '2'
        },
    ],
    'multi_turn': [
        {
            'role': 'user',
            'content': 'a'
        },
        {
            'role': 'assistant',
            'content': 'b'
        },
        {
            'role': 'user',
            'content': 'c'
        },
        {
            'role': 'assistant',
            'content': 'd'
        },
    ],
    'with_system': [
        {
            'role': 'system',
            'content': 'You are a helpful assistant.'
        },
        {
            'role': 'user',
            'content': 'hi'
        },
        {
            'role': 'assistant',
            'content': 'ok'
        },
    ],
    'reasoning_content_field': [
        {
            'role': 'user',
            'content': 'hi'
        },
        {
            'role': 'assistant',
            'content': 'ok',
            'reasoning_content': 'let me think'
        },
    ],
    'unicode_and_code': [
        {
            'role': 'user',
            'content': '用 Python 写平方函数，你好世界！'
        },
        {
            'role': 'assistant',
            'content': '```python\ndef f(x):\n    return x**2\n```'
        },
    ],
}

_TOOLS = [{
    'type': 'function',
    'function': {
        'name': 'get_weather',
        'description': 'Get the weather for a city.',
        'parameters': {
            'type': 'object',
            'properties': {
                'city': {
                    'type': 'string'
                }
            },
            'required': ['city'],
        },
    },
}]


@requires_model
class TestEncodeMatchesApplyChatTemplate:
    """``decode(encode(traj)) == apply_chat_template(messages)``."""

    @staticmethod
    def _encoded_text(template, trajectory, **kwargs) -> str:
        encoded = template.encode(trajectory, **kwargs)
        return template.tokenizer.decode(encoded['input_ids'])

    @staticmethod
    def _reference_text(template, messages, tools=None, add_generation_prompt=False) -> str:
        """Mirror the kwargs ``Template._apply_chat_template`` passes through."""
        return template.processor.apply_chat_template(
            messages,
            tools=list(tools or []),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=template.enable_thinking,
        )

    @pytest.mark.parametrize('case', sorted(_PLAIN_CASES), ids=sorted(_PLAIN_CASES))
    def test_encode_matches_reference(self, template, case):
        messages = _PLAIN_CASES[case]
        trajectory = Trajectory(messages=[Message(**m) for m in messages])
        assert self._encoded_text(template, trajectory) == self._reference_text(template, messages)

    def test_encode_matches_reference_with_generation_prompt(self, template):
        messages = [{'role': 'user', 'content': 'What is 1+1?'}]
        trajectory = Trajectory(messages=[Message(**m) for m in messages])
        got = self._encoded_text(template, trajectory, add_generation_prompt=True)
        expected = self._reference_text(template, messages, add_generation_prompt=True)
        assert got == expected

    def test_encode_matches_reference_with_tools(self, template):
        messages = [
            {
                'role': 'user',
                'content': 'Weather in Beijing?'
            },
            {
                'role': 'assistant',
                'content': 'Let me check.'
            },
        ]
        trajectory = Trajectory(messages=[Message(**m) for m in messages], tools=_TOOLS)
        got = self._encoded_text(template, trajectory)
        expected = self._reference_text(template, messages, tools=_TOOLS)
        assert got == expected

    def test_inline_thinking_is_normalized_not_double_wrapped(self, template):
        """Inline ``<think>`` must be hoisted into ``reasoning_content``.

        On Qwen3.8 the template no longer parses inline ``</think>`` itself, so
        passing raw inline thinking to ``apply_chat_template`` produces a
        doubly-nested ``<think>`` block. ``encode`` must instead match the
        explicitly-normalized message list.
        """
        inline = [
            {
                'role': 'user',
                'content': 'hi'
            },
            {
                'role': 'assistant',
                'content': '<think>\nreason step\n</think>\n\nanswer'
            },
        ]
        normalized = [
            {
                'role': 'user',
                'content': 'hi'
            },
            {
                'role': 'assistant',
                'content': 'answer',
                'reasoning_content': 'reason step'
            },
        ]
        trajectory = Trajectory(messages=[Message(**m) for m in inline])
        got = self._encoded_text(template, trajectory)

        assert got == self._reference_text(template, normalized)
        # The exact corruption this guards against.
        assert '<think>\n\n</think>\n\n<think>' not in got
        assert got.count('<think>') == got.count('</think>')

    def test_labels_align_with_input_ids(self, template):
        """Sanity: the assistant span is supervised, the prompt span is not."""
        trajectory = Trajectory(messages=[Message(**m) for m in _PLAIN_CASES['single_turn']])
        encoded = template.encode(trajectory)
        assert len(encoded['labels']) == len(encoded['input_ids'])
        assert (encoded['labels'] == -100).sum() > 0
        assert (encoded['labels'] != -100).sum() > 0
