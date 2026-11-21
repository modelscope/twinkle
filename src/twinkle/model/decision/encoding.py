# Copyright (c) ModelScope Contributors. All rights reserved.
# Clef span encoding adapted from ms-swift fbc61cf2 (Apache-2.0).
import json
import string
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from .heads import QUESTION_TYPES

_CLEF_SYSTEM_PROMPT = ('Read the complete state and schema. Decide every field jointly. Each answer '
                       'must be exactly one of that field\'s allowed options.')
_CLEF_PREFIX_TMPL = '<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\nSTATE:\n'
_CLEF_SUFFIX = '\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:'
_CLEF_NOUL_CRITERIA = {
    'true': 'The proposition is true or the answer is yes.',
    'false': 'The proposition is false or the answer is no.',
}


class DecisionEncoder:
    """Encode normalized decision rows without inserting answer labels into prompts."""

    def __init__(self, processor, family='generic', max_length=8192):
        if family not in ('generic', 'jev', 'clef', 'omnijev', 'startlux'):
            raise ValueError(f'Unknown decision family {family}')
        self.processor = processor
        self.tokenizer = getattr(processor, 'tokenizer', processor)
        self.family = family
        self.max_length = max_length

    def _tokenize(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    @staticmethod
    def _clef_fields(inputs):
        row = inputs.extra_kwargs
        state = row.get('state', row['messages'][0]['content'])
        if isinstance(state, str):
            try:
                state = json.loads(state)
            except ValueError:
                pass
        criteria = [json.loads(c) if isinstance(c, str) else c for c in row['criteria']]
        fields = list(zip(row['question_ids'], row['kinds'], row['questions'], criteria))
        if not fields or any(len(row[k]) != len(fields) for k in ('question_ids', 'kinds', 'questions', 'criteria')):
            raise ValueError('Inconsistent Clef schema fields')
        return state, fields

    def __call__(self, row):
        if row.get('videos'):
            raise ValueError('Precompose videos with video_state before encoding')
        inputs = SimpleNamespace(extra_kwargs=row, images=row.get('images'), videos=row.get('videos'))
        if self.family == 'startlux':
            from startlux_decision import jevfmt
            data = json.loads(row['startlux_row'])
            order = [o['id'] for o in data['options']]
            ids, rendered = jevfmt.render_ids(data, self.tokenizer, order, max_length=self.max_length or 8192)
            if rendered != order:
                raise ValueError('Renderer changed option order')
            encoded = {
                'input_ids': ids,
                'decision_meta': {
                    'n_options': [len(order)],
                    'kinds': [QUESTION_TYPES[data['type']]]
                }
            }
        elif self.family == 'clef':
            encoded = self._encode_clef(inputs)
        elif self.family == 'omnijev':
            encoded = self._encode_omnijev(row)
        else:
            if len(row['questions']) != 1:
                raise ValueError('JEV/generic requires one question per encoded row')
            kind, options = row['kinds'][0], row['options'][0]
            if kind == 'choice' and len(options) > 16:
                raise ValueError('This verbalizer supports at most 16 choice options')
            lines = [f'{string.ascii_uppercase[i]}) {o}'
                     for i, o in enumerate(options)] if kind == 'choice' else options
            prompt = (f'[kind] {kind}\n[state] {row["messages"][0]["content"]}\n'
                      f'[question] {row["questions"][0]}\n[options]\n' + '\n'.join(lines) + '\n[decision]:')
            if row.get('images'):
                if self.family != 'jev' or not hasattr(self.processor, 'image_processor'):
                    raise ValueError('Image decisions require a JEV vision processor')
                from PIL import Image
                frames = []
                for source in row['images']:
                    if isinstance(source, Image.Image):
                        frames.append(source.convert('RGB'))
                    else:
                        with Image.open(source) as image:
                            frames.append(image.convert('RGB'))
                marker = '<|vision_start|><|image_pad|><|vision_end|>'
                prompt = prompt.replace('[state] ', '[state] ' + marker * len(frames), 1)
                encoded = dict(self.processor(text=[prompt], images=frames, return_tensors='pt'))
                encoded['input_ids'] = encoded['input_ids'][0].tolist()
                encoded.pop('attention_mask', None)
                if 'mm_token_type_ids' in encoded:
                    encoded['mm_token_type_ids'] = encoded['mm_token_type_ids'][0].tolist()
            else:
                encoded = {'input_ids': self._tokenize(prompt)}
            encoded['decision_meta'] = {
                'n_options': [len(options)],
                'kinds': [QUESTION_TYPES[kind]],
                'mm_counts': {
                    'image': len(row.get('images') or []),
                    'video': 0
                }
            }
        if self.max_length is not None and len(encoded['input_ids']) > self.max_length:
            raise ValueError('Decision prompt exceeds max_length; answer positions must not be truncated')
        if 'target_probs' in row:
            encoded['target_probs'] = row['target_probs']
        return encoded

    def _encode_omnijev(self, row):
        from . import omnijev_media as media
        from .omnijev_branch import OPT_CLOSE, OPT_OPEN
        instruction = row['messages'][0]['content']
        images = row.get('images') or []
        video = row.get('omnijev_video')
        if images:
            frames, content = media.video_content(media.compose_state(images, video), video, instruction)
        else:
            frames, content = [], [{'type': 'text', 'text': instruction}]
        messages = [{
            'role': 'user',
            'content': content
        }, {
            'role': 'assistant',
            'content': '<think>\n' + media.option_body(row['options'][0])
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, continue_final_message=True)
        if frames:
            encoded = dict(self.processor(text=[text], images=frames, return_tensors='pt'))
            encoded['input_ids'] = encoded['input_ids'][0].tolist()
            encoded.pop('attention_mask', None)
            if 'mm_token_type_ids' in encoded:
                encoded['mm_token_type_ids'] = encoded['mm_token_type_ids'][0].tolist()
        else:
            encoded = {'input_ids': self._tokenize(text)}
        ids = encoded['input_ids']

        def positions(token):
            sub = self._tokenize(token)
            if not sub:
                raise ValueError('Missing option marker tokens')
            return [i for i in range(len(ids) - len(sub) + 1) if ids[i:i + len(sub)] == sub]

        opens = positions(OPT_OPEN)
        closes = [i + len(self._tokenize(OPT_CLOSE)) - 1 for i in positions(OPT_CLOSE)]
        if not opens or len(opens) != len(closes) or any(a >= b for a, b in zip(opens, closes)):
            raise ValueError('Invalid OmniJev option markers')
        markers = list(zip(opens, closes))
        encoded['decision_meta'] = {
            'n_options': [len(row['omnijev_column_keys'][0])],
            'kinds': [QUESTION_TYPES[row['kinds'][0]]],
            'option_markers': [markers],
            'mm_counts': {
                'image': len(frames),
                'video': 0
            },
            'question_spans': [(opens[0] - 1, opens[0])],
            'option_spans': [markers]
        }
        return encoded

    @staticmethod
    def _clef_render(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True)

    def _encode_clef(self, inputs) -> Dict[str, Any]:
        if getattr(inputs, 'images', None) or getattr(inputs, 'videos', None):
            raise NotImplementedError(
                'ClefTemplate supports text-only records this phase (Clef is a text decision model); '
                'images/videos require the deferred multimodal span-offset port.')
        from .heads import QUESTION_TYPES
        state, questions = self._clef_fields(inputs)

        schema_ids: List[int] = self._tokenize('\n\nSCHEMA FIELDS:\n')
        question_spans: List[Tuple[int, int]] = []
        option_spans: List[List[Tuple[int, int]]] = []
        kinds_int: List[int] = []
        n_options: List[int] = []
        for q_index, (qid, kind, instructions, criteria) in enumerate(questions):
            schema_ids += self._tokenize(f'\nFIELD {q_index + 1}\nID: {qid}\nTYPE: {kind}\nINSTRUCTION: ')
            question_start = len(schema_ids)
            if instructions is None or instructions == '':
                instructions = str(qid)
            schema_ids += self._tokenize(self._clef_render(instructions))
            question_end = len(schema_ids)
            schema_ids += self._tokenize('\nALLOWED OPTIONS:\n')
            spans: List[Tuple[int, int]] = []
            for o_index, (option_id, description) in enumerate(self._question_options(kind, criteria)):
                schema_ids += self._tokenize(f'OPTION {o_index + 1}: ')
                option_start = len(schema_ids)
                semantics: Dict[str, Any] = {'option_id': option_id}
                if description is not None:
                    semantics['description'] = description
                schema_ids += self._tokenize(self._clef_render(semantics))
                spans.append((option_start, len(schema_ids)))
                schema_ids += self._tokenize('\n')
            schema_ids += self._tokenize('END FIELD\n')
            question_spans.append((question_start, question_end))
            option_spans.append(spans)
            kinds_int.append(QUESTION_TYPES[kind])
            n_options.append(len(spans))

        prefix_ids = self._tokenize(_CLEF_PREFIX_TMPL.format(system=_CLEF_SYSTEM_PROMPT))
        suffix_ids = self._tokenize(_CLEF_SUFFIX)
        state_ids = self._tokenize(self._clef_render(state))
        if self.max_length is not None:
            fixed_length = len(prefix_ids) + len(schema_ids) + len(suffix_ids)
            if fixed_length > self.max_length:
                raise ValueError(
                    f'Clef schema requires {fixed_length} tokens before state; max_length={self.max_length}.')
            state_ids = state_ids[:self.max_length - fixed_length]
        schema_offset = len(prefix_ids) + len(state_ids)
        question_spans = [(s + schema_offset, e + schema_offset) for s, e in question_spans]
        option_spans = [[(s + schema_offset, e + schema_offset) for s, e in spans] for spans in option_spans]
        input_ids = prefix_ids + state_ids + schema_ids + suffix_ids
        if not input_ids or not questions:
            raise ValueError('Clef record produced no model input or questions.')
        return {
            'input_ids': input_ids,
            'decision_meta': {
                'n_options': n_options,
                'kinds': kinds_int,
                'question_spans': question_spans,
                'option_spans': option_spans,
            },
        }

    @staticmethod
    def _question_options(kind: str, criteria: Any) -> List[Tuple[str, Any]]:
        from .heads import clef_option_ids
        ids = clef_option_ids(kind, criteria)
        if kind == 'noul':
            merged = dict(_CLEF_NOUL_CRITERIA)
            merged.update(criteria or {})
            return [(key, merged[key]) for key in ids]
        if kind == 'choice':
            crit = {str(k): v for k, v in (criteria or {}).items()}
            return [(key, crit[key]) for key in ids]
        # score
        crit = criteria or []
        return [(ids[i], crit[i]) for i in range(len(crit))]
