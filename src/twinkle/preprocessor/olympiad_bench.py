# Copyright (c) ModelScope Contributors. All rights reserved.
"""Preprocessor for OlympiadBench multimodal math/physics dataset."""
import re
from typing import Any, Dict, List, Optional

from twinkle.data_format import Message, Trajectory
from .base import Preprocessor


class OlympiadBenchProcessor(Preprocessor):
    """Preprocessor for OlympiadBench dataset (multimodal math/physics problems).

    OlympiadBench fields:
        - question (str): The problem text
        - solution (sequence): Step-by-step solution
        - final_answer (sequence): The final answer(s)
        - context (str): Additional context
        - image_1 to image_9: Optional images
        - subject (str): 'maths' or 'physics'
        - is_multiple_answer (bool): Whether multiple answers are expected
        - unit (str): Unit of the answer

    The preprocessor:
        1. Collects all non-null images
        2. Creates multimodal message with image placeholders
        3. Stores ground truth in user_data for reward computation
    """
    system_prompt_zh = ('你是一个专业的数学和物理解题助手。请仔细分析题目，'
                        '逐步推理解答，并将最终答案用 \\boxed{} 格式给出。'
                        '如果有多个答案，请用逗号分隔。')

    system_prompt_en = ('You are a professional math and physics problem solver. '
                        'Analyze the problem carefully, solve it step by step, '
                        'and provide your final answer in \\boxed{} format. '
                        'If there are multiple answers, separate them with commas.')

    def __init__(self, system: Optional[str] = None, language: str = 'zh'):
        """Initialize the preprocessor.

        Args:
            system: Custom system prompt. If None, uses default based on language.
            language: 'zh' for Chinese, 'en' for English. Default 'zh'.
        """
        self.language = language
        if system is not None:
            self.system = system
        else:
            self.system = self.system_prompt_zh if language == 'zh' else self.system_prompt_en

    def _collect_images(self, row: Dict[str, Any]) -> List[Any]:
        """Collect all non-null images from row.

        Note: Images are kept as-is (PIL or bytes) since preprocessing is lazy.
        Conversion to bytes happens only when needed for serialization.
        """
        images = []
        for i in range(1, 10):
            img = row.get(f'image_{i}')
            if img is not None:
                images.append(img)
        return images

    def _format_final_answer(self, final_answer: Any, unit: str = '') -> str:
        """Format final answer(s) as string for comparison."""
        if isinstance(final_answer, list):
            answers = [str(a).strip() for a in final_answer if a]
            answer_str = ', '.join(answers)
        else:
            answer_str = str(final_answer).strip() if final_answer else ''

        if unit and answer_str:
            answer_str = f'{answer_str} {unit}'
        return answer_str

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    def preprocess(self, row: Dict[str, Any]) -> Trajectory:
        question = row.get('question', '')
        context = row.get('context', '')
        final_answer = row.get('final_answer', [])
        unit = row.get('unit', '')

        # Collect images
        images = self._collect_images(row)

        # Build question content with image placeholders
        content_parts = []

        # Add images directly in blocks
        for img in images:
            content_parts.append({'type': 'image', 'url': img})

        # Add context if exists
        if context:
            content_parts.append({'type': 'text', 'text': f'背景信息：{context}\n\n'})

        # Add question
        content_parts.append({'type': 'text', 'text': question})

        # Create user message with multimodal content
        if images:
            user_message = Message(
                role='user',
                content=content_parts,
            )
        else:
            # No images, use plain text
            full_text = f'背景信息：{context}\n\n{question}' if context else question
            user_message = Message(role='user', content=full_text)

        messages = [
            Message(role='system', content=self.system),
            user_message,
        ]

        # Format ground truth
        ground_truth = self._format_final_answer(final_answer, unit)

        return Trajectory(
            messages=messages,
            user_data=[
                ('ground_truth', ground_truth),
            ],
        )
