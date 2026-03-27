# Copyright (c) ModelScope Contributors. All rights reserved.
"""
DPO (Direct Preference Optimization) Data Preprocessors.

These preprocessors convert various preference dataset formats into the standard
format required by Twinkle for DPO training.

DPO output format:
    - positive: Trajectory - chosen response trajectory
    - negative: Trajectory - rejected response trajectory
"""
from typing import Any, Dict, List, Optional, Union

from twinkle.data_format import Message, Trajectory
from .base import Preprocessor


class DPOProcessor(Preprocessor):
    """Generic DPO preference data preprocessor.

    Converts preference data with chosen/rejected pairs into positive/negative Trajectories.
    Supports multiple common dataset formats.

    Expected input format (one of):
        1. {'prompt': str, 'chosen': str, 'rejected': str}
        2. {'prompt': str, 'chosen': List[Message], 'rejected': List[Message]}
        3. {'messages': List[Message], 'chosen': str, 'rejected': str}
        4. {'chosen': List[Message], 'rejected': List[Message]} (full conversations)

    Output format:
        - positive: Trajectory with chosen response
        - negative: Trajectory with rejected response

    Args:
        system: Optional system prompt to prepend.
        chosen_key: Key for chosen response (default: 'chosen').
        rejected_key: Key for rejected response (default: 'rejected').
        prompt_key: Key for prompt/question (default: 'prompt').
        messages_key: Key for conversation messages (default: 'messages').
    """

    def __init__(
        self,
        system: Optional[str] = None,
        chosen_key: str = 'chosen',
        rejected_key: str = 'rejected',
        prompt_key: str = 'prompt',
        messages_key: str = 'messages',
    ):
        self.system = system
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key
        self.prompt_key = prompt_key
        self.messages_key = messages_key

    def _parse_response(self, response: Union[str, List[Dict], List[Message]]) -> List[Message]:
        """Parse response into list of Messages."""
        if isinstance(response, str):
            return [Message(role='assistant', content=response)]
        elif isinstance(response, list):
            messages = []
            for msg in response:
                if isinstance(msg, Message):
                    messages.append(msg)
                elif isinstance(msg, dict):
                    messages.append(Message(role=msg.get('role', 'assistant'), content=msg.get('content', '')))
            return messages
        return [Message(role='assistant', content=str(response))]

    def _build_prompt_messages(self, row: Dict[str, Any]) -> List[Message]:
        """Build prompt messages from row data."""
        messages = []

        # Add system message if provided
        if self.system:
            messages.append(Message(role='system', content=self.system))

        # Check for messages field (conversation format)
        if self.messages_key in row and row[self.messages_key]:
            raw_messages = row[self.messages_key]
            for msg in raw_messages:
                if isinstance(msg, Message):
                    messages.append(msg)
                elif isinstance(msg, dict):
                    messages.append(Message(role=msg.get('role'), content=msg.get('content', '')))
            return messages

        # Check for prompt field
        if self.prompt_key in row and row[self.prompt_key]:
            prompt = row[self.prompt_key]
            if isinstance(prompt, str):
                messages.append(Message(role='user', content=prompt))
            elif isinstance(prompt, list):
                for msg in prompt:
                    if isinstance(msg, Message):
                        messages.append(msg)
                    elif isinstance(msg, dict):
                        messages.append(Message(role=msg.get('role'), content=msg.get('content', '')))

        return messages

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Trajectory]:
        """Process a single row into positive/negative Trajectories.

        Returns:
            Dict with 'positive' and 'negative' Trajectory.
        """
        # Build prompt messages
        prompt_messages = self._build_prompt_messages(row)

        # Get chosen response
        chosen_raw = row.get(self.chosen_key, '')
        chosen_response = self._parse_response(chosen_raw)

        # Get rejected response
        rejected_raw = row.get(self.rejected_key, '')
        rejected_response = self._parse_response(rejected_raw)

        # Build full message lists
        chosen_messages = prompt_messages + chosen_response
        rejected_messages = prompt_messages + rejected_response

        return {
            'positive': Trajectory(messages=chosen_messages),
            'negative': Trajectory(messages=rejected_messages),
        }

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Process batched data into DPO format."""
        rows = self.map_col_to_row(rows)
        results = [self.preprocess(row) for row in rows]
        # Collect all positive and negative trajectories
        positive_list = [r['positive'] for r in results]
        negative_list = [r['negative'] for r in results]
        return {
            'positive': positive_list,
            'negative': negative_list,
        }


class HHRLHFProcessor(Preprocessor):
    """Preprocessor for Anthropic HH-RLHF dataset format.

    HH-RLHF format:
        {'chosen': "Human: ... Assistant: ...", 'rejected': "Human: ... Assistant: ..."}

    The conversations use "Human:" and "Assistant:" prefixes.
    """

    def __init__(self, system: Optional[str] = None):
        self.system = system

    def _parse_hh_conversation(self, text: str) -> List[Message]:
        """Parse HH-RLHF style conversation text into Messages."""
        messages = []

        if self.system:
            messages.append(Message(role='system', content=self.system))

        # Split by Human/Assistant markers
        parts = text.split('\n\nHuman: ')
        for i, part in enumerate(parts):
            if i == 0 and not part.startswith('Human: '):
                if part.strip():
                    if part.startswith('Human: '):
                        part = part[7:]
                    messages.append(Message(role='user', content=part.strip()))
                continue

            # Split Human and Assistant parts
            if '\n\nAssistant: ' in part:
                human_part, assistant_part = part.split('\n\nAssistant: ', 1)
                messages.append(Message(role='user', content=human_part.strip()))
                messages.append(Message(role='assistant', content=assistant_part.strip()))
            else:
                messages.append(Message(role='user', content=part.strip()))

        return messages

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Trajectory]:
        """Process HH-RLHF format row."""
        chosen_text = row.get('chosen', '')
        rejected_text = row.get('rejected', '')

        chosen_messages = self._parse_hh_conversation(chosen_text)
        rejected_messages = self._parse_hh_conversation(rejected_text)

        return {
            'positive': Trajectory(messages=chosen_messages),
            'negative': Trajectory(messages=rejected_messages),
        }

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        results = [self.preprocess(row) for row in rows]
        return {
            'positive': [r['positive'] for r in results],
            'negative': [r['negative'] for r in results],
        }


class UltraFeedbackProcessor(Preprocessor):
    """Preprocessor for UltraFeedback dataset format.

    UltraFeedback format:
        {
            'instruction': str,
            'completions': [
                {'response': str, 'overall_score': float, ...},
                ...
            ]
        }

    Selects highest and lowest scored completions as chosen/rejected.
    """

    def __init__(
        self,
        system: Optional[str] = None,
        instruction_key: str = 'instruction',
        completions_key: str = 'completions',
        response_key: str = 'response',
        score_key: str = 'overall_score',
    ):
        self.system = system
        self.instruction_key = instruction_key
        self.completions_key = completions_key
        self.response_key = response_key
        self.score_key = score_key

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Trajectory]]:
        """Process UltraFeedback format row."""
        instruction = row.get(self.instruction_key, '')
        completions = row.get(self.completions_key, [])

        if len(completions) < 2:
            return None

        # Sort by score
        scored_completions = [
            (c.get(self.score_key, 0), c.get(self.response_key, ''))
            for c in completions
            if c.get(self.response_key)
        ]

        if len(scored_completions) < 2:
            return None

        scored_completions.sort(key=lambda x: x[0], reverse=True)
        chosen_response = scored_completions[0][1]
        rejected_response = scored_completions[-1][1]

        # Build messages
        prompt_messages = []
        if self.system:
            prompt_messages.append(Message(role='system', content=self.system))
        prompt_messages.append(Message(role='user', content=instruction))

        chosen_messages = prompt_messages + [Message(role='assistant', content=chosen_response)]
        rejected_messages = prompt_messages + [Message(role='assistant', content=rejected_response)]

        return {
            'positive': Trajectory(messages=chosen_messages),
            'negative': Trajectory(messages=rejected_messages),
        }

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        results = [self.preprocess(row) for row in rows]
        results = [r for r in results if r is not None]
        if not results:
            return {}
        return {
            'positive': [r['positive'] for r in results],
            'negative': [r['negative'] for r in results],
        }


class ShareGPTDPOProcessor(Preprocessor):
    """Preprocessor for ShareGPT-style DPO datasets.

    Expected format:
        {
            'conversations': [
                {'from': 'human', 'value': '...'},
                {'from': 'gpt', 'value': '...'},
                ...
            ],
            'chosen': {'from': 'gpt', 'value': '...'},
            'rejected': {'from': 'gpt', 'value': '...'}
        }
    """

    ROLE_MAPPING = {
        'human': 'user',
        'gpt': 'assistant',
        'system': 'system',
        'user': 'user',
        'assistant': 'assistant',
    }

    def __init__(self, system: Optional[str] = None):
        self.system = system

    def _parse_sharegpt_message(self, msg: Dict) -> Message:
        """Parse ShareGPT format message."""
        role = self.ROLE_MAPPING.get(msg.get('from', ''), 'user')
        content = msg.get('value', '') or msg.get('content', '')
        return Message(role=role, content=content)

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Trajectory]:
        """Process ShareGPT DPO format row."""
        conversations = row.get('conversations', [])

        # Build prompt messages
        prompt_messages = []
        if self.system:
            prompt_messages.append(Message(role='system', content=self.system))

        for msg in conversations:
            prompt_messages.append(self._parse_sharegpt_message(msg))

        # Remove last message if it's assistant (will be replaced)
        if prompt_messages and prompt_messages[-1]['role'] == 'assistant':
            prompt_messages = prompt_messages[:-1]

        # Get chosen and rejected
        chosen_msg = row.get('chosen', {})
        rejected_msg = row.get('rejected', {})

        if isinstance(chosen_msg, dict):
            chosen_content = chosen_msg.get('value', '') or chosen_msg.get('content', '')
        else:
            chosen_content = str(chosen_msg)

        if isinstance(rejected_msg, dict):
            rejected_content = rejected_msg.get('value', '') or rejected_msg.get('content', '')
        else:
            rejected_content = str(rejected_msg)

        chosen_messages = prompt_messages + [Message(role='assistant', content=chosen_content)]
        rejected_messages = prompt_messages + [Message(role='assistant', content=rejected_content)]

        return {
            'positive': Trajectory(messages=chosen_messages),
            'negative': Trajectory(messages=rejected_messages),
        }

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        results = [self.preprocess(row) for row in rows]
        return {
            'positive': [r['positive'] for r in results],
            'negative': [r['negative'] for r in results],
        }


class IntelOrcaDPOProcessor(Preprocessor):
    """Preprocessor for Intel ORCA DPO dataset format.

    Expected format:
        {
            'system': str,
            'question': str,
            'chosen': str,
            'rejected': str
        }
    """

    def __init__(self, default_system: Optional[str] = None):
        self.default_system = default_system

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Trajectory]:
        """Process Intel ORCA DPO format row."""
        system = row.get('system', self.default_system)
        question = row.get('question', '')
        chosen = row.get('chosen', '')
        rejected = row.get('rejected', '')

        prompt_messages = []
        if system:
            prompt_messages.append(Message(role='system', content=system))
        prompt_messages.append(Message(role='user', content=question))

        chosen_messages = prompt_messages + [Message(role='assistant', content=chosen)]
        rejected_messages = prompt_messages + [Message(role='assistant', content=rejected)]

        return {
            'positive': Trajectory(messages=chosen_messages),
            'negative': Trajectory(messages=rejected_messages),
        }

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        results = [self.preprocess(row) for row in rows]
        return {
            'positive': [r['positive'] for r in results],
            'negative': [r['negative'] for r in results],
        }


class EmojiDPOProcessor(Preprocessor):
    """Preprocessor for shareAI/DPO-zh-en-emoji dataset format.

    Dataset format:
        {
            'prompt': str,
            'answer_zh': str,  # chosen response (Chinese)
            'answer_en': str,  # rejected response (English)
        }

    Output format:
        - positive: Trajectory with chosen (answer_zh)
        - negative: Trajectory with rejected (answer_en)

    Args:
        system: Optional system prompt.
        chosen_key: Key for chosen response (default: 'answer_zh').
        rejected_key: Key for rejected response (default: 'answer_en').
        prompt_key: Key for prompt (default: 'prompt').
    """

    def __init__(
        self,
        system: Optional[str] = None,
        chosen_key: str = 'answer_zh',
        rejected_key: str = 'answer_en',
        prompt_key: str = 'prompt',
    ):
        self.system = system
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key
        self.prompt_key = prompt_key

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Trajectory]:
        """Process a single row."""
        prompt = row.get(self.prompt_key, '')
        chosen = row.get(self.chosen_key, '')
        rejected = row.get(self.rejected_key, '')

        prompt_messages = []
        if self.system:
            prompt_messages.append(Message(role='system', content=self.system))
        prompt_messages.append(Message(role='user', content=prompt))

        chosen_messages = prompt_messages + [Message(role='assistant', content=chosen)]
        rejected_messages = prompt_messages + [Message(role='assistant', content=rejected)]

        return {
            'positive': Trajectory(messages=chosen_messages),
            'negative': Trajectory(messages=rejected_messages),
        }

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        results = [self.preprocess(row) for row in rows]
        return {
            'positive': [r['positive'] for r in results],
            'negative': [r['negative'] for r in results],
        }


class UltraFeedbackKTOProcessor(Preprocessor):
    """Preprocessor for ultrafeedback-binarized-preferences-cleaned-kto dataset.

    Dataset format:
        {
            'prompt': str,
            'completion': str,
            'label': bool,  # True for chosen, False for rejected
        }

    For KTO training, we need (prompt, completion, label) format.
    The label is stored in user_data.

    Args:
        system: Optional system prompt.
    """

    def __init__(self, system: Optional[str] = None):
        self.system = system

    def preprocess(self, row: Dict[str, Any]) -> Trajectory:
        """Process a single row for KTO."""
        prompt = row.get('prompt', '')
        completion = row.get('completion', '')
        label = row.get('label', True)

        messages = []
        if self.system:
            messages.append(Message(role='system', content=self.system))
        messages.append(Message(role='user', content=prompt))
        messages.append(Message(role='assistant', content=completion))

        return Trajectory(
            messages=messages,
            user_data=[('kto_label', label)]
        )

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows
