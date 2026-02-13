# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Test Preprocessor functionality:
1. CompetitionMathProcessor - process math problem data
2. CompetitionMathGRPOProcessor - process math problem data (GRPO format)
3. SelfCognitionProcessor - process self-cognition data (with placeholders)
4. AlpacaProcessor - process Alpaca format data (various cases)
5. Dataset.map change tests (auto-filter None, batched=False)
"""
import os
import pytest
from pathlib import Path

from twinkle.data_format import Message, Trajectory
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.preprocessor import (AlpacaProcessor, CompetitionMathGRPOProcessor, CompetitionMathProcessor,
                                  SelfCognitionProcessor)

# Get test data directory
TEST_DATA_DIR = Path(__file__).parent / 'test_data'


class TestCompetitionMathProcessor:
    """Test CompetitionMathProcessor"""

    def test_process_math_data(self):
        """Test processing math problem data"""
        jsonl_path = str(TEST_DATA_DIR / 'math_data.jsonl')
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        dataset.map(CompetitionMathProcessor())

        assert len(dataset) == 4

        # Check first sample
        sample = dataset[0]
        assert 'messages' in sample
        messages = sample['messages']
        assert len(messages) == 2
        assert messages[0]['role'] == 'user'
        assert messages[0]['content'] == 'What is 2+2?'
        assert messages[1]['role'] == 'assistant'
        assert messages[1]['content'] == 'The answer is 4.'

        # Check no system message
        assert all(msg['role'] != 'system' for msg in messages)

    def test_process_all_samples(self):
        """Test processing all samples"""
        jsonl_path = str(TEST_DATA_DIR / 'math_data.jsonl')
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        dataset.map(CompetitionMathProcessor())

        # Verify all samples have correct structure
        for i in range(len(dataset)):
            sample = dataset[i]
            assert 'messages' in sample
            messages = sample['messages']
            assert len(messages) == 2
            assert messages[0]['role'] == 'user'
            assert messages[1]['role'] == 'assistant'


class TestCompetitionMathGRPOProcessor:
    """Test CompetitionMathGRPOProcessor"""

    def test_process_grpo_data(self):
        """Test processing GRPO format data"""
        jsonl_path = str(TEST_DATA_DIR / 'math_data.jsonl')
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        dataset.map(CompetitionMathGRPOProcessor())

        assert len(dataset) == 4

        # Check first sample
        sample = dataset[0]
        assert 'messages' in sample
        messages = sample['messages']
        assert len(messages) == 3

        # Check system message
        assert messages[0]['role'] == 'system'
        assert 'math assistant' in messages[0]['content'].lower()

        # Check user message
        assert messages[1]['role'] == 'user'
        assert messages[1]['content'] == 'What is 2+2?'

        # Check assistant message (should be empty)
        assert messages[2]['role'] == 'assistant'
        assert messages[2]['content'] == ''

        # Check user_data
        assert 'user_data' in sample
        user_data = sample['user_data']
        assert len(user_data) == 1
        assert user_data[0][0] == 'solution'
        assert user_data[0][1] == 'The answer is 4.'

    def test_user_data_storage(self):
        """Test user_data storage"""
        jsonl_path = str(TEST_DATA_DIR / 'math_data.jsonl')
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        dataset.map(CompetitionMathGRPOProcessor())

        # Verify all samples have user_data
        for i in range(len(dataset)):
            sample = dataset[i]
            assert 'user_data' in sample
            user_data = sample['user_data']
            assert len(user_data) == 1
            assert user_data[0][0] == 'solution'


class TestSelfCognitionProcessor:
    """Test SelfCognitionProcessor"""

    def test_process_self_cognition_data(self):
        """Test processing self-cognition data"""
        jsonl_path = str(TEST_DATA_DIR / 'self_cognition_data.jsonl')
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        dataset.map(SelfCognitionProcessor('twinkle模型', 'twinkle团队'))

        assert len(dataset) == 3

        # Check first sample
        sample = dataset[0]
        assert 'messages' in sample
        messages = sample['messages']
        assert len(messages) == 3

        # Check system message
        assert messages[0]['role'] == 'system'
        assert messages[0]['content'] == 'You are a helpful assistant.'

        # Check user message (placeholders should be replaced)
        assert messages[1]['role'] == 'user'
        assert messages[1]['content'] == 'What is twinkle模型?'
        assert '{{NAME}}' not in messages[1]['content']
        assert '{{AUTHOR}}' not in messages[1]['content']

        # Check assistant message (placeholders should be replaced)
        assert messages[2]['role'] == 'assistant'
        assert messages[2]['content'] == 'twinkle模型 is a language model developed by twinkle团队.'
        assert '{{NAME}}' not in messages[2]['content']
        assert '{{AUTHOR}}' not in messages[2]['content']

    def test_placeholder_replacement(self):
        """Test placeholder replacement"""
        jsonl_path = str(TEST_DATA_DIR / 'self_cognition_data.jsonl')
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        dataset.map(SelfCognitionProcessor('test_model', 'test_author'))

        # Verify all samples have placeholders replaced
        for i in range(len(dataset)):
            sample = dataset[i]
            messages = sample['messages']
            for msg in messages:
                assert '{{NAME}}' not in msg['content']
                assert '{{AUTHOR}}' not in msg['content']
                if msg['role'] in ['user', 'assistant']:
                    assert 'test_model' in msg['content'] or 'test_author' in msg['content']


class TestAlpacaProcessor:
    """Test AlpacaProcessor - various cases"""

    def test_alpaca_instruction_only(self):
        """Test instruction-only case"""
        jsonl_path = str(TEST_DATA_DIR / 'alpaca_data.jsonl')
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        dataset.map(AlpacaProcessor())

        # Find instruction-only sample (4th sample)
        sample = dataset[3]  # "What is the capital of France?"
        messages = sample['messages']
        assert len(messages) == 2
        assert messages[0]['role'] == 'user'
        assert messages[0]['content'] == 'What is the capital of France?'
        assert messages[1]['role'] == 'assistant'
        assert messages[1]['content'] == 'The capital of France is Paris.'

    def test_alpaca_instruction_with_input(self):
        """Test instruction + input case"""
        jsonl_path = str(TEST_DATA_DIR / 'alpaca_data.jsonl')
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        dataset.map(AlpacaProcessor())

        # Find sample with input (2nd sample)
        sample = dataset[1]  # "Translate the following text" + "Hello"
        messages = sample['messages']
        assert len(messages) == 2
        assert messages[0]['role'] == 'user'
        assert 'Translate the following text' in messages[0]['content']
        assert 'Hello' in messages[0]['content']
        assert '\n' in messages[0]['content']  # Should contain newline
        assert messages[1]['role'] == 'assistant'
        assert messages[1]['content'] == '你好'

    def test_alpaca_empty_input(self):
        """Test empty input string case"""
        jsonl_path = str(TEST_DATA_DIR / 'alpaca_data.jsonl')
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        dataset.map(AlpacaProcessor())

        # Find sample with empty input (1st sample)
        sample = dataset[0]  # "Explain what AI is" with empty input
        messages = sample['messages']
        assert len(messages) == 2
        assert messages[0]['role'] == 'user'
        assert messages[0]['content'] == 'Explain what AI is'
        assert '\n' not in messages[0]['content']

    def test_alpaca_missing_fields(self):
        """Test tolerance for missing fields"""
        # Create test data with missing fields
        import json
        import tempfile

        test_data = [
            {
                'instruction': 'Test',
                'output': 'Result'
            },
            {
                'instruction': 'Test2',
                'input': 'Input2'
            },
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
            temp_path = f.name

        try:
            dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=temp_path))
            dataset.map(AlpacaProcessor())

            # First sample should process normally (missing input)
            assert len(dataset) >= 1
            sample = dataset[0]
            messages = sample['messages']
            assert messages[0]['content'] == 'Test'
        finally:
            os.unlink(temp_path)

    def test_alpaca_all_samples(self):
        """Test processing all Alpaca format samples"""
        jsonl_path = str(TEST_DATA_DIR / 'alpaca_data.jsonl')
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        dataset.map(AlpacaProcessor())

        # Verify all samples have correct structure
        for i in range(len(dataset)):
            sample = dataset[i]
            assert 'messages' in sample
            messages = sample['messages']
            assert len(messages) == 2
            assert messages[0]['role'] == 'user'
            assert messages[1]['role'] == 'assistant'
            assert messages[0]['content']
            assert messages[1]['content']


class TestDatasetMapChanges:
    """Test Dataset.map changes"""

    def test_batched_false(self):
        """Test batched=False setting"""
        jsonl_path = str(TEST_DATA_DIR / 'math_data.jsonl')
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))

        # Verify map sets batched=False
        dataset.map(CompetitionMathProcessor())

        # Verify processing result is correct (single-sample processing)
        assert len(dataset) == 4
        for i in range(len(dataset)):
            sample = dataset[i]
            assert 'messages' in sample
            # Each sample should have independent messages
            assert isinstance(sample['messages'], list)

    def test_load_from_cache_file_false(self):
        """Test load_from_cache_file=False default"""
        jsonl_path = str(TEST_DATA_DIR / 'math_data.jsonl')
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))

        # Multiple map calls should not use cache
        dataset.map(CompetitionMathProcessor())
        first_result = dataset[0]['messages'][0]['content']

        # Modify processor, process again
        class ModifiedProcessor(CompetitionMathProcessor):

            def __call__(self, row):
                traj = super().__call__(row)
                traj['messages'][0]['content'] = 'Modified: ' + traj['messages'][0]['content']
                return traj

        dataset2 = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        dataset2.map(ModifiedProcessor())
        second_result = dataset2[0]['messages'][0]['content']

        assert first_result != second_result
        assert 'Modified: ' in second_result

    def test_processor_string_name(self):
        """Test loading processor by string name"""
        jsonl_path = str(TEST_DATA_DIR / 'math_data.jsonl')
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))

        dataset.map('CompetitionMathProcessor')

        assert len(dataset) == 4
        sample = dataset[0]
        assert 'messages' in sample

    def test_processor_with_init_args(self):
        """Test initializing processor with init_args"""
        jsonl_path = str(TEST_DATA_DIR / 'self_cognition_data.jsonl')
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))

        dataset.map('SelfCognitionProcessor', init_args={'model_name': 'test_model', 'model_author': 'test_author'})

        assert len(dataset) == 3
        sample = dataset[0]
        messages = sample['messages']
        assert 'test_model' in messages[1]['content'] or 'test_author' in messages[1]['content']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
