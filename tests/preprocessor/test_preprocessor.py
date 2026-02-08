# Copyright (c) ModelScope Contributors. All rights reserved.
"""
测试 Preprocessor 功能：
1. CompetitionMathProcessor - 处理数学问题数据
2. CompetitionMathGRPOProcessor - 处理数学问题数据（GRPO格式）
3. SelfCognitionProcessor - 处理自我认知数据（带占位符）
4. AlpacaProcessor - 处理 Alpaca 格式数据（多种情况）
5. Dataset.map 改动点测试（自动过滤 None、batched=False）
"""
import os
import pytest
from pathlib import Path

from twinkle.dataset import Dataset, DatasetMeta
from twinkle.preprocessor import (
    CompetitionMathProcessor,
    CompetitionMathGRPOProcessor,
    SelfCognitionProcessor,
    AlpacaProcessor
)
from twinkle.data_format import Trajectory, Message

# 获取测试数据目录
TEST_DATA_DIR = Path(__file__).parent / "test_data"


class TestCompetitionMathProcessor:
    """测试 CompetitionMathProcessor"""

    def test_process_math_data(self):
        """测试处理数学问题数据"""
        jsonl_path = str(TEST_DATA_DIR / "math_data.jsonl")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        dataset.map(CompetitionMathProcessor())
        
        assert len(dataset) == 4
        
        # 检查第一个样本
        sample = dataset[0]
        assert 'messages' in sample
        messages = sample['messages']
        assert len(messages) == 2
        assert messages[0]['role'] == 'user'
        assert messages[0]['content'] == "What is 2+2?"
        assert messages[1]['role'] == 'assistant'
        assert messages[1]['content'] == "The answer is 4."
        
        # 检查无 system message
        assert all(msg['role'] != 'system' for msg in messages)

    def test_process_all_samples(self):
        """测试处理所有样本"""
        jsonl_path = str(TEST_DATA_DIR / "math_data.jsonl")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        dataset.map(CompetitionMathProcessor())
        
        # 验证所有样本都有正确的结构
        for i in range(len(dataset)):
            sample = dataset[i]
            assert 'messages' in sample
            messages = sample['messages']
            assert len(messages) == 2
            assert messages[0]['role'] == 'user'
            assert messages[1]['role'] == 'assistant'


class TestCompetitionMathGRPOProcessor:
    """测试 CompetitionMathGRPOProcessor"""

    def test_process_grpo_data(self):
        """测试处理 GRPO 格式数据"""
        jsonl_path = str(TEST_DATA_DIR / "math_data.jsonl")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        dataset.map(CompetitionMathGRPOProcessor())
        
        assert len(dataset) == 4
        
        # 检查第一个样本
        sample = dataset[0]
        assert 'messages' in sample
        messages = sample['messages']
        assert len(messages) == 3
        
        # 检查 system message
        assert messages[0]['role'] == 'system'
        assert 'math assistant' in messages[0]['content'].lower()
        
        # 检查 user message
        assert messages[1]['role'] == 'user'
        assert messages[1]['content'] == "What is 2+2?"
        
        # 检查 assistant message（应该为空）
        assert messages[2]['role'] == 'assistant'
        assert messages[2]['content'] == ''
        
        # 检查 user_data
        assert 'user_data' in sample
        user_data = sample['user_data']
        assert len(user_data) == 1
        assert user_data[0][0] == 'solution'
        assert user_data[0][1] == "The answer is 4."

    def test_user_data_storage(self):
        """测试 user_data 存储"""
        jsonl_path = str(TEST_DATA_DIR / "math_data.jsonl")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        dataset.map(CompetitionMathGRPOProcessor())
        
        # 验证所有样本都有 user_data
        for i in range(len(dataset)):
            sample = dataset[i]
            assert 'user_data' in sample
            user_data = sample['user_data']
            assert len(user_data) == 1
            assert user_data[0][0] == 'solution'


class TestSelfCognitionProcessor:
    """测试 SelfCognitionProcessor"""

    def test_process_self_cognition_data(self):
        """测试处理自我认知数据"""
        jsonl_path = str(TEST_DATA_DIR / "self_cognition_data.jsonl")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        dataset.map(SelfCognitionProcessor('twinkle模型', 'twinkle团队'))
        
        assert len(dataset) == 3
        
        # 检查第一个样本
        sample = dataset[0]
        assert 'messages' in sample
        messages = sample['messages']
        assert len(messages) == 3
        
        # 检查 system message
        assert messages[0]['role'] == 'system'
        assert messages[0]['content'] == 'You are a helpful assistant.'
        
        # 检查 user message（占位符应被替换）
        assert messages[1]['role'] == 'user'
        assert messages[1]['content'] == "What is twinkle模型?"
        assert '{{NAME}}' not in messages[1]['content']
        assert '{{AUTHOR}}' not in messages[1]['content']
        
        # 检查 assistant message（占位符应被替换）
        assert messages[2]['role'] == 'assistant'
        assert messages[2]['content'] == "twinkle模型 is a language model developed by twinkle团队."
        assert '{{NAME}}' not in messages[2]['content']
        assert '{{AUTHOR}}' not in messages[2]['content']

    def test_placeholder_replacement(self):
        """测试占位符替换功能"""
        jsonl_path = str(TEST_DATA_DIR / "self_cognition_data.jsonl")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        dataset.map(SelfCognitionProcessor('test_model', 'test_author'))
        
        # 验证所有样本的占位符都被替换
        for i in range(len(dataset)):
            sample = dataset[i]
            messages = sample['messages']
            for msg in messages:
                assert '{{NAME}}' not in msg['content']
                assert '{{AUTHOR}}' not in msg['content']
                if msg['role'] in ['user', 'assistant']:
                    assert 'test_model' in msg['content'] or 'test_author' in msg['content']


class TestAlpacaProcessor:
    """测试 AlpacaProcessor - 多种情况"""

    def test_alpaca_instruction_only(self):
        """测试只有 instruction 的情况"""
        jsonl_path = str(TEST_DATA_DIR / "alpaca_data.jsonl")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        dataset.map(AlpacaProcessor())
        
        # 找到只有 instruction 的样本（第4个样本）
        sample = dataset[3]  # "What is the capital of France?"
        messages = sample['messages']
        assert len(messages) == 2
        assert messages[0]['role'] == 'user'
        assert messages[0]['content'] == "What is the capital of France?"
        assert messages[1]['role'] == 'assistant'
        assert messages[1]['content'] == "The capital of France is Paris."

    def test_alpaca_instruction_with_input(self):
        """测试 instruction + input 的情况"""
        jsonl_path = str(TEST_DATA_DIR / "alpaca_data.jsonl")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        dataset.map(AlpacaProcessor())
        
        # 找到有 input 的样本（第2个样本）
        sample = dataset[1]  # "Translate the following text" + "Hello"
        messages = sample['messages']
        assert len(messages) == 2
        assert messages[0]['role'] == 'user'
        assert "Translate the following text" in messages[0]['content']
        assert "Hello" in messages[0]['content']
        assert '\n' in messages[0]['content']  # 应该包含换行符
        assert messages[1]['role'] == 'assistant'
        assert messages[1]['content'] == "你好"

    def test_alpaca_empty_input(self):
        """测试 input 为空字符串的情况"""
        jsonl_path = str(TEST_DATA_DIR / "alpaca_data.jsonl")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        dataset.map(AlpacaProcessor())
        
        # 找到 input 为空字符串的样本（第1个样本）
        sample = dataset[0]  # "Explain what AI is" with empty input
        messages = sample['messages']
        assert len(messages) == 2
        assert messages[0]['role'] == 'user'
        assert messages[0]['content'] == "Explain what AI is"
        assert '\n' not in messages[0]['content']

    def test_alpaca_missing_fields(self):
        """测试缺失字段的容错处理"""
        # 创建包含缺失字段的测试数据
        import json
        import tempfile
        
        test_data = [
            {"instruction": "Test", "output": "Result"}, 
            {"instruction": "Test2", "input": "Input2"}, 
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
            temp_path = f.name
        
        try:
            dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=temp_path))
            dataset.map(AlpacaProcessor())
            
            # 第一个样本应该正常处理（缺失 input）
            assert len(dataset) >= 1
            sample = dataset[0]
            messages = sample['messages']
            assert messages[0]['content'] == "Test"
        finally:
            os.unlink(temp_path)

    def test_alpaca_all_samples(self):
        """测试处理所有 Alpaca 格式样本"""
        jsonl_path = str(TEST_DATA_DIR / "alpaca_data.jsonl")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        dataset.map(AlpacaProcessor())
        
        # 验证所有样本都有正确的结构
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
    """测试 Dataset.map 的改动点"""

    def test_auto_filter_none(self):
        """测试自动过滤 None 值"""
        import json
        import tempfile
        
        # 注意：不能对第一个样本返回 None，因为 datasets 库在第一个样本返回 None 时会认为不需要更新数据
        class NoneProcessor(CompetitionMathProcessor):
            def __call__(self, row):
                # 对第二个样本返回 None（不是第一个）
                if row['problem'] == "Solve for x: 3x + 5 = 14":
                    return None
                return super().__call__(row)
        
        jsonl_path = str(TEST_DATA_DIR / "math_data.jsonl")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        original_len = len(dataset)
        assert original_len == 4
        
        dataset.map(NoneProcessor())
        
        # 应该过滤掉返回 None 的样本
        assert len(dataset) < original_len
        assert len(dataset) == 3  # 4个样本，1个返回None，剩下3个
        
        # 验证没有 None 值，且所有样本都有正确的结构
        for i in range(len(dataset)):
            sample = dataset[i]
            assert sample is not None
            assert 'messages' in sample
            messages = sample['messages']
            assert messages[0]['content'] != "Solve for x: 3x + 5 = 14"

    def test_batched_false(self):
        """测试 batched=False 的设置"""
        jsonl_path = str(TEST_DATA_DIR / "math_data.jsonl")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        
        # 验证 map 方法会设置 batched=False
        dataset.map(CompetitionMathProcessor())
        
        # 这里验证处理结果是正确的（单样本处理）
        assert len(dataset) == 4
        for i in range(len(dataset)):
            sample = dataset[i]
            assert 'messages' in sample
            # 每个样本应该有独立的 messages
            assert isinstance(sample['messages'], list)

    def test_load_from_cache_file_false(self):
        """测试 load_from_cache_file=False 的默认设置"""
        jsonl_path = str(TEST_DATA_DIR / "math_data.jsonl")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        
        # 多次调用 map，不应该使用缓存
        dataset.map(CompetitionMathProcessor())
        first_result = dataset[0]['messages'][0]['content']
        
        # 修改 processor，再次处理
        class ModifiedProcessor(CompetitionMathProcessor):
            def __call__(self, row):
                traj = super().__call__(row)
                traj['messages'][0]['content'] = "Modified: " + traj['messages'][0]['content']
                return traj
        
        dataset2 = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        dataset2.map(ModifiedProcessor())
        second_result = dataset2[0]['messages'][0]['content']
        
        assert first_result != second_result
        assert "Modified: " in second_result

    def test_processor_string_name(self):
        """测试使用字符串名称加载 processor"""
        jsonl_path = str(TEST_DATA_DIR / "math_data.jsonl")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        
        dataset.map('CompetitionMathProcessor')
        
        assert len(dataset) == 4
        sample = dataset[0]
        assert 'messages' in sample

    def test_processor_with_init_args(self):
        """测试使用 init_args 初始化 processor"""
        jsonl_path = str(TEST_DATA_DIR / "self_cognition_data.jsonl")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        
        dataset.map('SelfCognitionProcessor', init_args={
            'model_name': 'test_model',
            'model_author': 'test_author'
        })
        
        assert len(dataset) == 3
        sample = dataset[0]
        messages = sample['messages']
        assert 'test_model' in messages[1]['content'] or 'test_author' in messages[1]['content']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
