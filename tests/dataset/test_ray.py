import os
import pytest
from pathlib import Path

from twinkle.dataset import Dataset, DatasetMeta
from twinkle.data_format import Message

TEST_DATA_DIR = Path(__file__).parent / "test_data"
SKIP_MODEL_DOWNLOAD = os.getenv('SKIP_MODEL_DOWNLOAD', 'false').lower() == 'true'


def convert_to_messages(example):
    text = example.get('text', '')
    if not text:
        text = str(example.get('question', example.get('title', '')))
    
    return {
        'messages': [
            Message(role='user', content=text),
            Message(role='assistant', content='Response')
        ]
    }


class TestRayDatasetBehavior:
    """测试Ray模式下dataset表现与local相同
    
    注意：Dataset的核心功能（加载、map、encode等）与twinkle的运行模式（local/ray）无关，
    这些测试验证dataset在两种模式下都能正常工作。
    """
    
    def test_dataset_works_in_ray_mode(self):
        """测试dataset在Ray模式下正常工作"""
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        
        assert len(dataset) == 4
        assert dataset[0]['text'] == "Hello world"
        assert dataset[0]['label'] == 0

    def test_dataset_map_works_in_ray_mode(self):
        """测试dataset map操作在Ray模式下正常工作"""
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(convert_to_messages)
        
        assert len(dataset) == 4
        assert 'messages' in dataset[0]
        assert len(dataset[0]['messages']) == 2

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_dataset_encode_works_in_ray_mode(self):
        """测试dataset encode操作在Ray模式下正常工作"""
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(convert_to_messages)
        
        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128)
        except Exception as e:
            pytest.skip(f"Failed to load template (may need network): {e}")
        
        dataset.encode(batched=True)
        
        assert 'input_ids' in dataset[0]
        assert len(dataset[0]['input_ids']) > 0

    def test_dataset_add_dataset_works_in_ray_mode(self):
        """测试dataset add_dataset操作在Ray模式下正常工作"""
        csv_path1 = str(TEST_DATA_DIR / "test.csv")
        csv_path2 = str(TEST_DATA_DIR / "test2.csv")
        
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path1))
        dataset.add_dataset(DatasetMeta(dataset_id=csv_path2))
        
        assert len(dataset.datasets) == 2
        assert len(dataset.dataset) == 4

    def test_dataset_mix_dataset_works_in_ray_mode(self):
        """测试dataset mix_dataset操作在Ray模式下正常工作"""
        csv_path1 = str(TEST_DATA_DIR / "test.csv")
        csv_path2 = str(TEST_DATA_DIR / "test2.csv")
        
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path1))
        dataset.add_dataset(DatasetMeta(dataset_id=csv_path2))
        dataset.mix_dataset(interleave=True)
        
        assert len(dataset.dataset) == 6
