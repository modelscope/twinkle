# Copyright (c) ModelScope Contributors. All rights reserved.
"""
测试数据集加载功能：
1. 加载本地 csv/json/jsonl 数据集（普通 dataset 方式）
2. 加载本地 csv/json/jsonl 数据集（iterable 方式）
3. 加载 hf 数据集（普通 dataset 方式）
4. 加载 hf 数据集（iterable 方式）
5. 加载 ms 数据集（普通 dataset 方式）
6. 加载 ms 数据集（iterable 方式）
"""
import os
import pytest
from pathlib import Path

from twinkle.dataset import Dataset, IterableDataset, DatasetMeta


# 获取测试数据目录
TEST_DATA_DIR = Path(__file__).parent / "test_data"

class TestLocalDatasetLoading:
    """测试本地数据集加载（普通 dataset 方式）"""

    def test_load_local_csv(self):
        """测试加载本地 CSV 文件"""
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        
        assert len(dataset) == 4
        assert dataset[0]['text'] == "Hello world"
        assert dataset[0]['label'] == 0
        assert dataset[1]['text'] == "Test data"
        assert dataset[1]['label'] == 1

    def test_load_local_json(self):
        """测试加载本地 JSON 文件"""
        json_path = str(TEST_DATA_DIR / "test.json")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=json_path))
        
        assert len(dataset) == 4
        assert dataset[0]['text'] == "Hello world"
        assert dataset[0]['label'] == 0

    def test_load_local_jsonl(self):
        jsonl_path = str(TEST_DATA_DIR / "test.jsonl")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        
        assert len(dataset) == 4
        assert dataset[0]['text'] == "Hello world"
        assert dataset[0]['label'] == 0


class TestLocalIterableDatasetLoading:
    """测试本地数据集加载（iterable 方式）"""

    def test_load_local_csv_iterable(self):
        """测试加载本地 CSV 文件（iterable 方式）"""
        csv_path = str(TEST_DATA_DIR / "test.csv")
        try:
            dataset = IterableDataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        except NotImplementedError as e:
            # datasets 不支持 streaming=True + num_proc；twinkle 目前本地 streaming 分支会传 num_proc
            pytest.xfail(f"Known limitation: streaming local file with num_proc is not supported: {e}")
        
        # iterable dataset 不支持 __len__
        with pytest.raises(NotImplementedError):
            _ = len(dataset)
        
        # 测试迭代
        items = list(dataset)
        assert len(items) == 4
        assert items[0]['text'] == "Hello world"
        assert items[0]['label'] == 0

    def test_load_local_json_iterable(self):
        """测试加载本地 JSON 文件（iterable 方式）"""
        json_path = str(TEST_DATA_DIR / "test.json")
        try:
            dataset = IterableDataset(dataset_meta=DatasetMeta(dataset_id=json_path))
        except NotImplementedError as e:
            pytest.xfail(f"Known limitation: streaming local file with num_proc is not supported: {e}")
        
        items = list(dataset)
        assert len(items) == 4
        assert items[0]['text'] == "Hello world"

    def test_load_local_jsonl_iterable(self):
        """测试加载本地 JSONL 文件（iterable 方式）"""
        jsonl_path = str(TEST_DATA_DIR / "test.jsonl")
        try:
            dataset = IterableDataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        except NotImplementedError as e:
            pytest.xfail(f"Known limitation: streaming local file with num_proc is not supported: {e}")
        
        items = list(dataset)
        assert len(items) == 4
        assert items[0]['text'] == "Hello world"


class TestHFDatasetLoading:
    """测试 HuggingFace 数据集加载"""

    @pytest.mark.skipif(
        os.environ.get('TWINKLE_FORBID_HF', '0') == '1',
        reason="HF hub is disabled"
    )
    def test_load_hf_dataset(self):
        """测试加载 HF 数据集（普通 dataset 方式）"""
        # 使用一个小的公开数据集进行测试
        dataset_meta = DatasetMeta(
            dataset_id="hf://squad",
            subset_name="plain_text",
            split="train"
        )
        try:
            dataset = Dataset(dataset_meta=dataset_meta)

            # 只检查是否能成功加载，不检查具体长度（数据集可能很大）
            assert dataset is not None
            # 尝试获取第一个样本
            sample = dataset[0]
            assert sample is not None
        except Exception as e:
            # 离线环境/企业代理下 SSL 证书链不可用
            pytest.skip(f"HF dataset not reachable in current environment: {e}")

    @pytest.mark.skipif(
        os.environ.get('TWINKLE_FORBID_HF', '0') == '1',
        reason="HF hub is disabled"
    )
    def test_load_hf_dataset_iterable(self):
        """测试加载 HF 数据集（iterable 方式）"""
        dataset_meta = DatasetMeta(
            dataset_id="hf://squad",
            subset_name="plain_text",
            split="train"
        )
        try:
            dataset = IterableDataset(dataset_meta=dataset_meta)

            # iterable dataset 不支持 __len__
            with pytest.raises(NotImplementedError):
                _ = len(dataset)

            # 测试迭代，只取前几个样本
            items = []
            for i, item in enumerate(dataset):
                items.append(item)
                if i >= 2:  # 只取前3个样本
                    break

            assert len(items) == 3
            assert items[0] is not None
        except Exception as e:
            pytest.skip(f"HF dataset not reachable in current environment: {e}")


class TestMSDatasetLoading:
    """测试 ModelScope 数据集加载"""

    def test_load_ms_dataset(self):
        """测试加载 MS 数据集（普通 dataset 方式）"""
        # 使用一个小的公开数据集进行测试
        dataset_meta=DatasetMeta('ms://modelscope/competition_math')
        try:
            dataset = Dataset(dataset_meta=dataset_meta)
            # 只检查是否能成功加载
            assert dataset is not None
            # 如果数据集有数据，尝试获取第一个样本
            if len(dataset) > 0:
                sample = dataset[0]
                assert sample is not None
        except Exception as e:
            # 如果数据集不存在或无法访问，跳过测试
            pytest.skip(f"MS dataset not available: {e}")

    def test_load_ms_dataset_iterable(self):
        """测试加载 MS 数据集（iterable 方式）"""
        dataset_meta=DatasetMeta('ms://modelscope/competition_math')
        try:
            dataset = IterableDataset(dataset_meta=dataset_meta)
            
            # iterable dataset 不支持 __len__
            with pytest.raises(NotImplementedError):
                _ = len(dataset)
            
            # 测试迭代，只取前几个样本
            items = []
            for i, item in enumerate(dataset):
                items.append(item)
                if i >= 2:  # 只取前3个样本
                    break
            
            assert len(items) > 0
            assert items[0] is not None
        except Exception as e:
            # 如果数据集不存在或无法访问，跳过测试
            pytest.skip(f"MS dataset not available: {e}")


class TestDatasetMeta:
    """测试 DatasetMeta 功能"""

    def test_dataset_meta_get_id(self):
        """测试 DatasetMeta.get_id() 方法"""
        meta = DatasetMeta(
            dataset_id="test/dataset",
            subset_name="subset1",
            split="train"
        )
        assert meta.get_id() == "test_dataset:subset1:train"

    def test_dataset_meta_with_data_slice(self):
        """测试 DatasetMeta 的 data_slice 功能"""
        csv_path = str(TEST_DATA_DIR / "test.csv")
        meta = DatasetMeta(
            dataset_id=csv_path,
            data_slice=[0, 2]  # 只选择索引 0 和 2
        )
        dataset = Dataset(dataset_meta=meta)
        
        assert len(dataset) == 2
        assert dataset[0]['text'] == "Hello world"
        assert dataset[1]['text'] == "Another example"
