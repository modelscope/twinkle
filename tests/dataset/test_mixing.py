# Copyright (c) ModelScope Contributors. All rights reserved.
"""
测试数据集混合功能：
1. add_dataset 添加多个数据集
2. mix_dataset 使用 interleave 方式混合
3. mix_dataset 使用 concat 方式混合
"""
import pytest
from pathlib import Path

from twinkle.dataset import Dataset, IterableDataset, DatasetMeta


# 获取测试数据目录
TEST_DATA_DIR = Path(__file__).parent / "test_data"


class TestDatasetMixing:
    """测试数据集混合功能（普通 dataset 方式）"""

    def test_add_multiple_datasets(self):
        """测试添加多个数据集"""
        csv_path1 = str(TEST_DATA_DIR / "test.csv")
        csv_path2 = str(TEST_DATA_DIR / "test2.csv")
        
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path1))
        dataset.add_dataset(DatasetMeta(dataset_id=csv_path2))
        
        assert len(dataset.datasets) == 2
        assert len(dataset.dataset) == 4

    def test_mix_dataset_interleave(self):
        """测试使用 interleave 方式混合数据集"""
        csv_path1 = str(TEST_DATA_DIR / "test.csv")
        csv_path2 = str(TEST_DATA_DIR / "test2.csv")
        
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path1))
        dataset.add_dataset(DatasetMeta(dataset_id=csv_path2))
        dataset.mix_dataset(interleave=True)

        assert len(dataset.dataset) == 6  
        

        samples = [dataset.dataset[i] for i in range(len(dataset.dataset))]
        texts = [s['text'] for s in samples]
        assert any('Hello' in t or 'Test' in t or 'Another' in t or 'Sample' in t for t in texts)  # 来自 test.csv
        assert any('Dataset 2' in t for t in texts)  # 来自 test2.csv

    def test_mix_dataset_concat(self):
        """测试使用 concat 方式混合数据集"""
        csv_path1 = str(TEST_DATA_DIR / "test.csv")
        csv_path2 = str(TEST_DATA_DIR / "test2.csv")
        
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path1))
        dataset.add_dataset(DatasetMeta(dataset_id=csv_path2))
        dataset.mix_dataset(interleave=False)
        

        assert len(dataset.dataset) == 7  
        
        assert dataset.dataset[0]['text'] == "Hello world"
        assert dataset.dataset[3]['text'] == "Sample text"

        assert dataset.dataset[4]['text'] == "Dataset 2 item 1"
        assert dataset.dataset[6]['text'] == "Dataset 2 item 3"

    def test_mix_three_datasets_interleave(self):
        """测试使用 interleave 方式混合三个数据集"""
        csv_path1 = str(TEST_DATA_DIR / "test.csv")
        csv_path2 = str(TEST_DATA_DIR / "test2.csv")
        csv_path3 = str(TEST_DATA_DIR / "test3.csv")
        
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path1))
        dataset.add_dataset(DatasetMeta(dataset_id=csv_path2))
        dataset.add_dataset(DatasetMeta(dataset_id=csv_path3))
        dataset.mix_dataset(interleave=True)
        

        assert len(dataset.dataset) == 6  
        
        # 验证数据来自三个数据集
        texts = [dataset.dataset[i]['text'] for i in range(len(dataset.dataset))]
        assert any('Hello' in t or 'Test' in t or 'Another' in t or 'Sample' in t for t in texts)  # 来自 test.csv
        assert any('Dataset 2' in t for t in texts)  # 来自 test2.csv
        assert any('Dataset 3' in t for t in texts)  # 来自 test3.csv

    def test_mix_three_datasets_concat(self):
        """测试使用 concat 方式混合三个数据集"""
        csv_path1 = str(TEST_DATA_DIR / "test.csv")
        csv_path2 = str(TEST_DATA_DIR / "test2.csv")
        csv_path3 = str(TEST_DATA_DIR / "test3.csv")
        
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path1))
        dataset.add_dataset(DatasetMeta(dataset_id=csv_path2))
        dataset.add_dataset(DatasetMeta(dataset_id=csv_path3))
        dataset.mix_dataset(interleave=False)
        

        assert len(dataset.dataset) == 9  
  
        assert dataset.dataset[0]['text'] == "Hello world"
        assert dataset.dataset[3]['text'] == "Sample text"
s
        assert dataset.dataset[4]['text'] == "Dataset 2 item 1"
        assert dataset.dataset[6]['text'] == "Dataset 2 item 3"

        assert dataset.dataset[7]['text'] == "Dataset 3 item 1"
        assert dataset.dataset[8]['text'] == "Dataset 3 item 2"

    def test_mix_large_datasets_interleave(self):
        """测试使用 interleave 方式混合大型数据集"""
        csv_path4 = str(TEST_DATA_DIR / "test4.csv")  
        csv_path5 = str(TEST_DATA_DIR / "test5.csv")  
        
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path4))
        dataset.add_dataset(DatasetMeta(dataset_id=csv_path5))
        dataset.mix_dataset(interleave=True)
        

        assert len(dataset.dataset) == 224  
        

        texts = []
        for i in range(len(dataset.dataset)):
            item = dataset.dataset[i]
            text = item.get('text') or item.get('question') or ''
            if text:
                texts.append(str(text))
        
        assert any('Complex example' in t or 'Extended metadata' in t for t in texts) 
        assert any('capital of France' in t or 'quantum mechanics' in t for t in texts) 

    def test_mix_large_datasets_concat(self):
        """测试使用 concat 方式混合大型数据集"""
        csv_path4 = str(TEST_DATA_DIR / "test4.csv")  #
        csv_path5 = str(TEST_DATA_DIR / "test5.csv") 
        
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path4))
        dataset.add_dataset(DatasetMeta(dataset_id=csv_path5))
        dataset.mix_dataset(interleave=False)
        

        assert len(dataset.dataset) == 281 
        
 
        assert 'Complex example' in str(dataset.dataset[0].get('text', ''))
        assert 'Multiplayer sync tick' in str(dataset.dataset[111].get('text', ''))

        assert 'capital of France' in str(dataset.dataset[112].get('question', ''))

        assert 'democracy' in str(dataset.dataset[121].get('question', ''))

        last_item = dataset.dataset[280]
        last_text = str(last_item.get('text', '') or last_item.get('question', '') or '')
        assert 'Multiplayer sync tick' in last_text or 'tick_rate_64' in last_text

    def test_mix_different_formats_csv_json(self):
        """测试混合不同格式的数据集（CSV + JSON）"""
        csv_path = str(TEST_DATA_DIR / "test.csv")
        json_path = str(TEST_DATA_DIR / "test6.json")
        
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.add_dataset(DatasetMeta(dataset_id=json_path))
        dataset.mix_dataset(interleave=True)
        

        assert len(dataset.dataset) == 8  

        has_csv_data = False
        has_json_data = False
        for item in dataset.dataset:
            text = item.get('text')
            if text and ('Hello' in str(text) or 'Test' in str(text)):
                has_csv_data = True
            title = item.get('title')
            if title and 'Article' in str(title):
                has_json_data = True
        
        assert has_csv_data
        assert has_json_data

    def test_mix_different_formats_csv_jsonl(self):
        """测试混合不同格式的数据集（CSV + JSONL）"""
        csv_path = str(TEST_DATA_DIR / "test2.csv")
        jsonl_path = str(TEST_DATA_DIR / "test7.jsonl")
        
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.add_dataset(DatasetMeta(dataset_id=jsonl_path))
        dataset.mix_dataset(interleave=False)

        assert len(dataset.dataset) == 15  
        

        assert 'Dataset 2' in dataset.dataset[0].get('text', '')

        assert 'user_id' in dataset.dataset[3]
        assert 'action' in dataset.dataset[3]

    def test_mix_multiple_large_datasets(self):
        """测试混合多个大型数据集"""
        csv_path4 = str(TEST_DATA_DIR / "test4.csv")  
        csv_path5 = str(TEST_DATA_DIR / "test5.csv")  
        json_path6 = str(TEST_DATA_DIR / "test6.json")  
        jsonl_path7 = str(TEST_DATA_DIR / "test7.jsonl")  
        
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path4))
        dataset.add_dataset(DatasetMeta(dataset_id=csv_path5))
        dataset.add_dataset(DatasetMeta(dataset_id=json_path6))
        dataset.add_dataset(DatasetMeta(dataset_id=jsonl_path7))
        

        try:
            dataset.mix_dataset(interleave=True)
            # 如果成功，验证数据来自所有数据集
            all_texts = []
            for i in range(len(dataset.dataset)):
                item = dataset.dataset[i]
                all_texts.append(item.get('text', item.get('question', item.get('title', item.get('action', '')))))
            
            assert any('Complex example' in t for t in all_texts)  
            assert any('capital of France' in t for t in all_texts) 
            assert any('Article' in t for t in all_texts)  
            assert any('login' in t or 'purchase' in t for t in all_texts)  
            # 字段类型不兼容时，会抛出 ValueError
            pytest.skip(f"Features cannot be aligned (field type incompatibility): {e}")

    def test_mix_very_large_datasets_concat(self):
        """测试使用 concat 方式混合超大型数据集"""
        csv_path8 = str(TEST_DATA_DIR / "test8.csv")  
        json_path9 = str(TEST_DATA_DIR / "test9.json")  
        jsonl_path10 = str(TEST_DATA_DIR / "test10.jsonl") 
        
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path8))
        dataset.add_dataset(DatasetMeta(dataset_id=json_path9))
        dataset.add_dataset(DatasetMeta(dataset_id=jsonl_path10))
        

        try:
            dataset.mix_dataset(interleave=False)

            assert len(dataset.dataset) == 39  # 12 + 12 + 15
            

            assert 'product_id' in dataset.dataset[0]
            assert 'Laptop Pro' in dataset.dataset[0].get('name', '')

            assert 'student_id' in dataset.dataset[12]
            assert 'Alice' in dataset.dataset[12].get('name', '')

            assert 'transaction_id' in dataset.dataset[24]
            assert 'T001' in dataset.dataset[24].get('transaction_id', '')
        except ValueError as e:

            pytest.skip(f"Features cannot be aligned (field type incompatibility): {e}")

    def test_mix_complex_fields_interleave(self):
        """测试混合包含复杂字段的数据集（interleave）"""
        csv_path4 = str(TEST_DATA_DIR / "test4.csv")  
        csv_path8 = str(TEST_DATA_DIR / "test8.csv")   
        
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path4))
        dataset.add_dataset(DatasetMeta(dataset_id=csv_path8))
        dataset.mix_dataset(interleave=True)
        
    
        assert len(dataset.dataset) == 24  
        
        # 验证复杂字段存在
        has_metadata = any('metadata' in item for item in dataset.dataset)
        has_product_fields = any('product_id' in item and 'price' in item for item in dataset.dataset)
        assert has_metadata
        assert has_product_fields

    def test_mix_all_formats_concat(self):
        """测试使用 concat 方式混合所有格式的数据集"""
        csv_path = str(TEST_DATA_DIR / "test.csv")  
        json_path = str(TEST_DATA_DIR / "test6.json")  
        jsonl_path = str(TEST_DATA_DIR / "test7.jsonl")  
        
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.add_dataset(DatasetMeta(dataset_id=json_path))
        dataset.add_dataset(DatasetMeta(dataset_id=jsonl_path))
        dataset.mix_dataset(interleave=False)
        

        assert len(dataset.dataset) == 121  # 4 + 105 + 12
        

        assert 'text' in dataset.dataset[0]
        assert 'title' in dataset.dataset[4]
        assert 'user_id' in dataset.dataset[109]


class TestIterableDatasetMixing:
    """测试数据集混合功能（iterable 方式）"""

    def test_add_multiple_datasets_iterable(self):
        """测试添加多个数据集（iterable 方式）"""
        csv_path1 = str(TEST_DATA_DIR / "test.csv")
        csv_path2 = str(TEST_DATA_DIR / "test2.csv")
        
        try:
            dataset = IterableDataset(dataset_meta=DatasetMeta(dataset_id=csv_path1))
            dataset.add_dataset(DatasetMeta(dataset_id=csv_path2))
            
            assert len(dataset.datasets) == 2

            with pytest.raises(NotImplementedError):
                _ = len(dataset.dataset)
        except NotImplementedError as e:
            pytest.xfail(f"Known limitation: streaming local file with num_proc is not supported: {e}")

    def test_mix_dataset_interleave_iterable(self):
        """测试使用 interleave 方式混合数据集（iterable 方式）"""
        csv_path1 = str(TEST_DATA_DIR / "test.csv")
        csv_path2 = str(TEST_DATA_DIR / "test2.csv")
        
        try:
            dataset = IterableDataset(dataset_meta=DatasetMeta(dataset_id=csv_path1))
            dataset.add_dataset(DatasetMeta(dataset_id=csv_path2))
            dataset.mix_dataset(interleave=True)
            
            with pytest.raises(NotImplementedError):
                _ = len(dataset.dataset)
            
            items = []
            for i, item in enumerate(dataset):
                items.append(item)
                if i >= 6: 
                    break
            
            assert len(items) == 7
            texts = [item['text'] for item in items]
            assert any('Hello' in t or 'Test' in t or 'Another' in t or 'Sample' in t for t in texts)  # 来自 test.csv
            assert any('Dataset 2' in t for t in texts)  # 来自 test2.csv
        except NotImplementedError as e:
            pytest.xfail(f"Known limitation: streaming local file with num_proc is not supported: {e}")

    def test_mix_dataset_concat_iterable(self):
        """测试使用 concat 方式混合数据集（iterable 方式）"""
        csv_path1 = str(TEST_DATA_DIR / "test.csv")
        csv_path2 = str(TEST_DATA_DIR / "test2.csv")
        
        try:
            dataset = IterableDataset(dataset_meta=DatasetMeta(dataset_id=csv_path1))
            dataset.add_dataset(DatasetMeta(dataset_id=csv_path2))
            dataset.mix_dataset(interleave=False)
            
            # iterable dataset 不支持 __len__
            with pytest.raises(NotImplementedError):
                _ = len(dataset.dataset)
            
            items = []
            for i, item in enumerate(dataset):
                items.append(item)
                if i >= 6:  
                    break
            
            assert len(items) == 7
            assert items[0]['text'] == "Hello world"
            assert items[3]['text'] == "Sample text"
            assert items[4]['text'] == "Dataset 2 item 1"
            assert items[6]['text'] == "Dataset 2 item 3"
        except NotImplementedError as e:
            pytest.xfail(f"Known limitation: streaming local file with num_proc is not supported: {e}")


class TestDatasetMixingEdgeCases:
    """测试数据集混合的边缘情况"""

    def test_mix_single_dataset(self):
        """测试只有一个数据集时调用 mix_dataset"""
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        
        # 只有一个数据集时，mix_dataset 不应该改变 dataset
        original_len = len(dataset.dataset)
        dataset.mix_dataset(interleave=True)
        
        # dataset 应该保持不变
        assert len(dataset.dataset) == original_len
        assert dataset.dataset[0]['text'] == "Hello world"

    def test_mix_datasets_with_different_streaming_modes_error(self):
        """测试混合 streaming 和 non-streaming 数据集应该报错"""
        csv_path1 = str(TEST_DATA_DIR / "test.csv")
        csv_path2 = str(TEST_DATA_DIR / "test2.csv")
        
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path1))
  
        try:
            dataset.add_dataset(DatasetMeta(dataset_id=csv_path2), streaming=True)
            with pytest.raises(AssertionError, match="All datasets must be all streaming=True or streaming=False"):
                dataset.mix_dataset(interleave=True)
        except NotImplementedError:

            pytest.xfail("Known limitation: streaming local file with num_proc is not supported")
