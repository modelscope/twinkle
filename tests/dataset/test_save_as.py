# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Test Dataset.save_as:
1. Immediate mode (bulk/incremental) across formats (jsonl, csv, parquet)
2. Training mode (write-through) across formats (jsonl, csv, parquet)
"""
import json
import os
import pytest
import tempfile

from twinkle.dataset import Dataset, DatasetMeta

SAMPLE_DATA = [
    {
        'text': 'Hello world',
        'label': 0
    },
    {
        'text': 'Test data',
        'label': 1
    },
    {
        'text': 'Another example',
        'label': 0
    },
    {
        'text': 'Sample text',
        'label': 1
    },
]


def _make_dataset(data=None, streaming=False):
    """Create a Dataset from in-memory data."""
    d = data or SAMPLE_DATA
    if streaming:

        def gen():
            yield from d

        return Dataset(dataset_meta=DatasetMeta(data=gen), streaming=True)
    return Dataset(dataset_meta=DatasetMeta(data=d))


class TestSaveAsImmediate:
    """Immediate mode: save the entire dataset at once."""

    def test_save_jsonl(self, tmp_path):
        ds = _make_dataset()
        out = str(tmp_path / 'output.jsonl')
        ds.save_as(out)

        with open(out) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 4
        assert lines[0]['text'] == 'Hello world'
        assert lines[3]['label'] == 1

    def test_save_csv(self, tmp_path):
        import pandas as pd
        ds = _make_dataset()
        out = str(tmp_path / 'output.csv')
        ds.save_as(out, format='csv')

        df = pd.read_csv(out)
        assert len(df) == 4
        assert df.iloc[0]['text'] == 'Hello world'
        assert df.iloc[1]['label'] == 1

    def test_save_parquet(self, tmp_path):
        import pandas as pd
        ds = _make_dataset()
        out = str(tmp_path / 'output.parquet')
        ds.save_as(out)

        df = pd.read_parquet(out)
        assert len(df) == 4
        assert df.iloc[0]['text'] == 'Hello world'
        assert df.iloc[2]['label'] == 0

    def test_save_json_extension_inferred_as_jsonl(self, tmp_path):
        ds = _make_dataset()
        out = str(tmp_path / 'output.json')
        ds.save_as(out)

        with open(out) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 4

    def test_save_incremental_streaming(self, tmp_path):
        """Streaming (IterableDataset) triggers incremental save path."""
        ds = _make_dataset(streaming=True)
        out = str(tmp_path / 'stream_out.jsonl')
        ds.save_as(out)

        with open(out) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 4
        assert lines[0]['text'] == 'Hello world'

    def test_save_incremental_csv_streaming(self, tmp_path):
        import pandas as pd
        ds = _make_dataset(streaming=True)
        out = str(tmp_path / 'stream_out.csv')
        ds.save_as(out, format='csv', batch_size=2)

        df = pd.read_csv(out)
        assert len(df) == 4

    def test_save_incremental_parquet_streaming(self, tmp_path):
        import pandas as pd
        ds = _make_dataset(streaming=True)
        out = str(tmp_path / 'stream_out.parquet')
        ds.save_as(out, format='parquet', batch_size=2)

        df = pd.read_parquet(out)
        assert len(df) == 4

    def test_error_no_dataset(self, tmp_path):
        ds = Dataset()
        with pytest.raises(ValueError, match='No dataset to save'):
            ds.save_as(str(tmp_path / 'x.jsonl'))

    def test_error_unsupported_format(self, tmp_path):
        ds = _make_dataset()
        with pytest.raises(ValueError, match='Unsupported format'):
            ds.save_as(str(tmp_path / 'x.txt'), format='txt')

    def test_creates_output_directory(self, tmp_path):
        ds = _make_dataset()
        out = str(tmp_path / 'nested' / 'dir' / 'output.jsonl')
        ds.save_as(out)
        assert os.path.isfile(out)


class TestSaveAsTraining:
    """Training mode: write-through as items are consumed via __getitem__."""

    def test_training_jsonl(self, tmp_path):
        ds = _make_dataset()
        out = str(tmp_path / 'train.jsonl')
        ds.save_as(out, mode='training')

        # Consume all items via __getitem__
        for i in range(len(ds)):
            _ = ds[i]

        ds.flush_save()

        with open(out) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 4
        assert lines[0]['text'] == 'Hello world'
        assert lines[3]['label'] == 1

    def test_training_csv(self, tmp_path):
        import pandas as pd
        ds = _make_dataset()
        out = str(tmp_path / 'train.csv')
        ds.save_as(out, format='csv', batch_size=2, mode='training')

        for i in range(len(ds)):
            _ = ds[i]

        ds.flush_save()

        df = pd.read_csv(out)
        assert len(df) == 4
        assert df.iloc[0]['text'] == 'Hello world'

    def test_training_parquet(self, tmp_path):
        import pandas as pd
        ds = _make_dataset()
        out = str(tmp_path / 'train.parquet')
        ds.save_as(out, format='parquet', batch_size=2, mode='training')

        for i in range(len(ds)):
            _ = ds[i]

        ds.flush_save()

        df = pd.read_parquet(out)
        assert len(df) == 4
        assert df.iloc[2]['text'] == 'Another example'

    def test_training_partial_consume(self, tmp_path):
        """Only consumed items are written."""
        ds = _make_dataset()
        out = str(tmp_path / 'partial.jsonl')
        ds.save_as(out, mode='training')

        # Only consume first 2 items
        _ = ds[0]
        _ = ds[1]

        ds.flush_save()

        with open(out) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 2
        assert lines[0]['text'] == 'Hello world'
        assert lines[1]['text'] == 'Test data'

    def test_training_flush_idempotent(self, tmp_path):
        """Double flush_save should not raise."""
        ds = _make_dataset()
        out = str(tmp_path / 'idem.jsonl')
        ds.save_as(out, mode='training')
        _ = ds[0]
        ds.flush_save()
        ds.flush_save()  # second call is a no-op

    def test_lock_file_cleanup(self, tmp_path):
        """Lock file is created during training mode."""
        ds = _make_dataset()
        out = str(tmp_path / 'lock_test.jsonl')
        ds.save_as(out, mode='training')
        _ = ds[0]
        ds.flush_save()
        # Lock file should exist (created by PosixFileLock)
        assert os.path.isfile(out + '.lock')
