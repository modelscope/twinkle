"""QualityPreprocessor + HF batched map must remove fully-dropped batches."""

from datasets import Dataset

from twinkle_agentic.preprocessor import QualityPreprocessor
from twinkle_agentic.preprocessor.model_filter import ModelFilter


def test_fully_dropped_batch_does_not_leave_ghost_rows():
    """Returning ``{}`` from an empty batch used to keep raw rows; use empty column lists."""
    qp = QualityPreprocessor(pipeline=[ModelFilter()], dropped_log_path='')
    batch = {
        'id': ['bad1', 'bad2'],
        'model_id': ['Qwen/Qwen3.5-27B', 'Qwen/Qwen3-VL-8B-Instruct'],
        'messages': [[], []],
        'user_data': [[], []],
    }
    out = qp(batch)
    assert out == {
        'id': [],
        'model_id': [],
        'messages': [],
        'user_data': [],
    }

    ds = Dataset.from_dict({
        'id': ['bad1', 'keep', 'bad2'],
        'model_id': [
            'Qwen/Qwen3.5-27B',
            'MiniMax/MiniMax-M2.5',
            'Qwen/Qwen3-VL-8B-Instruct',
        ],
        'messages': [[], [{'role': 'user', 'content': 'hi'}], []],
        'user_data': [[], [], []],
    })
    mapped = ds.map(qp, batched=True, batch_size=3)
    assert len(mapped) == 1
    assert mapped[0]['model_id'] == 'MiniMax/MiniMax-M2.5'


class _AddTag:
    """Mapper: add a top-level `tag` column to every row (never drops)."""

    def __call__(self, rows):
        rows = QualityPreprocessor.map_col_to_row(rows)
        return [dict(r, tag='T') for r in rows], []


class _DropOdd:
    """Filter: drop rows whose `id` ends in an odd digit."""

    def __call__(self, rows):
        rows = QualityPreprocessor.map_col_to_row(rows)
        kept, dropped = [], []
        for r in rows:
            (dropped if int(str(r['id'])[-1]) % 2 else kept).append(
                dict(r, drop_reason='odd') if int(str(r['id'])[-1]) % 2 else r)
        return kept, dropped


def test_mark_mode_returns_equal_length_columns():
    """drop_mode='mark' must never change row count inside map (ghost-proof)."""
    qp = QualityPreprocessor(pipeline=[_AddTag(), _DropOdd()], drop_mode='mark')
    batch = {
        'id': ['r0', 'r1', 'r2', 'r3'],
        'messages': [[], [], [], []],
        'user_data': [[], [], [], []],
    }
    out = qp(batch)
    # every column has the SAME length as the input (4), no shrinkage
    lengths = {k: len(v) for k, v in out.items()}
    assert set(lengths.values()) == {4}, lengths
    # the survivor-only tag column exists for all rows (None for dropped)
    assert '_keep' in out and 'tag' in out
    assert out['_keep'] == [True, False, True, False]     # r0,r2 kept; r1,r3 dropped
    assert out['tag'] == ['T', None, 'T', None]           # dropped rows have no tag


def test_mark_mode_end_to_end_filter():
    """map(mark) + filter(_keep) yields the correct survivors, no ghosts, at scale."""
    from twinkle_agentic.preprocessor import run_quality_pipeline

    class _DS:
        def __init__(self, hf):
            self.dataset = hf
            self.datasets = {'d': hf}

        def map(self, fn, num_proc=1, **kw):
            self.dataset = self.dataset.map(fn, batched=True, num_proc=num_proc, **kw)
            self.datasets['d'] = self.dataset

        def filter(self, fn, **kw):
            self.dataset = self.dataset.filter(fn, **kw)
            self.datasets['d'] = self.dataset

    n = 500  # large enough to cross HF's internal batch boundary (the ghost trigger)
    hf = Dataset.from_dict({
        'id': [f'r{i}' for i in range(n)],
        'messages': [[{'role': 'user', 'content': 'x'}] for _ in range(n)],
        'user_data': [[] for _ in range(n)],
    })
    ds = _DS(hf)
    qp = QualityPreprocessor(pipeline=[_AddTag(), _DropOdd()], drop_mode='mark')
    run_quality_pipeline(ds, qp, num_proc=1)

    survivors = ds.dataset
    assert len(survivors) == n // 2                      # exactly the even-id rows
    assert '_keep' not in survivors.column_names         # transient flag stripped
    assert all(int(str(survivors[i]['id'])[-1]) % 2 == 0 for i in range(len(survivors)))
    assert all(survivors[i]['tag'] == 'T' for i in range(len(survivors)))  # tags intact
