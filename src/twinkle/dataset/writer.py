# Copyright (c) ModelScope Contributors. All rights reserved.
import json as _json
import os.path
import threading
from queue import Queue
from typing import Any, Dict, Iterable, List

from twinkle.utils.parallel import PosixFileLock


class DatasetWriter:
    """Row serialization for ``Dataset.save_as``.

    ``Dataset`` owns the decision of WHAT to write (which rows, bulk vs incremental); this class
    owns HOW to write it. Kept out of the dataset class because export formats change for reasons
    that have nothing to do with dataset loading, transformation or remote execution.
    """

    _EXT_TO_FORMAT = {'jsonl': 'jsonl', 'json': 'jsonl', 'csv': 'csv', 'parquet': 'parquet', 'pq': 'parquet'}
    SUPPORTED_FORMATS = ('jsonl', 'json', 'csv', 'parquet')

    @staticmethod
    def infer_format(path: str) -> str:
        ext = os.path.splitext(path)[1].lstrip('.').lower()
        return DatasetWriter._EXT_TO_FORMAT.get(ext, 'jsonl')

    @staticmethod
    def default_serializer(obj: Any) -> Any:
        """Handle numpy types in JSON serialization."""
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')

    @staticmethod
    def write_bulk(dataset, path: str, fmt: str, **kwargs) -> None:
        """Export via the HF dataset's own writers (no per-row Python loop)."""
        if fmt in ('jsonl', 'json'):
            dataset.to_json(path, **kwargs)
        elif fmt == 'csv':
            dataset.to_csv(path, **kwargs)
        elif fmt == 'parquet':
            dataset.to_parquet(path, **kwargs)

    @classmethod
    def write_incremental(cls, iterator: Iterable[Dict], path: str, fmt: str, batch_size: int) -> None:
        """Export row by row, for datasets whose rows only exist once materialized."""
        if fmt in ('jsonl', 'json'):
            cls._write_jsonl(path, iterator)
        elif fmt == 'csv':
            cls._write_csv(path, iterator, batch_size)
        elif fmt == 'parquet':
            cls._write_parquet(path, iterator, batch_size)

    @classmethod
    def _write_jsonl(cls, path: str, iterator: Iterable[Dict]) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            for row in iterator:
                f.write(_json.dumps(row, ensure_ascii=False, default=cls.default_serializer) + '\n')

    @staticmethod
    def _write_csv(path: str, iterator: Iterable[Dict], batch_size: int) -> None:
        import pandas as pd
        first = True
        batch: List[Dict] = []
        for row in iterator:
            batch.append(row)
            if len(batch) >= batch_size:
                pd.DataFrame(batch).to_csv(path, mode='a', header=first, index=False)
                first = False
                batch = []
        if batch:
            pd.DataFrame(batch).to_csv(path, mode='a', header=first, index=False)

    @staticmethod
    def _write_parquet(path: str, iterator: Iterable[Dict], batch_size: int) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        writer = None
        batch: List[Dict] = []
        for row in iterator:
            batch.append(row)
            if len(batch) >= batch_size:
                table = pa.Table.from_pylist(batch)
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema)
                writer.write_table(table)
                batch = []
        if batch:
            table = pa.Table.from_pylist(batch)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
        if writer:
            writer.close()


class AsyncRowWriter:
    """Write-through writer for ``save_as(mode='training')``.

    Writes happen on a background daemon thread so the training loop is never blocked.
    Uses fcntl file-lock for cross-process safety when multiple ranks write one file.
    """

    _SENTINEL = object()

    def __init__(self, path: str, fmt: str, batch_size: int):

        self._path = path
        self._fmt = fmt
        self._batch_size = batch_size
        self._queue: Queue = Queue(maxsize=batch_size * 4)
        self._lock = PosixFileLock(path + '.lock')
        self._error = None

        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def write(self, row: Dict) -> None:
        self._queue.put(row)

    def close(self) -> None:
        self._queue.put(self._SENTINEL)
        self._thread.join()
        self._lock.close()
        if self._error:
            raise self._error

    def _writer_loop(self) -> None:
        try:
            if self._fmt in ('jsonl', 'json'):
                self._loop_jsonl()
            elif self._fmt == 'csv':
                self._loop_csv()
            elif self._fmt == 'parquet':
                self._loop_parquet()
        except Exception as e:
            self._error = e

    def _acquire_lock(self):
        self._lock.acquire()

    def _release_lock(self):
        self._lock.release()

    def _loop_jsonl(self) -> None:
        buffer: List[str] = []

        def _flush(f):
            if not buffer:
                return
            payload = ''.join(buffer)
            self._acquire_lock()
            try:
                f.write(payload)
                f.flush()
            finally:
                self._release_lock()
            buffer.clear()

        with open(self._path, 'a', encoding='utf-8') as f:
            while True:
                item = self._queue.get()
                if item is self._SENTINEL:
                    _flush(f)
                    return
                buffer.append(_json.dumps(item, ensure_ascii=False, default=DatasetWriter.default_serializer) + '\n')
                if len(buffer) >= self._batch_size:
                    _flush(f)

    def _loop_csv(self) -> None:
        import pandas as pd
        header_written = False
        buffer: List[Dict] = []
        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                if buffer:
                    self._acquire_lock()
                    try:
                        pd.DataFrame(buffer).to_csv(self._path, mode='a', header=not header_written, index=False)
                    finally:
                        self._release_lock()
                return
            buffer.append(item)
            if len(buffer) >= self._batch_size:
                self._acquire_lock()
                try:
                    pd.DataFrame(buffer).to_csv(self._path, mode='a', header=not header_written, index=False)
                    header_written = True
                finally:
                    self._release_lock()
                buffer = []

    def _loop_parquet(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        writer = None
        buffer: List[Dict] = []
        try:
            while True:
                item = self._queue.get()
                if item is self._SENTINEL:
                    if buffer:
                        table = pa.Table.from_pylist(buffer)
                        if writer is None:
                            writer = pq.ParquetWriter(self._path, table.schema)
                        self._acquire_lock()
                        try:
                            writer.write_table(table)
                        finally:
                            self._release_lock()
                    return
                buffer.append(item)
                if len(buffer) >= self._batch_size:
                    table = pa.Table.from_pylist(buffer)
                    if writer is None:
                        writer = pq.ParquetWriter(self._path, table.schema)
                    self._acquire_lock()
                    try:
                        writer.write_table(table)
                    finally:
                        self._release_lock()
                    buffer = []
        finally:
            if writer:
                writer.close()
