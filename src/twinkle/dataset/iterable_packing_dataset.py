# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import multiprocessing as mp
from typing import TypeVar
import numpy as np
from twinkle.infra import remote_class, remote_function
from .base import DatasetMeta
from .iterable_dataset import IterableDataset
from .packing_dataset import PackingDataset

_T = TypeVar('_T')


@remote_class(execute='first')
class IterablePackingDataset(IterableDataset):

    def __init__(self, dataset_meta: DatasetMeta,
                 packing_interval: int = 128,
                 packing_num_proc: int = 1,
                 cyclic: bool = False, **kwargs):
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        self.packing_num_proc = packing_num_proc
        kwargs['streaming'] = True
        super().__init__(dataset_meta, **kwargs)
        self._out_queue = mp.Queue()
        self.packed_idx = []
        self.packed_length = []
        self.packing_interval = packing_interval
        self._in_queue = mp.Queue()
        self._out_queue = mp.Queue()
        self.workers = []
        self.cyclic = cyclic
    
    @remote_function()
    def set_template(self, template_cls, **kwargs):
        super().set_template(template_cls, **kwargs)
        assert self.template.truncation_strategy != 'split', 'Iterable packing does not support truncation_strategy==`split`'

    @remote_function()
    def pack_dataset(self):
        for _ in range(self.packing_num_proc):
            worker = mp.Process(target=self._processor, daemon=True)
            worker.start()
            self.workers.append(worker)

    def _processor(self):
        while True:
            i, data = self._in_queue.get()
            encoded_data = self.template.batch_encode([data])
            data.update(encoded_data[0])
            self._out_queue.put((i, data))

    def _put_data_in_queue(self, iterator) -> int:
        for i in range(self.packing_interval):
            try:
                data = next(iterator)
            except StopIteration:
                return i
            self._in_queue.put((i, data))
        return i + 1

    def _fetch_data_out_queue(self, last_res, num_samples):
        res = [None] * num_samples
        for _ in range(num_samples):
            i, data = self._out_queue.get()
            if not data:
                continue
            res[i] = (data, len(data['input_ids']))
        res = [data for data in res if data]
        last_res += res
        return last_res

    @staticmethod
    def cyclic_iter(iterable):
        while True:
            for x in iterable:
                yield x

    @remote_function()
    def __iter__(self):
        assert self.template is not None, 'Set template first to do packing.'
        try:
            next(iter(self.dataset))
        except StopIteration:
            return

        if self.cyclic:
            iterator = self.cyclic_iter(self.dataset)
        else:
            iterator = iter(self.dataset)
        data = []
        max_length = self.template.max_length or 2048
        while True:
            num_samples = self._put_data_in_queue(iterator)
            finished = num_samples != self.packing_interval
            data = self._fetch_data_out_queue(data, num_samples)
            sequences, data = PackingDataset._calculate_matched_group(data, max_length, is_finished=finished)
            res = []
            for rows in sequences:
                output = {}
                # rows: [({'input_ids': [0,1,2,...]}, length), ({'input_ids': [0,1,2,...]}, length)]
                for key in rows[0][0]:
                    output[key] = [r[0][key] for r in rows]
                    if isinstance(rows[0][0][key], (list, np.ndarray)) and isinstance(rows[0][0][key][0], (int, float, np.number)):
                        output[key] = [v for lst in output[key] for v in lst]
                res.append(output)
            yield from res
            if finished:
                break
