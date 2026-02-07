# Copyright (c) ModelScope Contributors. All rights reserved.
import multiprocessing as mp
from itertools import chain
from typing import List, TypeVar
import os
import numpy as np
from tqdm import tqdm

from twinkle.infra import remote_class, remote_function
from .base import Dataset, DatasetMeta

_T = TypeVar('_T')


@remote_class(execute='first')
class PackingDataset(Dataset):
    """A packing dataset wrapper, this will use binpacking to pack the dataset rows to minimum number of batches,
        whose lengths are almost `max_length`

    Args:
        dataset_meta: The dataset meta
        packing_num_proc: The number of processes to use for packing
    """

    PACKING_BATCH_SIZE = 1000

    def __init__(self, dataset_meta: DatasetMeta, packing_num_proc: int = 1, **kwargs):
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        self.packing_num_proc = packing_num_proc
        super().__init__(dataset_meta, **kwargs)
        self._out_queue = mp.Queue()
        self.packed_idx = []
        self.packed_length = []
        self._packed_called = False

    @remote_function()
    def pack_dataset(self):
        """Call to start packing dataset"""
        assert 'input_ids' in self.dataset[0], 'Tokenize dataset first to do packing.'
        assert self.template is not None, 'Set template first to do packing.'
        lengths = self.dataset['length']
        offset = 0
        chunked_lengths = PackingDataset._split_list(lengths, self.packing_num_proc)
        for i in range(self.packing_num_proc):
            worker = mp.Process(
                target=self.create_packed_idx, args=(
                    i,
                    offset,
                    chunked_lengths[i],
                ), daemon=True)
            worker.start()
            offset += len(chunked_lengths[i])
        self.packed_idx = [[] for _ in range(self.packing_num_proc)]
        self.packed_length = [[] for _ in range(self.packing_num_proc)]
        desc = 'Packing: ' if self.packing_num_proc == 1 else f'Packing (num_proc={self.packing_num_proc}): '
        with tqdm(total=len(lengths), dynamic_ncols=True, desc=desc) as prog_bar:
            finished_workers = 0
            while finished_workers < self.packing_num_proc:
                rank, sequences, data_len = self._out_queue.get()
                if data_len == -1:
                    finished_workers += 1
                    continue
                prog_bar.update(data_len)
                self.packed_idx[rank] += [[x[0] for x in seq] for seq in sequences]
                self.packed_length[rank] += [sum(x[1] for x in seq) for seq in sequences]
        self.packed_idx = list(chain.from_iterable(self.packed_idx))
        self.packed_length = list(chain.from_iterable(self.packed_length))
        self._packed_called = True

    def create_packed_idx(self, rank, offset, lengths):
        data = [(i + offset, sum(length) if isinstance(length, list) else length) for i, length in enumerate(lengths)]
        i = 0
        input_data = []
        while True:
            new_data = data[i:i + self.PACKING_BATCH_SIZE]
            input_data += new_data
            if not input_data:
                break
            i += self.PACKING_BATCH_SIZE
            is_finished = i >= len(data)
            sequences, input_data = PackingDataset._calculate_matched_group(
                input_data, self.template.max_length or 2048, is_finished=is_finished)
            self._out_queue.put((rank, sequences, len(new_data)))
        self._out_queue.put((rank, [], -1))

    @staticmethod
    def _calculate_matched_group(sequences, packing_length: int, is_finished: bool = True):
        if len(sequences) == 0:
            return [], []
        # https://arxiv.org/pdf/2404.10830
        import binpacking
        sequences = binpacking.to_constant_volume(sequences, packing_length, weight_pos=1)
        if sequences and not is_finished:
            sequences, ret_sequences = sequences[:-1], sequences[-1]
        else:
            ret_sequences = []
        return sequences, ret_sequences

    @staticmethod
    def _split_list(ori_list: List[_T], num_shards: int, contiguous=True) -> List[List[_T]]:
        shard = []
        if contiguous:
            idx_list = np.linspace(0, len(ori_list), num_shards + 1, dtype=np.int64)
            for i in range(len(idx_list) - 1):
                shard.append(ori_list[idx_list[i]:idx_list[i + 1]])
        else:
            ori_list = np.array(ori_list)
            for i in range(num_shards):
                shard.append(ori_list[np.arange(i, len(ori_list), num_shards)].tolist())
        return shard

    @remote_function()
    def __getitem__(self, index):
        assert self._packed_called, 'Call `pack_dataset()` first before index the sample.'
        sequence = self.packed_idx[index]
        rows = [self.dataset[i] for i in sequence]
        output = {}
        for key in rows[0]:
            output[key] = [r[key] for r in rows]
            if isinstance(rows[0][key], (list, np.ndarray)) and isinstance(rows[0][key][0], (int, float, np.number)):
                output[key] = [v for lst in output[key] for v in lst]
        return output

    @remote_function()
    def __len__(self):
        assert self._packed_called, 'Call `pack_dataset()` first before index the sample.'
        return len(self.packed_idx)
