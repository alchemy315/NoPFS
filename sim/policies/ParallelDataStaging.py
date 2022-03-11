import numpy as np
from .StagingPoolPrefetcher import *
from storage.CacheEntry import *
import copy


class ParallelStagingPolicy(StagingPoolPrefetcher):

    def __init__(self,
                 node_local_batches: List[Dict[int, List[int]]],
                 file_sizes: List[float],
                 N: int,
                 storage_classes: Dict[int, Dict[int, float]],
                 p: Dict[int, int],
                 t: Dict[int, int],
                 d: Dict[int, int],
                 b_fs: float,
                 b_c: float,
                 beta: float,
                 c: float,
                 stats: Statistics):
        super().__init__(node_local_batches, file_sizes, N, storage_classes, p, t, d, b_fs, b_c, beta, c, stats, False,
                         False, False, False, False)
        self.file_batch_size = int(len(file_sizes) / N)
        self.node_local_batches = copy.deepcopy(self.node_local_batches)
        read_offset = 0
        stall_times = []
        for i in range(N):
            stall_times.append(self._init_from_pfs(i, read_offset, min(read_offset + self.file_batch_size, len(file_sizes))))
            read_offset += self.file_batch_size
        for i in range(N):
            self._init_from_remote(i, stall_times[i])
        self._reorder_batches()
        for i in range(N):
            self.fill_buffers(0, i)

    def fill_buffers(self, time: float, node_id: int):
        super().fill_buffers(time, node_id)

    def cleanup_buffers(self, time: float, node_id: int):
        super().cleanup_buffers(time, node_id)

    def _init_from_pfs(self, node_id, read_start, read_end):
        # Fill buffers and update stall time at start
        curr_cache = self.caches[node_id][2]
        stall_times = [0] * self.p[2]
        thread_id = 0
        read_offset = read_start
        curr_file_size = self.file_sizes[read_offset]
        while curr_cache.can_add(curr_file_size) and read_offset < read_end:
            cache_entry = CacheEntry(read_offset, 0, curr_file_size, None, None, 0)
            curr_cache.add(cache_entry)
            stall_times[thread_id] += self._get_read_time(curr_file_size, self.N, 0, None)
            read_offset += 1
            if read_offset != len(self.file_sizes):
                curr_file_size = self.file_sizes[read_offset]
            self.stats.add_prefetch_read(node_id, stall_times[thread_id], 1)
            thread_id = (thread_id + 1) % self.p[2]

        stall_time = max(stall_times)
        return stall_time

    def _init_from_remote(self, node_id, curr_stall_time):
        curr_cache = self.caches[node_id][2]
        thread_id = 0
        stall_times = [curr_stall_time] * self.p[2]
        for i in range(0, len(self.file_sizes)):
            if node_id * self.file_batch_size <= i < (i + 1) * self.file_batch_size:
                continue
            curr_file_size = self.file_sizes[i]
            if not curr_cache.can_add(curr_file_size):
                break
            cache_entry = CacheEntry(i, 0, curr_file_size, None, None, 0)
            curr_cache.add(cache_entry)
            self.stats.add_prefetch_read(node_id, stall_times[thread_id], 3)
            stall_times[thread_id] += self._get_read_time(curr_file_size, None, 1, 2)
            thread_id = (thread_id + 1) % self.p[2]

        stall_time = max(stall_times)
        self.stats.add_init_stall_time(node_id, stall_time)

    def _reorder_batches(self):
        max_file_ids = []
        for i in range(self.N):
            max_file_ids.append(max(list(self.caches[i][2].entries.keys())))
        # Only choose files that are locally stored
        for batch_no in range(len(self.node_local_batches)):
            batch = self.node_local_batches[batch_no]
            for i, file_ids in batch.copy().items():
                no_items = len(file_ids)
                for offset in range(no_items):
                    file_id = np.random.randint(i * self.file_batch_size, max_file_ids[i])
                    batch[i][offset] = file_id