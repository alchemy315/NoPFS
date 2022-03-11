import numpy as np
import copy
from .StagingPoolPrefetcher import *
from storage.CacheEntry import *


class DeepIOPolicy(StagingPoolPrefetcher):

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
                 stats: Statistics,
                 opportunistic_reordering: bool = False):
        if opportunistic_reordering:
            super().__init__(node_local_batches, file_sizes, N, storage_classes, p, t, d, b_fs, b_c, beta, c, stats, True,
                         False, False, False, False)
        else:
            super().__init__(node_local_batches, file_sizes, N, storage_classes, p, t, d, b_fs, b_c, beta, c, stats, True,
                         False, False, False, False)
        if opportunistic_reordering:
            self.node_local_batches = copy.deepcopy(self.node_local_batches)
        dataset_size = sum(self.file_sizes)
        self.offsets = []
        if dataset_size / N > self.d[1]:
            self.hybrid = True  # Hybrid mode, i.e. dataset doesn't fit on aggregated memory
            self.offsets.append(0)
        else:
            self.hybrid = False
            self.bucket_size = int(len(file_sizes) / N)
            for i in range(self.N):
                self.offsets.append(self.bucket_size * i)
        for i in range(N):
            self._init_buffers(i)
        if opportunistic_reordering:
            self._reorder_batches()
        for i in range(N):
            self.fill_buffers(0, i)

    def fill_buffers(self, time: float, node_id: int):
        super().fill_buffers(time, node_id)

    def cleanup_buffers(self, time: float, node_id: int):
        super().cleanup_buffers(time, node_id)

    def _init_buffers(self, node_id):
        # Fill buffers and update stall time at start
        curr_cache = self.caches[node_id][1]
        stall_times = [0] * self.p[1]
        thread_id = 0
        # Add as much files as possible to individual nodes
        file_offset = self.offsets[node_id] if node_id > 0 else 0
        if not self.hybrid:
            if node_id != self.N - 1:
                end = self.offsets[node_id + 1]
            else:
                end = len(self.file_sizes)
        curr_file_size = self.file_sizes[file_offset]
        while (self.hybrid and curr_cache.can_add(curr_file_size)) or (not self.hybrid and file_offset < end):
            cache_entry = CacheEntry(file_offset, 0, curr_file_size, None, None, 0)
            curr_cache.add(cache_entry)
            stall_times[thread_id] += self._get_read_time(curr_file_size, self.N, 0, None)
            file_offset += 1
            if self.hybrid or file_offset != end:
                curr_file_size = self.file_sizes[file_offset]
            self.stats.add_prefetch_read(node_id, stall_times[thread_id], 1)
            thread_id = (thread_id + 1) % self.p[1]

        if self.hybrid:
            self.offsets.append(file_offset)
        stall_time = max(stall_times)
        self.stats.add_init_stall_time(node_id, stall_time)

    def _reorder_batches(self):
        # To simulate opportunistic reordering:
        # Choose with prob. 1/N local file, N-1/N remote file
        for batch_no in range(len(self.node_local_batches)):
            batch = self.node_local_batches[batch_no]
            for i, file_ids in batch.copy().items():
                no_items = len(file_ids)
                for offset in range(no_items):
                    source = np.random.randint(1, self.N)
                    if source == 1 or i == 0:
                        if i == self.N - 1:
                            end = max(self.caches[i][1].entries.keys())
                        else:
                            end = self.offsets[i + 1] - 1
                        file_id = np.random.randint(self.offsets[i], end)
                    else:
                        if i != 0:
                            start = self.offsets[i - 1]
                            end = self.offsets[i] - 1
                        else:
                            # TODO: Handle case i=0 properly (currently prefetching fails, because file isn't available at the moment) ->
                            # In staging pool prefetcher, when remoxte_prefetch is enabled, could do two runs: One for checking which items will be available, second one for the avail time (or update times in second run)
                            start = self.offsets[1]
                            end = self.offsets[2] - 1
                        file_id = np.random.randint(start, end)
                    batch[i][offset] = file_id