from .StagingPoolPrefetcher import *
from storage.CacheEntry import *


class LBANNInMemoryStorePolicy(StagingPoolPrefetcher):

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
                 preloading: bool = True):
        super().__init__(node_local_batches, file_sizes, N, storage_classes, p, t, d, b_fs, b_c, beta, c, stats, True,
                         False, False, False, False)
        aggregated_capacity = d[1] * N
        dataset_size = sum(file_sizes)
        if dataset_size > aggregated_capacity:
            raise Exception("Policy doesn't support datasets that don't fit in memory")
        self.preloading = preloading
        self.file_batch_size = int(len(file_sizes) / N)
        read_offset = 0
        for i in range(N):
            self._init_from_pfs(i, read_offset, min(read_offset + self.file_batch_size, len(file_sizes)))
            read_offset += self.file_batch_size
        for i in range(N):
            self.fill_buffers(0, i)

    def fill_buffers(self, time: float, node_id: int):
        super().fill_buffers(time, node_id)

    def cleanup_buffers(self, time: float, node_id: int):
        super().cleanup_buffers(time, node_id)

    def _init_from_pfs(self, node_id, read_start, read_end):
        # Fill buffers and update stall time at start
        curr_cache = self.caches[node_id][1]
        stall_times = [0] * self.p[1]
        thread_id = 0
        if self.preloading:
            for k in range(read_start, read_end):
                file_size = self.file_sizes[k]
                stall_times[thread_id] += self._get_read_time(file_size, self.N, 0, None)
                cache_entry = CacheEntry(k, 0, file_size, None, None, 0)
                curr_cache.add(cache_entry)
                self.stats.add_prefetch_read(node_id, stall_times[thread_id], 1)
                thread_id = (thread_id + 1) % self.p[1]
        else:
            prefetch_counter = 0
            batch_no = 0
            offset = 0
            while prefetch_counter < self.file_batch_size:
                file_id = self.node_local_batches[batch_no][node_id][offset]
                file_size = self.file_sizes[file_id]
                stall_times[thread_id] += self._get_read_time(file_size, self.N, 0, None)
                cache_entry = CacheEntry(file_id, stall_times[thread_id], file_size, None, None, 0)
                curr_cache.add(cache_entry)
                self.stats.add_prefetch_read(node_id, stall_times[thread_id], 1)
                thread_id = (thread_id + 1) % self.p[1]

                if offset < len(self.node_local_batches[batch_no][node_id]) - 1:
                    offset += 1
                else:
                    offset = 0
                    batch_no += 1
                prefetch_counter += 1

        if self.preloading:
            stall_time = max(stall_times)
            self.stats.add_init_stall_time(node_id, stall_time)
