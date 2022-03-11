from .StagingPoolPrefetcher import *
from storage.CacheEntry import *
import copy


class LocalityAwareDataLoadingPolicy(StagingPoolPrefetcher):

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
        super().__init__(node_local_batches, file_sizes, N, storage_classes, p, t, d, b_fs, b_c, beta, c, stats, True,
                         False, True, False, False)
        self.file_batch_size = int(len(file_sizes) / N)
        self.node_local_batches = copy.deepcopy(self.node_local_batches)
        read_offset = 0
        for i in range(N):
            self._init_from_pfs(i, read_offset, min(read_offset + self.file_batch_size, len(file_sizes)))
            read_offset += self.file_batch_size
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
        self.stats.add_init_stall_time(node_id, stall_time)

    def _reorder_batches(self):
        for batch_no in range(len(self.node_local_batches)):
            batch = self.node_local_batches[batch_no]
            distributed = [[False] * len(file_ids) for file_ids in batch.values()]
            for i, file_ids in batch.copy().items():
                curr_cache = self.caches[i][2]
                no_items = len(file_ids)
                for offset in range(no_items):
                    if curr_cache.get_by_id(batch[i][offset]) is not None:
                        distributed[i][offset] = True
                        continue
                    local_item = False
                    for node_id in range(0, self.N):
                        if node_id == i:
                            continue
                        for distr_offset in range(len(self.node_local_batches[batch_no][node_id])):
                            if not distributed[node_id][distr_offset] and \
                                    curr_cache.get_by_id(
                                        self.node_local_batches[batch_no][node_id][distr_offset]) is not None:
                                distributed[node_id][distr_offset] = True
                                batch[i][offset] = self.node_local_batches[batch_no][node_id][distr_offset]
                                local_item = True
                                break
                        if local_item:
                            break
                    if not local_item:
                        distributed[i][offset] = True
