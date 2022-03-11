from .StagingPoolPrefetcher import *
from storage.CacheEntry import *


class FrequencyPrefetcher(StagingPoolPrefetcher):
    """
    Prefetching strategy that fills buffers according to the number of times a node will access an item.
    """

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
                 in_order: bool = False,
                 cleanup_read: bool = False):
        self.in_order = in_order
        self.cleanup_read = cleanup_read
        self.prefetch_order = {}
        self.prefetched_until = {j: 0 for j in range(N)}
        self.to_remove = {j: [] for j in range(N)}
        self.freq_prefetch_time = {j: {k: {l: 0 for l in range(p[k])}
                                for k in range(len(storage_classes))}
                                for j in range(N)}
        self.accesses = {j: {} for j in range(N)}
        super().__init__(node_local_batches, file_sizes, N, storage_classes, p, t, d, b_fs, b_c, beta, c, stats)

    def fill_buffers(self, time, node_id):
        if time == 0:
            self.prefetch_order[node_id] = self._get_access_statistics(node_id)
        prefetch_string = self.prefetch_order[node_id]
        prefetch_offset = self.prefetched_until[node_id]
        next_file = prefetch_string[prefetch_offset]
        next_file_size = self.file_sizes[next_file]
        caches = self.caches[node_id]
        for j in range(1, len(caches)):
            cache = caches[j]
            thread_counter = 0
            while cache.can_add(next_file_size) and prefetch_offset < len(prefetch_string) - 1:
                curr_time = self.freq_prefetch_time[node_id][j][thread_counter]
                avail_at = curr_time + self._get_read_time(next_file_size, self.N, 0, None)
                self.freq_prefetch_time[node_id][j][thread_counter] = avail_at
                self.stats.add_prefetch_read(node_id, curr_time, 1)
                entry = CacheEntry(next_file, avail_at, next_file_size, None, None, 0)
                cache.add(entry)

                prefetch_offset += 1
                next_file = prefetch_string[prefetch_offset]
                next_file_size = self.file_sizes[next_file]
                thread_counter = (thread_counter + 1) % self.p[j]
        self.prefetched_until[node_id] = prefetch_offset
        super().fill_buffers(time, node_id)

    def cleanup_buffers(self, time, node_id):
        if self.cleanup_read:
            for file_id in self.to_remove[node_id]:
                caches = self.caches[node_id]
                for j in range(1, len(caches)):
                    cache = caches[j]
                    if cache.get_by_id(file_id):
                        cache.remove_by_id(file_id)
            self.to_remove[node_id] = []
        super().cleanup_buffers(time, node_id)

    def request(self, batch_no: int, offset: int, node_id: int, time: float):
        if self.cleanup_read:
            file_id = self.node_local_batches[batch_no][node_id][offset]
            if len(self.accesses[node_id][file_id]) == 1:
                self.to_remove[node_id].append(file_id)
            else:
                self.accesses[node_id][file_id].remove((batch_no, offset))
        return super().request(batch_no, offset, node_id, time)

    def _get_access_statistics(self, node_id) -> List[int]:
        """
        Returns the list of file ids, sorted by the number of times a file will be accessed by the node
        """
        no_accesses = {i: 0 for i in range(len(self.file_sizes))}
        for batch_no in range(0, len(self.node_local_batches)):
            batch = self.node_local_batches[batch_no]
            node_local_batch = batch[node_id]
            for offset in range(0, len(node_local_batch)):
                file_id = node_local_batch[offset]
                no_accesses[file_id] += 1
                if file_id not in self.accesses[node_id]:
                    self.accesses[node_id][file_id] = [(batch_no, offset)]
                else:
                    self.accesses[node_id][file_id].append((batch_no, offset))
        file_list = sorted(no_accesses.items(), key=lambda x: x[1], reverse=True)
        file_list = [x[0] for x in file_list]
        if self.in_order:
            level_start = 0
            level_end = 0
            level = 0
            while level < len(self.storage_classes):
                level_capacity = self.d[level]
                size = 0
                while size <= level_capacity and level_end != len(self.file_sizes):
                    next_file = file_list[level_end]
                    size += self.file_sizes[next_file]
                    level_end += 1
                file_list[level_start:level_end] = sorted(file_list[level_start:level_end],
                                                          key=lambda x: self.accesses[node_id][x][0] if x in self.accesses[node_id] else (len(self.node_local_batches),0))
                level_start = level_end
                level += 1

        return file_list
