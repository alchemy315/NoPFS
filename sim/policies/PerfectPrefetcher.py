from .BasePolicy import *
from storage.CacheEntry import *


class PerfectPrefetcher(BasePolicy):
    """
    Prefetcher that immediately returns, i.e. no stalling
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
                 stats: Statistics):
        super().__init__(node_local_batches, file_sizes, N, storage_classes, p, t, d, b_fs, b_c, beta, c, stats)

    def fill_buffers(self, time, node_id):
        staging_pool = self.caches[node_id][0]
        next_batch_no = self.request_batch_no[node_id]
        if len(self.node_local_batches[next_batch_no][node_id]) == 0:
            return
        next_batch_offset = (self.request_batch_offset[node_id] + 1) % len(self.node_local_batches[next_batch_no][node_id])
        if next_batch_offset == 0:
            next_batch_no += 1
        if next_batch_no == len(self.node_local_batches):
            return
        next_file = self.node_local_batches[next_batch_no][node_id][next_batch_offset]
        next_file_size = self.file_sizes[next_file]
        entry = CacheEntry(next_file, 0, next_file_size, next_batch_no, next_batch_offset, 0)
        staging_pool.add(entry)

    def cleanup_buffers(self, time, node_id):
        super()._cleanup_read(node_id, 0)
        self.fill_buffers(time, node_id)