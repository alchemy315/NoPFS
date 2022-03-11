from .BasePolicy import *
from storage.CacheEntry import *


class StagingPoolPrefetcher(BasePolicy):
    """
    Prefetching policy that fills the staging pool from the best option until it's full and drops files when consumed
    If the policy is instantiated directly, this corresponds to prefetching from the PFS. Other policies can extend
    the policy and implement their own prefetching strategy for storage levels other than the staging pool, while reusing
    the staging pool prefetching logic from this class. If they do this, they have to call super().fill_buffers /
    super().cleanup_buffers after they prefetched according to their strategy.
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
                 remote_prefetch: bool = False,
                 jit_prefetch: bool = False,
                 pfs_prefetch: bool = True,
                 wait_for_staging_pool: bool = False,
                 init_buffer: bool = True):
        """
        @param remote_prefetch: Indicates whether remote files should be considered
        @param jit_prefetch: Prefetch such that file can be consumed in time for sure (i.e. skip files if consumption is too fast)
        @param pfs_prefetch: Indicates whether the PFS should be considered
        """
        self.prefetch_batch_no = {j: 0 for j in range(N)}
        self.prefetch_offsets = {j: 0 for j in range(N)}  # next offset that should be fetched
        self.prefetch_time = {j: {k: 0 for k in range(p[0])} for j in range(N)}
        self.cleanup_batch_no = {j: 0 for j in range(N)}
        self.cleanup_offsets = {j: 0 for j in range(N)}
        self.thread_counter = 0  # threads prefetch in circular fashion
        self.remote_prefetch = remote_prefetch
        self.jit_prefetch = jit_prefetch
        self.pfs_prefetch = pfs_prefetch
        super().__init__(node_local_batches, file_sizes, N, storage_classes, p, t, d, b_fs, b_c, beta, c, stats,
                         wait_for_staging_pool, init_buffer)

    def fill_buffers(self, time, node_id):
        # Check for empty batches
        while self.prefetch_batch_no[node_id] != len(self.node_local_batches) and len(
                self.node_local_batches[self.prefetch_batch_no[node_id]][node_id]) == 0:
            self.prefetch_offsets[node_id] = 0
            self.prefetch_batch_no[node_id] += 1

        # Check if finished prefetching
        if self.prefetch_batch_no[node_id] == len(self.node_local_batches) and self.prefetch_offsets[node_id] == 0:
            return

        staging_pool = self.caches[node_id][0]

        # Skip prefetching if consumed too fast
        if self.jit_prefetch:
            next_read_batch_no, next_read_offset = self._get_next_read(node_id)
            next_read_file = self.node_local_batches[next_read_batch_no][node_id][next_read_offset]
            next_read_filesize = self.file_sizes[next_read_file]
            consumed_in = 0
            prefetched_in = 0 if staging_pool.get_by_id_and_batch(next_read_file, next_read_batch_no, next_read_offset) \
                                 is not None else self._get_read_time(next_read_filesize, self.N, 0, None)
            while prefetched_in > consumed_in:
                consumed_in += next_read_filesize / self.c
                if next_read_offset == len(self.node_local_batches[next_read_batch_no][node_id]) - 1:
                    if next_read_batch_no == len(self.node_local_batches) - 1:
                        break
                    else:
                        next_read_batch_no += 1
                        next_read_offset = 0
                else:
                    next_read_offset += 1
                next_read_file = self.node_local_batches[next_read_batch_no][node_id][next_read_offset]
                next_read_filesize = self.file_sizes[next_read_file]
                prefetched_in = 0 if staging_pool.get_by_id_and_batch(next_read_file, next_read_batch_no,
                                                                      next_read_offset) is not None else self._get_read_time(
                    next_read_filesize, self.N, 0, None)

                self.prefetch_batch_no[node_id] = next_read_batch_no
                self.prefetch_offsets[node_id] = next_read_offset

        next_file = self.node_local_batches[self.prefetch_batch_no[node_id]][node_id][self.prefetch_offsets[node_id]]
        next_file_size = self.file_sizes[next_file]


        for k in range(self.p[0]):
            self.prefetch_time[node_id][k] = max(time, self.prefetch_time[node_id][k])

        while staging_pool.can_add(next_file_size) and self.prefetch_batch_no[node_id] < len(self.node_local_batches):
            curr_time = self.prefetch_time[node_id][self.thread_counter]
            cache_entry, cache_level = self._get_maxspeed_local_buffer_entry(next_file, node_id, curr_time, self.prefetch_batch_no[node_id], self.prefetch_offsets[node_id], False)
            option = 2
            avail_at = None
            if cache_entry is not None:
                avail_at = cache_entry.avail_from
            if self.remote_prefetch:
                remote_entry, remote_level = self._get_maxspeed_remote_buffer_entry(next_file, node_id, curr_time, self.prefetch_batch_no[node_id], self.prefetch_offsets[node_id], False)
                if remote_entry is not None and(cache_entry is None or
                                                 self._get_speed(cache_level) < min(self.b_c, self._get_speed(remote_level))):
                    if remote_entry.avail_from > curr_time and self.pfs_prefetch:
                        # Only consider non available remote entries when we don't prefetch from PFS
                        break
                    cache_entry = remote_entry
                    cache_level = remote_level
                    avail_at = remote_entry.avail_from
                    option = 1
            pfs_clients = self.stats.get_pfs_clients(curr_time, node_id)
            if self.pfs_prefetch and (cache_entry is None or
                    self._get_read_time(next_file_size, pfs_clients, 0, None) < self._get_read_time(next_file_size, None, option, cache_level) or
                    avail_at > curr_time):
                option = 0
                avail_at = curr_time + self._get_read_time(next_file_size, pfs_clients, 0, None)
            elif cache_entry is not None:
                if avail_at is None:
                    avail_at = curr_time + self._get_read_time(next_file_size, None, option, cache_level)
                else:
                    avail_at = max(avail_at, curr_time) + self._get_read_time(next_file_size, None, option, cache_level)
            if avail_at is not None:
                self.prefetch_time[node_id][self.thread_counter] = avail_at
                if option == 0:
                    log_option = 1
                elif option == 1:
                    log_option = 3
                elif option == 2:
                    log_option = 2
                entry = CacheEntry(next_file, avail_at, next_file_size, self.prefetch_batch_no[node_id],
                                   self.prefetch_offsets[node_id], log_option)
                staging_pool.add(entry)
                self.stats.add_prefetch_read(node_id, curr_time, log_option) # + 1?

            curr_offset = self.prefetch_offsets[node_id]
            curr_batch_no = self.prefetch_batch_no[node_id]
            if curr_offset < len(self.node_local_batches[curr_batch_no][node_id]) - 1:
                self.prefetch_offsets[node_id] += 1
            else:
                self.prefetch_offsets[node_id] = 0
                self.prefetch_batch_no[node_id] += 1
            # Check if node has empty batches
            while self.prefetch_batch_no[node_id] != len(self.node_local_batches) and len(self.node_local_batches[self.prefetch_batch_no[node_id]][node_id]) == 0:
                self.prefetch_batch_no[node_id] += 1
            if self.prefetch_batch_no[node_id] != len(self.node_local_batches):
                next_file = self.node_local_batches[self.prefetch_batch_no[node_id]][node_id][
                    self.prefetch_offsets[node_id]]
                next_file_size = self.file_sizes[next_file]

            self.thread_counter = (self.thread_counter + 1) % self.p[0]

    def cleanup_buffers(self, time, node_id):
        super()._cleanup_read(node_id, 0)
        self.fill_buffers(time, node_id)