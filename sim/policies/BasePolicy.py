from storage.Buffer import Buffer
from stats.Statistics import Statistics
from typing import List, Dict


class BasePolicy:
    """
    Base class for policies that is extended by specific policies.
    Contains helper functions and global prefetching / caching logic
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
                 wait_for_staging_pool: bool = False,
                 init_buffer: bool = True):
        """
        @param node_local_batches: Global access string
        @param file_sizes: List with file sizes (indexed by file id)
        @param N: Number of nodes
        @param storage_classes: Dict with storage classes and their read speed as a function (dict) of the number of threads
        @param p: Prefetcher threads per storage class
        @param t: Speed of the PFS
        @param d: Capacity of the storage classes
        @param b_fs: Filesystem bandwidth
        @param b_c: Network bandwidth for communication with clients
        @param beta: Preprocessing rate
        @param c: Processing rate of neural network
        @param stats: Object for representing the statistics of a policy execution
        @param wait_for_staging_pool: Wait until item is ready in staging pool, even if it takes longer
        @param init_buffer: Call fill_buffers in init method
        """
        self.node_local_batches = node_local_batches
        self.file_sizes = file_sizes
        self.N = N
        self.storage_classes = storage_classes
        self.p = p
        self.t = t
        self.d = d
        self.beta = beta
        self.b_fs = b_fs
        self.b_c = b_c
        self.c = c
        self.wait_for_staging_pool = wait_for_staging_pool
        self.stats = stats
        self.caches = {}  # One entry per node that contains dict with storage class and array with CacheEntries
        for i in range(N):
            self.caches[i] = {j: Buffer(j, d[j]) for j in storage_classes.keys()}

        self.request_batch_no = {j: -1 for j in range(N)} # Which requests were already served
        self.request_batch_offset = {j: -1 for j in range(N)}

        self.cleanup_batch_no = {i: {j: 0 for j in range(len(storage_classes))} for i in range(N)}
        self.cleanup_offsets = {i: {j: 0 for j in range(len(storage_classes))} for i in range(N)}

        if init_buffer:
            for i in range(N):
                self.fill_buffers(0, i)

    def request(self, batch_no: int, offset: int, node_id: int, time: float):
        """
        Returns the time when request was fulfilled
        If entry is not yet available in staging pool, need to fetch and write it (to memory)
        """
        file_id = self.node_local_batches[batch_no][node_id][offset]
        file_size = self.file_sizes[file_id]
        staging_pool_entry = self.caches[node_id][0].get_by_id_and_batch(file_id, batch_no, offset)
        staging_pool_time = float('inf')
        immediate = False # Because of floating point errors, ensures stall time is 0
        if staging_pool_entry is not None:
            if staging_pool_entry.avail_from <= time:
                staging_pool_time = time
                immediate = True
                self.stats.add_read(node_id, time, 0)
            else:
                staging_pool_time = staging_pool_entry.avail_from
        elif self.wait_for_staging_pool:
            print("File {} not in staging pool".format(file_id))
        write_time = self._get_write_time(file_size)

        ideal_option = 1
        pfs_clients = self.stats.get_pfs_clients(time, node_id)
        fetch_pfs = self._get_fetch_time(file_size, pfs_clients, 0, None)
        min_fetch_time = fetch_pfs

        local_buffer_entry, local_level = self._get_maxspeed_local_buffer_entry(file_id, node_id, time, batch_no, offset)
        if local_buffer_entry is not None:
            fetch_local = file_size / self._get_speed(local_level)
            if fetch_local < min_fetch_time:
                ideal_option = 2
                min_fetch_time = fetch_local
        remote_buffer_entry, remote_level = self._get_maxspeed_remote_buffer_entry(file_id, node_id, time, batch_no, offset)
        if remote_buffer_entry is not None:
            fetch_remote = file_size / min(self.b_c, self._get_speed(remote_level))
            assert(fetch_remote > 0)
            if fetch_remote < min_fetch_time:
                ideal_option = 3
                min_fetch_time = fetch_remote
        self.request_batch_no[node_id] = batch_no
        self.request_batch_offset[node_id] = offset
        if time + min_fetch_time + write_time < staging_pool_time and not self.wait_for_staging_pool:
            avail_at = time + min_fetch_time + write_time
            self.stats.add_read(node_id, time, ideal_option)
        else:
            avail_at = staging_pool_time
            self.stats.add_read(node_id, avail_at, staging_pool_entry.source)
        self.cleanup_buffers(avail_at, node_id)
        return avail_at, immediate

    def fill_buffers(self, time: float, node_id: int):
        """
        Fills buffers according to prefetch strategy. Should be overwritten by policies.
        """
        return

    def cleanup_buffers(self, time: float, node_id: int):
        """
        Evicts from buffers according to strategy. Should be overwritten by policies.
        """
        return

    def get_stats(self):
        return self.stats

    def _get_maxspeed_local_buffer_entry(self, file_id, node_id, avail_at, batch_no, batch_offset, strict_deadline = True):
        max_local_buffer_entry = None
        storage_level = 0
        max_speed = 0
        for local_level, local_buffer in self.caches[node_id].items():
            if local_level != 0:
                level_speed = self._get_speed(local_level)
                local_buffer_entry = local_buffer.get_by_id_and_batch(file_id, batch_no, batch_offset)
                if local_buffer_entry is not None and (not strict_deadline or local_buffer_entry.avail_from <= avail_at) and level_speed > max_speed:
                    max_local_buffer_entry = local_buffer_entry
                    max_speed = level_speed
                    storage_level = local_level
        return max_local_buffer_entry, storage_level

    def _get_maxspeed_remote_buffer_entry(self, file_id, node_id, avail_at, batch_no, batch_offset, strict_deadline = True):
        max_remote_buffer_entry = None
        max_speed = 0
        storage_level = 0
        for i in range(self.N):
            if i != node_id:
                max_node_buffer_entry, remote_storage_level = self._get_maxspeed_local_buffer_entry(file_id, i,
                                                                                                    avail_at, batch_no, batch_offset, strict_deadline)
                if max_node_buffer_entry is not None and self._get_speed(remote_storage_level) > max_speed:
                    max_remote_buffer_entry = max_node_buffer_entry
                    storage_level = remote_storage_level
        return max_remote_buffer_entry, storage_level

    def _get_speed(self, storage_level):
        return self.storage_classes[storage_level][self.p[storage_level]]

    def _get_pfs_speed(self, clients):
        if clients in self.t:
            speed = self.t[clients] / clients
        else:
            # Interpolate linearly
            smaller_keys = [key for key in self.t.keys() if key < clients]
            larger_keys = [key for key in self.t.keys() if key > clients]
            if not smaller_keys:
                speed = self.t[min(larger_keys)] / clients
            elif not larger_keys:
                speed = self.t[max(larger_keys)] / clients
            else:
                lower_key = max(smaller_keys)
                upper_key = min(larger_keys)
                speed = (self.t[lower_key] + (clients - lower_key) * (self.t[upper_key] - self.t[lower_key])/(upper_key - lower_key)) / clients
        return speed

    def _get_write_time(self, file_size):
        memory_write_speed = self.storage_classes[0][self.p[0]] / self.p[0]
        return max(file_size / self.beta, file_size / memory_write_speed)

    def _get_fetch_time(self, file_size, pfs_clients, option, storage_level):
        if option == 0:
            return file_size / min(self.b_fs, self._get_pfs_speed(pfs_clients))
        elif option == 1:
            return file_size / min(self.b_c, self._get_speed(storage_level))
        elif option == 2:
            return file_size / self._get_speed(storage_level)

    def _get_read_time(self, file_size, pfs_clients, option, storage_level):
        return self._get_write_time(file_size) + self._get_fetch_time(file_size, pfs_clients, option, storage_level)

    def _get_next_read(self, node_id):
        curr_batch_no = self.request_batch_no[node_id]
        curr_offset = self.request_batch_offset[node_id]
        if curr_batch_no == -1 and curr_offset == -1:
            return 0, 0
        if curr_batch_no == len(self.node_local_batches) - 1 and curr_offset == len(self.node_local_batches[-1][node_id]) - 1:
            return None
        if curr_offset == len(self.node_local_batches[curr_batch_no][node_id]) - 1:
            return curr_batch_no + 1, 0
        else:
            return curr_batch_no, curr_offset + 1

    def _cleanup_read(self, node_id, storage_class):
        """
        Helper method to cleanup all already read entries for a given storage_class
        @param node_id:
        @param storage_class:
        @return:
        """
        buffer = self.caches[node_id][storage_class]
        while self.cleanup_batch_no[node_id][storage_class] < self.request_batch_no[node_id] or \
            self.cleanup_batch_no[node_id][storage_class] == self.request_batch_no[node_id] and self.cleanup_offsets[node_id][storage_class] <= self.request_batch_offset[node_id]:
                # Check for empty node local batches
                while len(self.node_local_batches[self.cleanup_batch_no[node_id][storage_class]][node_id]) == 0 and \
                    self.cleanup_batch_no[node_id][storage_class] < self.request_batch_no[node_id]:
                    self.cleanup_offsets[node_id][storage_class] = 0
                    self.cleanup_batch_no[node_id][storage_class] += 1
                if self.cleanup_batch_no[node_id][storage_class] == self.request_batch_no[node_id]:
                    break
                file_id = self.node_local_batches[self.cleanup_batch_no[node_id][storage_class]][node_id][self.cleanup_offsets[node_id][storage_class]]
                buffer.remove_by_id_and_batch(file_id, self.cleanup_batch_no[node_id][storage_class], self.cleanup_offsets[node_id][storage_class])
                self.cleanup_offsets[node_id][storage_class] += 1
                if self.cleanup_offsets[node_id][storage_class] == len(self.node_local_batches[self.cleanup_batch_no[node_id][storage_class]][node_id]):
                    self.cleanup_batch_no[node_id][storage_class] += 1
                    self.cleanup_offsets[node_id][storage_class] = 0