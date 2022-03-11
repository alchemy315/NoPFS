import numpy as np
import math
import os
import time
from typing import List, Dict


class Dataset:

    def __init__(self, config, E, B, N, distr_scheme, drop_last_iter, seed):
        self.config = config
        if self.config['synthetic']:
            self.D = self.config['synthetic_params']['samples']
        else:
            self.D = 0
        self.E = E
        self.B = B
        self.N = N
        self.distr_scheme = distr_scheme
        self.drop_last_iter = drop_last_iter
        if seed is not None:
            np.random.seed(seed)
        self.file_sizes = None
        self.node_local_batches = None

    def get_file_sizes(self):
        if self.file_sizes is not None:
            return self.file_sizes

        if self.config['synthetic']:
            params = self.config['synthetic_params']
            self.file_sizes = self._generate_synth_filesize_distr(params['distribution'],
                                                                  params['loc'],
                                                                  params['scale'],
                                                                  params['min'],
                                                                  params['max'],
                                                                  self.D)
            return self.file_sizes
        else:
            path = self.config['dataset_params']['path']
            self.file_sizes = self._get_path_file_sizes(path)
            return self.file_sizes

    def _generate_global_access_string(self):
        """
        Generates the access string that indicates which sample is accessed when
        """
        num = self.D
        epochs = self.E
        ar = np.empty(num * epochs, dtype=int)
        for i in range(epochs):
            ar[i * num:(i + 1) * num] = np.random.permutation(num)
        return ar

    def get_node_local_batches(self) -> List[
        Dict[int, List[int]]]:
        if self.node_local_batches is not None:
            return self.node_local_batches

        global_access_string = self._generate_global_access_string()
        D = self.D
        E = self.E
        B = self.B
        N = self.N
        distr_scheme = self.distr_scheme
        drop_last_iter = self.drop_last_iter
        node_local_batches = []
        for i in range(E):
            iterations = math.floor(D / B) if drop_last_iter else math.ceil(D / B)
            for j in range(iterations):
                batch_start = D * i + j * B
                batch_end = min(batch_start + B, (i + 1) * D)
                batch = global_access_string[batch_start:batch_end]
                local_batches = {}
                if distr_scheme == "uniform":
                    local_batch_size = math.ceil(B / N)
                    node_id = 0
                    while node_id * local_batch_size < B:
                        local_batch = batch[node_id * local_batch_size:min((node_id + 1) * local_batch_size, B)]
                        local_batches[node_id] = local_batch
                        node_id += 1
                    if node_id != self.N:
                        for empty_nodes in range(node_id, self.N):
                            local_batches[empty_nodes] = []
                    node_local_batches.append(local_batches)
                elif distr_scheme == "whole":
                    for node_id in range(N):
                        local_batches[node_id] = batch
                    node_local_batches.append(local_batches)
                elif distr_scheme == "imbalanced":
                    node_id = 0
                    batch_offset = 0
                    local_batch_size = math.ceil(B / N) / 2 if node_id < N / 2 else math.ceil(B / N) * 2
                    while batch_offset < B:
                        local_batch = batch[batch_offset:min(batch_offset + local_batch_size, B)]
                        local_batches[node_id] = local_batch
                        node_id += 1
                    if node_id != self.N:
                        for empty_nodes in range(node_id, self.N):
                            local_batches[empty_nodes] = []
                    node_local_batches.append(local_batches)
                else:
                    raise NotImplementedError()
        self.node_local_batches = node_local_batches
        return self.node_local_batches

    @staticmethod
    def _generate_synth_filesize_distr(dist, loc, scale, mini, maxi, num):
        if mini is not None and mini == maxi:
            return [mini] * num
        ar = dist.rvs(loc=loc, scale=scale, size=num)
        if mini is not None or maxi is not None:
            for i in range(num):
                while (mini is not None and ar[i] < mini) or (maxi is not None and ar[i] > maxi):
                    ar[i] = dist.rvs(loc=loc, scale=scale)
        return ar

    @staticmethod
    def _get_path_file_sizes(path):
        file_sizes = []
        if not os.path.exists(path):
            raise ValueError("Invalid path specified")
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                file_sizes.append(os.path.getsize(file_path) / 1024 / 1024)
        return file_sizes
