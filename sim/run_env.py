import os
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats

from policies.PolicyExecutor import *
from policies.StagingPoolPrefetcher import *
from policies.PerfectPrefetcher import *
from policies.FrequencyPrefetcher import *
from policies.DeepIOPolicy import *
from policies.ParallelDataStaging import *
from policies.LBANNInMemoryStore import *
from policies.LocalityAwareDataLoading import *
from policies.BasePolicy import *


# Define various parameters for the simulation.
num_nodes = 4
compute_throughput = 64*5
preproc_throughput = 200*5
bandwidth = 24000
prefetch_threads = {
    0: 8,
    1: 4,
    2: 2
}
storage_capacity = {
    0: 5120
}
pfs_bandwidth = {
    1: 338,
    2: 727,
    4: 1540,
    8: 2871,
    16: 4787,
    32: 5266
}
storage_classes = {
    0: {  # Staging buffer
        1: 28000,
        2: 50450,
        4: 85800,
        8: 111300,
        16: 123700
    }
}
epochs = 5
drop_last_iter = True
node_distr_scheme = 'uniform'
aggregated_stats = True
seed = 42

imagenet22k_dataset = {
    'synthetic': True,
    'synthetic_params': {
        'distribution': scipy.stats.norm,
        'loc': 0.1077,
        'scale': 0.1,
        'samples': 14197103,
        'min': 0,
        'max': None
    }
}

policies = {
    "HDMLP": {'class': FrequencyPrefetcher, 'opts': {'in_order': True}},
    "Lower Bound": {'class': PerfectPrefetcher, 'opts': {}},
}

ds = Dataset(imagenet22k_dataset, epochs, 32*num_nodes, num_nodes,
             node_distr_scheme, drop_last_iter, seed)

for staging_buffer in [1024, 2048, 4096, 5120]:
    print('Staging buffer', staging_buffer)
    storage_capacity[0] = staging_buffer
    policy_executor = PolicyExecutor(policies, ds, num_nodes, storage_classes,
                                 prefetch_threads, pfs_bandwidth,
                                 storage_capacity, bandwidth, bandwidth,
                                 preproc_throughput, compute_throughput,
                                 aggregated_stats)
    statistics = policy_executor.run()
    Statistics.table_run_time(statistics, aggregated_stats)

del policies['Lower Bound']

storage_classes = {
    0: {  # Staging buffer
        1: 28000,
        2: 50450,
        4: 85800,
        8: 111300,
        16: 123700
    },
    1: {  # RAM
        1: 28000,
        2: 50450,
        4: 85800,
        8: 111300,
        16: 123700
    },
    2: {  # SSD
        1: 627,
        2: 1168,
        4: 2048,
        8: 3136,
        16: 4425,
        32: 5338
    }
}

for mem in [0, 32, 64, 128, 256, 512]:
    mem *= 1024
    for ssd in [0, 128, 256, 512, 1024, 2048]:
        ssd *= 1024
        storage_capacity = {
            0: 5120,
            1: mem,
            2: ssd
        }
        print('Mem', mem // 1024, 'SSD', ssd // 1024)
        policy_executor = PolicyExecutor(policies, ds, num_nodes, storage_classes,
                                 prefetch_threads, pfs_bandwidth,
                                 storage_capacity, bandwidth, bandwidth,
                                 preproc_throughput, compute_throughput,
                                 aggregated_stats)
        statistics = policy_executor.run()
        Statistics.table_run_time(statistics, aggregated_stats)
