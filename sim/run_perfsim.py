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
compute_throughput = 64
preproc_throughput = 200
bandwidth = 24000
prefetch_threads = {
    0: 8,
    1: 4,
    2: 2
}
storage_capacity = {
    0: 5120,
    1: 133120,
    2: 1024*1024
}
pfs_bandwidth = {
    1: 338 ,
    2: 7272,
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
epochs = 5
drop_last_iter = True
node_distr_scheme = 'uniform'
aggregated_stats = True
seed = 42

policies = {
    "Naive": {'class': BasePolicy, 'opts': {}},
    "Staging Buffer": {'class': StagingPoolPrefetcher, 'opts': {}},
    "DeepIO (Ord.)": {'class': DeepIOPolicy, 'opts': {'opportunistic_reordering': False}},
    "DeepIO (Opp.)": {'class': DeepIOPolicy, 'opts': {'opportunistic_reordering': True}},
    "Parallel Staging": {'class': ParallelStagingPolicy, 'opts': {}},
    "LBANN (Dynamic)": {'class': LBANNInMemoryStorePolicy, 'opts': {'preloading': False}},
    "LBANN (Preloading)": {'class': LBANNInMemoryStorePolicy, 'opts': {'preloading': True}},
    "Locality-Aware": {'class': LocalityAwareDataLoadingPolicy, 'opts': {}},
    "HDMLP": {'class': FrequencyPrefetcher, 'opts': {'in_order': True}},
    "Lower Bound": {'class': PerfectPrefetcher, 'opts': {}},
}

# Define datasets.
mnist_dataset = {
    'synthetic': True,
    'synthetic_params': {
        'distribution': None,
        'loc': None,
        'scale': None,
        'samples': 50000,
        'min': 28*28 / 1024 / 1024,
        'max': 28*28 / 1024 / 1024
    }
}

imagenet1k_dataset = {
    'synthetic': True,
    'synthetic_params': {
        'distribution': scipy.stats.norm,
        'loc': 0.1077,
        'scale': 0.1,
        'samples': 1281167,
        'min': 0,
        'max': None
    }
}

openimages_dataset = {
    'synthetic': True,
    'synthetic_params': {
        'distribution': scipy.stats.norm,
        'loc': 0.2937,
        'scale': 0.2,
        'samples': 1743042,
        'min': 0,
        'max': None
    }
}

places_dataset = {
    'synthetic': True,
    'synthetic_params': {
        'distribution': scipy.stats.norm,
        'loc': 0.0595,
        'scale': 0.2,
        'samples': 8000000,
        'min': 0,
        'max': None
    }
}

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

cosmoflow128_dataset = {
    'synthetic': True,
    'synthetic_params': {
        'distribution': None,
        'loc': None,
        'scale': None,
        'samples': 262144,
        'min': 4*128**3 / 1024 / 1024,
        'max': 4*128**3 / 1024 / 1024
    }
}

cosmoflow512_dataset = {
    'synthetic': True,
    'synthetic_params': {
        'distribution': None,
        'loc': None,
        'scale': None,
        'samples': 10018,
        'min': 4*512**3 / 1024 / 1024,
        'max': 4*512**3 / 1024 / 1024
    }
}

# Directory to save results to.
os.makedirs('data', exist_ok=True)

print('MNIST')
ds = Dataset(mnist_dataset, epochs, 32*num_nodes, num_nodes, node_distr_scheme,
             drop_last_iter, seed)
policy_executor = PolicyExecutor(policies, ds, num_nodes, storage_classes,
                                 prefetch_threads, pfs_bandwidth,
                                 storage_capacity, bandwidth, bandwidth,
                                 preproc_throughput, compute_throughput,
                                 aggregated_stats)
statistics = policy_executor.run()
Statistics.create_plots(statistics, aggregated_stats, 'perfsim_mnist_stats')

print('ImageNet-1k')
ds = Dataset(imagenet1k_dataset, epochs, 32*num_nodes, num_nodes,
             node_distr_scheme, drop_last_iter, seed)
policy_executor = PolicyExecutor(policies, ds, num_nodes, storage_classes,
                                 prefetch_threads, pfs_bandwidth,
                                 storage_capacity, bandwidth, bandwidth,
                                 preproc_throughput, compute_throughput,
                                 aggregated_stats)
statistics = policy_executor.run()
Statistics.create_plots(statistics, aggregated_stats, 'perfsim_imagenet1k_stats')

print('ImageNet-22k')
ds = Dataset(imagenet22k_dataset, epochs, 32*num_nodes, num_nodes,
             node_distr_scheme, drop_last_iter, seed)
policy_executor = PolicyExecutor(policies, ds, num_nodes, storage_classes,
                                 prefetch_threads, pfs_bandwidth,
                                 storage_capacity, bandwidth, bandwidth,
                                 preproc_throughput, compute_throughput,
                                 aggregated_stats)
statistics = policy_executor.run()
Statistics.create_plots(statistics, aggregated_stats, 'perfsim_imagenet22k_stats')

print('OpenImages')
ds = Dataset(openimages_dataset, epochs, 32*num_nodes, num_nodes,
             node_distr_scheme, drop_last_iter, seed)
policy_executor = PolicyExecutor(policies, ds, num_nodes, storage_classes,
                                 prefetch_threads, pfs_bandwidth,
                                 storage_capacity, bandwidth, bandwidth,
                                 preproc_throughput, compute_throughput,
                                 aggregated_stats)
statistics = policy_executor.run()
Statistics.create_plots(statistics, aggregated_stats, 'perfsim_openimages_stats')

print('CosmoFlow 128')
ds = Dataset(cosmoflow128_dataset, epochs, 16*num_nodes, num_nodes,
             node_distr_scheme, drop_last_iter, seed)
policy_executor = PolicyExecutor(policies, ds, num_nodes, storage_classes,
                                 prefetch_threads, pfs_bandwidth,
                                 storage_capacity, bandwidth, bandwidth,
                                 preproc_throughput, compute_throughput,
                                 aggregated_stats)
statistics = policy_executor.run()
Statistics.create_plots(statistics, aggregated_stats, 'perfsim_cosmoflow128_stats')

print('CosmoFlow 512')
num_nodes = 32
ds = Dataset(cosmoflow512_dataset, epochs, num_nodes, num_nodes,
             node_distr_scheme, drop_last_iter, seed)
policy_executor = PolicyExecutor(policies, ds, num_nodes, storage_classes,
                                 prefetch_threads, pfs_bandwidth,
                                 storage_capacity, bandwidth, bandwidth,
                                 preproc_throughput, compute_throughput,
                                 aggregated_stats)
statistics = policy_executor.run()
Statistics.create_plots(statistics, aggregated_stats, 'perfsim_cosmoflow512_stats')
