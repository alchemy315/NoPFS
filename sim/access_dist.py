"""Plot access distribution for a dataset.

Adapted from the original script by Roman Röhringer.

"""

import argparse

import numpy as np
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


sns.set()

parser = argparse.ArgumentParser(
    description='Plot access distribution')
parser.add_argument('--epochs', type=int, required=True,
                    help='Number of epochs')
parser.add_argument('--workers', type=int, required=True,
                    help='Number of workers')
parser.add_argument('--size', type=int, required=True,
                    help='Number of samples in dataset')
parser.add_argument('--delta', type=float, default=0.1,
                    help='Factor over expected value')
parser.add_argument('--plot-file', type=str, default='access_freq.pdf',
                    help='Filename to save frequency plot to')
parser.add_argument('--node', type=int, default=0,
                    help='Node to plot for')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed')
parser.add_argument('--num-bins', type=int, default=100,
                    help='Number of histogram bins')


def plot_access_frequency(args, counts):
    """Plot generated access frequency."""
    fig, ax = plt.subplots()
    hist, bins = np.histogram(counts, bins=args.num_bins)
    sns.histplot(x=bins[:-1], weights=hist, bins=args.num_bins, ax=ax)
    ax.get_xaxis().set_major_locator(
        matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel('Access frequency')
    ax.set_ylabel('# Samples')
    fig.tight_layout()
    fig.savefig(args.plot_file)


def print_stats(args, counts):
    """Print stats for generated access frequency."""
    thresh = args.epochs*(1/args.workers)*(1 + args.delta)
    # Estimated access counts.
    est_counts = (
        args.size
        * (1 - scipy.stats.binom.cdf(thresh, args.epochs, 1/args.workers)))
    # Actual access counts (per simulation).
    actual_counts = int(np.sum(counts > thresh))
    print('Average access count:', args.epochs*(1/args.workers))
    print('Thresh:', thresh)
    print('Estimated #samples exceeding thresh:', int(np.round(est_counts)))
    print('Actual #samples exceeding thresh:', int(np.round(actual_counts)))


def simulate_access_frequency(args):
    """Run access frequency simulation."""
    rng = np.random.default_rng(args.seed)
    file_ids = np.arange(args.size)
    # Contains for each node the number of times each sample is accessed.
    counts = np.zeros((args.workers, args.size), dtype=np.int64)
    for _ in range(args.epochs):
        rng.shuffle(file_ids)
        # Assign samples to workers.
        node_ids = np.resize(np.arange(args.workers), args.size)
        for i in range(args.workers):
            # Update counts for each sample worker i accesses this epoch.
            counts[i, file_ids[node_ids == i]] += 1
    plot_access_frequency(args, counts[args.node])
    print_stats(args, counts[args.node])


if __name__ == '__main__':
    simulate_access_frequency(parser.parse_args())
