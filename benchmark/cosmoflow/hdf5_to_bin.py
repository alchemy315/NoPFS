"""Convert split HDF5 CosmoFlow files to raw binary."""

import argparse
import glob
import os
import os.path
import multiprocessing
import functools
import pickle

import numpy as np
import h5py
import tqdm


parser = argparse.ArgumentParser(
    description='Convert split HDF5 CosmoFlow files to raw binary.')
parser.add_argument('data_dir', type=str,
                    help='Directory containing CosmoFlow HDF5 files')
parser.add_argument('out_dir', type=str,
                    help='Directory to output data')
parser.add_argument('--ntasks', type=int, default=8,
                    help='Number of tasks to use for processing (defaukt: 8)')


def list_files(data_dir):
    """Recursively list all HDF5 files in data_dir."""
    return glob.glob(data_dir + '/**/*.hdf5', recursive=True)


def make_index_file(out_dir, files):
    """Write an index file."""
    files = [os.path.splitext(os.path.basename(x))[0] for x in files]
    with open(os.path.join(out_dir, 'idx'), 'wb') as f:
        pickle.dump(files, f, pickle.HIGHEST_PROTOCOL)


def read_file(filename):
    """Load universe and target from filename."""
    with h5py.File(filename, 'r') as f:
        return f['full'][:], f['unitPar'][:]


def write_file(filename, x, y):
    """Write x and y to filename."""
    with open(filename, 'wb') as f:
        y.tofile(f)
        x.tofile(f)


def process_file(filename, out_dir):
    """Convert HDF5 filename to binary."""
    x, y = read_file(filename)
    out_filename = os.path.join(
        out_dir,
        os.path.splitext(os.path.basename(filename))[0]) + '.bin'
    write_file(out_filename, x, y)


def process_all_files(args):
    """Convert all files."""
    if not os.path.isdir(args.data_dir):
        raise ValueError(f'Bad data-dir: {args.data_dir}')
    os.makedirs(args.out_dir)
    files = list_files(args.data_dir)
    make_index_file(args.out_dir, files)
    with multiprocessing.Pool(processes=args.ntasks) as pool:
        for _ in tqdm.tqdm(pool.imap(functools.partial(
                process_file, out_dir=args.out_dir),
                                     files),
                           total=len(files)):
            pass


if __name__ == '__main__':
    process_all_files(parser.parse_args())
