"""Datasets for CosmoFlow."""

import pickle
import os.path
import functools
import operator
import glob

import torch
import numpy as np

try:
    import h5py
except:
    h5py = None


class CosmoFlowDataset(torch.utils.data.Dataset):
    """Cosmoflow data."""

    SUBDIR_FORMAT = '{:03d}'

    def __init__(self, data_dir, dataset_size=None,
                 transform=None, transform_y=None, base_universe_size=512):
        """Set up the CosmoFlow HDF5 dataset.

        This expects pre-split universes per split_hdf5_cosmoflow.py.

        You may need to transpose the universes to make the channel
        dimension be first. It is up to you to do this in the
        transforms or preprocessing.

        The sample will be provided to transforms in int16 format.
        The target will be provided to transforms in float format.

        """
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.transform_y = transform_y

        if h5py is None:
            raise ImportError('HDF5 dataset requires h5py')

        # Load info from cached index.
        idx_filename = os.path.join(data_dir, 'idx')
        if not os.path.exists(idx_filename):
            if torch.distributed.get_rank() == 0:
                CosmoFlowDataset._make_index(
                    data_dir, base_universe_size=base_universe_size)
            # Wait for file to be created.
            torch.distributed.barrier()
        with open(idx_filename, 'rb') as f:
            idx_data = pickle.load(f)
        self.sample_base_filenames = idx_data['filenames']
        self.num_subdirs = idx_data['num_subdirs']
        self.num_splits = (base_universe_size // idx_data['split_size'])**3

        self.num_samples = len(self.sample_base_filenames) * self.num_splits
        if dataset_size is not None:
            self.num_samples = min(dataset_size, self.num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Handle flat vs nested directory structure.
        base_index = index // self.num_splits  # Base filename.
        split_index = index % self.num_splits  # Split within the universe.
        if self.num_subdirs:
            subdir = CosmoFlowDataset.SUBDIR_FORMAT.format(
                base_index // self.num_subdirs)
            filename = os.path.join(
                self.data_dir,
                subdir,
                self.sample_base_filenames[base_index]
                + f'_{split_index:03d}.hdf5')
            x_idx = 'split'
        else:
            filename = os.path.join(
                self.data_dir,
                self.sample_base_filenames[base_index]
                + f'_{split_index:03d}.hdf5')
            x_idx = 'full'
        with h5py.File(filename, 'r') as f:
            x, y = f[x_idx][:], f['unitPar'][:]
        # Convert to Tensors.
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        if self.transform is not None:
            x = self.transform(x)
        if self.transform_y is not None:
            y = self.transform_y(y)
        return x, y

    @staticmethod
    def _make_index(data_dir, split_size=128, base_universe_size=512):
        """Generate an index file if a dataset does not have one."""
        print(f'Generating index file for {data_dir}', flush=True)
        subdirs = glob.glob(os.path.join(data_dir, '*', ''))
        if subdirs:
            raise RuntimeError(
                'Will not reconstruct index for subdir-based data')
        files = glob.glob(os.path.join(data_dir, '*.hdf5'))
        univs_per_subdir = 0
        # Identify the base universe names.
        univ_names = set(map(
            lambda x: os.path.splitext(os.path.basename(x))[0][:-4], files))
        data = {
            'split_size': split_size,
            'num_subdirs': univs_per_subdir,
            'filenames': list(univ_names)
        }
        with open(os.path.join(data_dir, 'idx'), 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


class CosmoFlowDatasetBinary(torch.utils.data.Dataset):
    """CosmoFlow data, stored in binary format."""

    def __init__(self, data_dir, input_shape, dataset_size=None,
                 transform=None, transform_y=None, base_universe_size=512):
        super().__init__()
        self.data_dir = data_dir
        self.input_shape = input_shape
        self.transform = transform
        self.transform_y = transform_y

        # Load info from cached index.
        idx_filename = os.path.join(data_dir, 'idx')
        if not os.path.exists(idx_filename):
            raise RuntimeError(f'Index file {idx_filename} not found')
        with open(idx_filename, 'rb') as f:
            self.filenames = pickle.load(f)
        self.num_samples = len(self.filenames)
        if dataset_size is not None:
            self.num_samples = min(dataset_size, self.num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        filename = os.path.join(
            self.data_dir, self.filenames[index]) + '.bin'
        with open(filename, 'rb') as f:
            y = np.fromfile(f, dtype=np.float32, count=4)
            x = np.fromfile(f, dtype=np.int16)
        x = x.reshape(self.input_shape)
        # Convert to Tensors.
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        if self.transform is not None:
            x = self.transform(x)
        if self.transform_y is not None:
            y = self.transform_y(y)
        return x, y


class CosmoFlowTransform:
    """Standard transformations for a single CosmoFlow sample."""

    def __init__(self, apply_log):
        """Set up the transform.

        apply_log: If True, log-transform the data, otherwise use
        mean normalization.

        """
        self.apply_log = apply_log

    def __call__(self, x):
        x = x.float()
        if self.apply_log:
            x.log1p_()
        else:
            x /= x.mean() / functools.reduce(operator.__mul__, x.size())
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


# Adapted from:
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/image_classification/dataloaders.py#L319
class PrefetchWrapper:
    """Prefetch ahead and perform preprocessing on GPU."""

    def __init__(self, data_loader, transform):
        self.data_loader = data_loader
        self.transform = transform
        # Will perform transforms on a separate CUDA stream.
        self.stream = torch.cuda.Stream()
        # Simplifies set_epoch.
        if hasattr(data_loader, 'sampler'):
            self.sampler = data_loader.sampler

    @staticmethod
    def prefetch_loader(data_loader, transform, stream):
        """Actual iterator for loading."""
        first = True
        sample, target = None, None
        for next_sample, next_target in data_loader:
            with torch.cuda.stream(stream):
                next_sample = next_sample.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                if transform is not None:
                    next_sample = transform(next_sample)

            if not first:
                yield sample, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            sample = next_sample
            target = next_target
        yield sample, target  # Last sample.

    def __iter__(self):
        return PrefetchWrapper.prefetch_loader(
            self.data_loader, self.transform, self.stream)

    def __len__(self):
        return len(self.data_loader)


class RandomDataset(torch.utils.data.Dataset):
    """Dataset that just returns a random tensor for debugging."""

    def __init__(self, sample_shape, target_shape, dataset_size,
                 transform=None):
        super().__init__()
        self.sample_shape = sample_shape
        self.target_shape = target_shape
        self.dataset_size = dataset_size
        self.transform = transform
        self.sample = torch.randint(0, 1000, sample_shape, dtype=torch.int16)
        self.target = torch.rand(target_shape)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        x = self.sample
        y = self.target
        if self.transform is not None:
            x = self.transform(x)
        return x, y
