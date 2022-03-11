"""Train the CosmoFlow model."""

import argparse
import random
import os
import os.path
import time
import shutil
import pickle

import torch
import apex
import yaml

from model import CosmoFlowModel
from utils import (Logger, AverageTracker, AverageTrackerDevice,
                   get_cosmoflow_lr_schedule,
                   get_num_gpus, get_local_rank, get_local_size,
                   get_world_rank, get_world_size, initialize_dist,
                   get_cuda_device)
from data import (CosmoFlowDataset, CosmoFlowDatasetBinary,
                  CosmoFlowTransform, PrefetchWrapper,
                  RandomDataset)

try:
    import hdmlp
    import hdmlp.lib.torch
except ImportError:
    hdmlp = None


SUPPORTED_MODELS = {
    'cosmoflow': CosmoFlowModel
}
SUPPORTED_ACTS = {
    # Keras default slope for LeakyReLU is 0.3.
    'leakyrelu': lambda: torch.nn.LeakyReLU(negative_slope=0.3)
}
SUPPORTED_POOLS = {
    'maxpool': torch.nn.MaxPool3d
}
SUPPORTED_OPTIMIZERS = {
    'sgd': apex.optimizers.FusedSGD
}
SUPPORTED_LOSSES = {
    'mse': torch.nn.MSELoss
}


def get_args():
    """Get arguments from command line and config file."""
    parser = argparse.ArgumentParser(description='CosmoFlow training')
    parser.add_argument('config', type=str, nargs='?',
                        help='Configuration file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--job-id', type=str, required=True,
                        help='Job identifier')
    parser.add_argument('--print-freq', type=int, default=None,
                        help='Frequency for printing batch info')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from given checkpoint')
    parser.add_argument('--resume-epoch', type=int, default=None,
                        help='Epoch to resume training from')
    parser.add_argument('--no-eval', default=False, action='store_true',
                        help='Do not evaluate on validation set each epoch')
    parser.add_argument('--save-stats', default=False, action='store_true',
                        help='Save all performance statistics to files')

    # Training details.
    parser.add_argument('--data-dir', type=str,
                        help='Path to input data')
    parser.add_argument('--dataset', type=str, default='cosmoflow',
                        choices=['cosmoflow', 'rand'],
                        help='Dataset to use')
    parser.add_argument('--bin-data', default=False, action='store_true',
                        help='Use binary instead of HDF5 data')
    parser.add_argument('--n-train', type=int,
                        help='Number of training samples (-1 to infer)')
    parser.add_argument('--n-valid', type=int,
                        help='Number of validation samples (-1 to infer)')
    parser.add_argument('--drop-last', default=False, action='store_true',
                        help='Drop last small mini-batch')
    parser.add_argument('--input-shape', type=int, nargs='+',
                        help='Shape of input data (CHWD format)')
    parser.add_argument('--output-shape', type=int,
                        help='Shape of output data')
    parser.add_argument('--apply-log', default=None, action='store_true',
                        help='Log-transform data')
    parser.add_argument('--batch-size', type=int,
                        help='Per-GPU batch size')
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs')
    parser.add_argument('--loss', type=str, choices=['mse'],
                        help='Loss function')
    parser.add_argument('--seed', type=int,
                        help='Random seed')

    # Model details.
    parser.add_argument('--model', type=str,
                        help='Model to use')
    parser.add_argument('--num-convs', type=int,
                        help='Number of convolutional layers')
    parser.add_argument('--conv-channels', type=int,
                        help='Base number of convolution channels')
    parser.add_argument('--kernel-size', type=int,
                        help='Size of convolution kernel')
    parser.add_argument('--fc1-size', type=int,
                        help='Number of neurons in first FC layer')
    parser.add_argument('--fc2-size', type=int,
                        help='Number of neurons in second FC layer')
    parser.add_argument('--act', type=str,
                        help='Activation function')
    parser.add_argument('--pool', type=str,
                        help='Pooling operation')
    parser.add_argument('--dropout', type=float,
                        help='Dropout rate')

    # Optimization.
    parser.add_argument('--optimizer', type=str, choices=['sgd'],
                        help='Optimizer to use')
    parser.add_argument('--lr', type=float,
                        help='Learning rate')
    parser.add_argument('--base-lr', type=float,
                        help='Starting learning rate')
    parser.add_argument('--momentum', type=float,
                        help='Momentum for SGD')
    parser.add_argument('--warmup-epochs', type=int,
                        help='Number of epochs to warm up')
    parser.add_argument('--decay-epochs', type=int, nargs='+',
                        help='Epochs to decay learning rate at')
    parser.add_argument('--decay-factors', type=float, nargs='+',
                        help='Factors to decay by at given epochs')

    # Performance settings.
    parser.add_argument('--dist', action='store_true',
                        help='Do distributed training')
    parser.add_argument('-r', '--rendezvous', type=str, default='file',
                        help='Distributed rendezvous scheme (file, tcp)')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16/AM training')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of workers for reading samples')
    parser.add_argument('--no-cudnn-bm', action='store_true',
                        help='Do not do benchmarking to select cuDNN algos')
    parser.add_argument('--no-prefetch', action='store_true',
                        help='Do not use fast prefetch pipeline')
    parser.add_argument('--hdmlp', action='store_true',
                        help='Use HDMLP')
    parser.add_argument('--hdmlp-config-path', type=str,
                        help='Config path for HDMLP')
    parser.add_argument('--hdmlp-lib-path', type=str,
                        help='Library path for HDMLP')
    parser.add_argument('--hdmlp-stats', default=False, action='store_true',
                        help='Save HDMLP statistics every epoch')
    parser.add_argument('--channels-last', action='store_true',
                        help='Use channels-last memory order')

    return parser.parse_args()


def load_config(args):
    """Load the config file and apply it to args.

    args will be updated in-place.

    """
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = {'data': {}, 'train': {}, 'model': {}, 'optimizer': {}}

    if args.data_dir is None:
        args.data_dir = config['data']['data_dir']
    if args.n_train is None:
        if 'n_train' in config['data']:
            args.n_train = config['data']['n_train']
        else:
            args.n_train = -1
    if args.n_valid is None:
        if 'n_valid' in config['data']:
            args.n_valid = config['data']['n_valid']
        else:
            args.n_valid = -1
    if args.input_shape is None:
        args.input_shape = config['data']['input_shape']
    if args.output_shape is None:
        args.output_shape = config['data']['output_shape']
    if args.apply_log is None:
        args.apply_log = config['data']['apply_log']
    if args.batch_size is None:
        args.batch_size = config['data']['batch_size']
    if args.epochs is None:
        args.epochs = config['data']['epochs']
    if args.loss is None:
        args.loss = config['train']['loss']
    if args.seed is None:
        args.seed = config['train']['seed']
    if args.model is None:
        args.model = config['model']['model']
    args.model = args.model.lower()
    if args.num_convs is None:
        args.num_convs = config['model']['num_convs']
    if args.conv_channels is None:
        args.conv_channels = config['model']['conv_channels']
    if args.kernel_size is None:
        args.kernel_size = config['model']['kernel_size']
    if args.fc1_size is None:
        args.fc1_size = config['model']['fc1_size']
    if args.fc2_size is None:
        args.fc2_size = config['model']['fc2_size']
    if args.act is None:
        args.act = config['model']['act']
    args.act = args.act.lower()
    if args.pool is None:
        args.pool = config['model']['pool']
    args.pool = args.pool.lower()
    if args.dropout is None:
        args.dropout = config['model']['dropout']
    if args.optimizer is None:
        args.optimizer = config['optimizer']['optimizer']
    args.optimizer = args.optimizer.lower()
    if args.lr is None:
        args.lr = config['optimizer']['lr']
    if args.base_lr is None:
        args.base_lr = config['optimizer']['base_lr']
    if args.momentum is None:
        args.momentum = config['optimizer']['momentum']
    if args.warmup_epochs is None:
        args.warmup_epochs = config['optmizer']['warmup_epochs']
    if args.decay_epochs is None:
        args.decay_epochs = config['optimizer']['decay_epochs']
    if args.decay_factors is None:
        args.decay_factors = config['optimizer']['decay_factors']


def main():
    """Main CosmoFlow training."""
    args = get_args()
    load_config(args)

    # Determine whether this is the primary rank.
    args.primary = get_world_rank(required=args.dist) == 0

    # Seed RNGs.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.dist:
        if get_local_size() > get_num_gpus():
            raise RuntimeError(
                'Do not use more ranks per node than there are GPUs')
        initialize_dist(f'./init_{args.job_id}', args.rendezvous)
    else:
        if get_world_size() > 1:
            print('Multiple processes detected, but --dist not passed',
                  flush=True)

    if not args.no_cudnn_bm:
        torch.backends.cudnn.benchmark = True
    if args.channels_last:
        raise RuntimeError('channels_last not supported for 3D conv. :(')

    # Set up output directory.
    if args.primary:
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)

    if args.hdmlp and not hdmlp:
        raise RuntimeError('HDMLP not available')

    log = Logger(os.path.join(args.output_dir, f'log_{args.job_id}.txt'),
                 args.primary)

    # Set up model.
    if args.model not in SUPPORTED_MODELS:
        raise ValueError(f'Unknown model {args.model}')
    if args.act not in SUPPORTED_ACTS:
        raise ValueError(f'Unknown activation {args.act}')
    if args.pool not in SUPPORTED_POOLS:
        raise ValueError(f'Unknown pool type {args.pool}')
    if args.optimizer not in SUPPORTED_OPTIMIZERS:
        raise ValueError(f'Unknown optimizer {args.optimizer}')
    if args.loss not in SUPPORTED_LOSSES:
        raise ValueError(f'Unknown loss {args.loss}')
    net = SUPPORTED_MODELS[args.model](
        args.input_shape, args.output_shape,
        conv_channels=args.conv_channels,
        kernel_size=args.kernel_size,
        n_conv_layers=args.num_convs,
        fc1_size=args.fc1_size, fc2_size=args.fc2_size,
        act=SUPPORTED_ACTS[args.act],
        pool=SUPPORTED_POOLS[args.pool],
        dropout=args.dropout)
    net = net.to(get_cuda_device())
    if args.dist:
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[get_local_rank()],
            output_device=get_local_rank(),
            find_unused_parameters=True)
    # Gradient scaler for FP16.
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    # Loss.
    criterion = SUPPORTED_LOSSES[args.loss]()
    # Optimizer.
    optimizer = SUPPORTED_OPTIMIZERS[args.optimizer](
        net.parameters(), args.lr, momentum=args.momentum)
    # Learning rate scheduler.
    scheduler = get_cosmoflow_lr_schedule(
        optimizer, args.base_lr, args.lr, args.warmup_epochs,
        args.decay_epochs, args.decay_factors)
    # Record the current best validation loss during training.
    best_loss = float('inf')

    # Load checkpoint if needed.
    if args.resume:
        # Attempt to determine the checkpoint location.
        # First check if we have a valid path, then check the checkpoint dir.
        checkpoint_name = args.resume
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                args.output_dir, 'checkpoints', checkpoint_name)
            if not os.path.exists(checkpoint_name):
                raise ValueError(f'Cannot load checkpoint {args.resume}')
        checkpoint = torch.load(checkpoint_name,
                                map_location=get_cuda_device())
        if args.resume_epoch is None:
            # + 1 since we checkpoint at the end of each epoch.
            args.resume_epoch = checkpoint['epoch'] + 1
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        best_loss = checkpoint['best_loss']
        log.log(f'==>> Loaded checkpoint {args.resume} '
                f'from epoch {checkpoint["epoch"]} '
                f'with best loss {best_loss}')
        log.log(f'==>> Resuming at epoch {args.resume_epoch}')

    # Set up data loaders.
    if args.hdmlp:
        if args.hdmlp_stats:
            os.environ['HDMLPPROFILING'] = '1'

        if args.dataset != 'cosmoflow':
            raise RuntimeError('Only CosmoFlow dataset supported with HDMLP')
        if args.bin_data:
            hdmlp_backend = 'cosmoflow'
        else:
            hdmlp_backend = 'hdf5'
        hdmlp_train_job = hdmlp.Job(
            os.path.join(args.data_dir, 'train'),
            args.batch_size * get_world_size(),
            args.epochs,
            'uniform',
            args.drop_last,
            seed=args.seed,
            config_path=args.hdmlp_config_path,
            libhdmlp_path=args.hdmlp_lib_path,
            filesystem_backend=hdmlp_backend,
            hdf5_data_name='full',
            hdf5_target_name='unitPar',
            collate_data=True)
        if not args.no_eval:
            hdmlp_validation_job = hdmlp.Job(
                os.path.join(args.data_dir, 'validation'),
                args.batch_size * get_world_size(),
                args.epochs,
                'uniform',
                args.drop_last,
                seed=args.seed,
                config_path=args.hdmlp_config_path,
                libhdmlp_path=args.hdmlp_lib_path,
                filesystem_backend=hdmlp_backend,
                hdf5_data_name='full',
                hdf5_target_name='unitPar',
                collate_data=True)
        transform = CosmoFlowTransform(args.apply_log)
        if args.bin_data:
            train_dataset = hdmlp.lib.torch.HDMLPCosmoFlow(
                os.path.join(args.data_dir, 'train'),
                hdmlp_train_job,
                sample_shape=args.input_shape,
                label_shape=(args.output_shape,))
            if not args.no_eval:
                validation_dataset = hdmlp.lib.torch.HDMLPCosmoFlow(
                    os.path.join(args.data_dir, 'validation'),
                    hdmlp_validation_job,
                    sample_shape=args.input_shape,
                    label_shape=(args.output_shape,))
        else:
            train_dataset = hdmlp.lib.torch.HDMLPHDF5(
                os.path.join(args.data_dir, 'train'),
                hdmlp_train_job,
                sample_shape=args.input_shape,
                label_shape=(args.output_shape,))
            if not args.no_eval:
                validation_dataset = hdmlp.lib.torch.HDMLPHDF5(
                    os.path.join(args.data_dir, 'validation'),
                    hdmlp_validation_job,
                    sample_shape=args.input_shape,
                    label_shape=(args.output_shape,))
        train_loader = hdmlp.lib.torch.HDMLPDataLoader(
            train_dataset, label_type=torch.float)
        if not args.no_eval:
            validation_loader = hdmlp.lib.torch.HDMLPDataLoader(
                validation_dataset, label_type=torch.float)
    else:
        transform = CosmoFlowTransform(args.apply_log)
        dataset_transform = transform if args.no_prefetch else None
        if args.dataset == 'cosmoflow':
            n_train = args.n_train if args.n_train >= 0 else None
            n_valid = args.n_valid if args.n_valid >= 0 else None
            if args.bin_data:
                train_dataset = CosmoFlowDatasetBinary(
                    os.path.join(args.data_dir, 'train'),
                    args.input_shape,
                    dataset_size=n_train,
                    transform=dataset_transform)
                if not args.no_eval:
                    validation_dataset = CosmoFlowDatasetBinary(
                        os.path.join(args.data_dir, 'validation'),
                        args.input_shape,
                        dataset_size=n_valid,
                        transform=dataset_transform)
            else:
                train_dataset = CosmoFlowDataset(
                    os.path.join(args.data_dir, 'train'),
                    dataset_size=n_train,
                    transform=dataset_transform)
                if not args.no_eval:
                    validation_dataset = CosmoFlowDataset(
                        os.path.join(args.data_dir, 'validation'),
                        dataset_size=n_valid,
                        transform=dataset_transform)
        elif args.dataset == 'rand':
            if args.n_train == -1:
                raise ValueError('Must specify n_train for rand dataset')
            train_dataset = RandomDataset(
                args.input_shape, (args.output_shape,), args.n_train,
                transform=dataset_transform)
            if not args.no_eval:
                validation_dataset = RandomDataset(
                    args.input_shape, (args.output_shape,), args.n_valid,
                    transform=dataset_transform)
        if args.dist:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=get_world_size(),
                rank=get_world_rank())
            if not args.no_eval:
                validation_sampler = torch.utils.data.distributed.DistributedSampler(
                    validation_dataset, num_replicas=get_world_size(),
                    rank=get_world_rank())
        else:
            train_sampler = torch.utils.data.RandomSampler(train_dataset)
            if not args.no_eval:
                validation_sampler = torch.utils.data.RandomSampler(validation_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            num_workers=args.workers, pin_memory=True,
            sampler=train_sampler, drop_last=args.drop_last)
        if not args.no_eval:
            validation_loader = torch.utils.data.DataLoader(
                validation_dataset, batch_size=args.batch_size,
                num_workers=args.workers, pin_memory=True,
                sampler=validation_sampler, drop_last=args.drop_last)
    if not args.no_prefetch:
        # Wrap data loaders.
        train_loader = PrefetchWrapper(train_loader, transform)
        if not args.no_eval:
            validation_loader = PrefetchWrapper(validation_loader, transform)

    # Estimate reasonable print frequency as every 5%.
    if args.print_freq is None:
        args.print_freq = len(train_loader) // 20

    # Log training configuration.
    log.log(str(args))
    log.log(str(net))
    log.log(f'Using {get_world_size()} processes')
    log.log(f'Global batch size is {args.batch_size*get_world_size()}'
            f' ({args.batch_size} per GPU)')
    log.log(f'Training data size: {len(train_dataset)}')
    if args.no_eval:
        log.log('No validation')
    else:
        log.log(f'Validation data size: {len(validation_dataset)}')

    # Train.
    log.log('Starting training at ' +
            time.strftime('%Y-%m-%d %X', time.gmtime(time.time())))
    epoch_times = AverageTracker()
    train_start_time = time.perf_counter()
    if args.resume_epoch is None:
        args.resume_epoch = 0
    for epoch in range(args.resume_epoch, args.epochs):
        start_time = time.perf_counter()
        if args.dist and hasattr(train_loader, 'sampler'):
            train_loader.sampler.set_epoch(epoch)
            if not args.no_eval:
                validation_loader.sampler.set_epoch(epoch)
        log.log(f'==>> Epoch={epoch:03d}/{args.epochs:03d} '
                f'Elapsed={int(time.perf_counter()-train_start_time)} s '
                f'(avg epoch time: {epoch_times.mean():5.3f} s'
                f', current epoch: {epoch_times.latest():5.3f} s) '
                f'[learning_rate={scheduler.get_last_lr()[0]:6.4f}]')
        train(args, train_loader, net, scaler, criterion, optimizer, log)
        if not args.no_eval:
            val_loss = validate(args, validation_loader, net, criterion, log)
            is_best = False
            if val_loss < best_loss:
                is_best = True
                best_loss = val_loss
                log.log(f'New best loss: {best_loss:.5f}')
        scheduler.step()

        if not args.no_eval:
            # Preserve behavior of only checkpointing when doing validation.
            save_checkpoint(args, epoch, net, optimizer, scheduler, scaler,
                            best_loss, is_best)

        epoch_times.update(time.perf_counter() - start_time)

        # Store HDMLP statistics every epoch
        if args.primary and args.hdmlp and args.hdmlp_stats:
            stat_path = os.path.join(args.output_dir,
                                     f'stats_hdmlp_{args.job_id}.pkl')
            with open(stat_path, 'wb') as f:
                pickle.dump(hdmlp_train_job.get_metrics(), f)
    log.log(f'==>> Done Elapsed={int(time.perf_counter()-train_start_time)} s '
            f'(avg epoch time: {epoch_times.mean():5.3f} s '
            f'current epoch: {epoch_times.latest():5.3f} s)')


def save_checkpoint(args, cur_epoch, net, optimizer, scheduler, scaler,
                    best_loss, is_best):
    """Save a checkpoint of the current state."""
    # Only the primary saves.
    if not args.primary:
        return
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    state = {
        'epoch': cur_epoch,
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'best_loss': best_loss
    }
    checkpoint_name = os.path.join(checkpoint_dir, f'checkpoint_{cur_epoch}')
    torch.save(state, checkpoint_name)
    if is_best:
        shutil.copyfile(checkpoint_name,
                        os.path.join(checkpoint_dir, 'checkpoint_best'))


def train(args, train_loader, net, scaler, criterion, optimizer, log):
    """Perform one epoch of training."""
    net.train()
    losses = AverageTrackerDevice(len(train_loader), get_cuda_device(),
                                  allreduce=args.dist)
    maes = AverageTrackerDevice(len(train_loader), get_cuda_device(),
                                allreduce=args.dist)
    batch_times = AverageTracker()
    data_times = AverageTracker()
    end_time = time.perf_counter()
    for batch, (samples, targets) in enumerate(train_loader):
        samples = samples.to(get_cuda_device(), non_blocking=True)
        targets = targets.to(get_cuda_device(), non_blocking=True)
        data_times.update(time.perf_counter() - end_time)

        with torch.cuda.amp.autocast(enabled=args.fp16):
            output = net(samples)
            loss = criterion(output, targets)
            with torch.no_grad():
                mae = torch.nn.functional.l1_loss(output, targets)

        losses.update(loss, samples.size(0))
        maes.update(mae, samples.size(0))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        batch_times.update(time.perf_counter() - end_time)
        end_time = time.perf_counter()

        if batch % args.print_freq == 0 and batch != 0:
            log.log(f'    [{batch}/{len(train_loader)}] '
                    f'Avg loss: {losses.mean():.5f} '
                    f'Avg MAE: {maes.mean():.5f} '
                    f'Avg time/batch: {batch_times.mean():.3f} s '
                    f'Avg data fetch time/batch: {data_times.mean():.3f} s')
    log.log(f'    **Train** Loss {losses.mean():.5f} | MAE {maes.mean():.5f}')
    if args.primary and args.save_stats:
        batch_times.save(os.path.join(
            args.output_dir, f'stats_batch_{args.job_id}.csv'))
        data_times.save(os.path.join(
            args.output_dir, f'stats_data_{args.job_id}.csv'))
    return losses.mean()


def validate(args, validation_loader, net, criterion, log):
    """Validate on the given dataset."""
    net.eval()
    losses = AverageTrackerDevice(len(validation_loader), get_cuda_device(),
                                  allreduce=args.dist)
    maes = AverageTrackerDevice(len(validation_loader), get_cuda_device(),
                                allreduce=args.dist)
    with torch.no_grad():
        for samples, targets in validation_loader:
            samples = samples.to(get_cuda_device(), non_blocking=True)
            targets = targets.to(get_cuda_device(), non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                output = net(samples)
                loss = criterion(output, targets)
                mae = torch.nn.functional.l1_loss(output, targets)

            losses.update(loss, samples.size(0))
            maes.update(mae, samples.size(0))
    log.log(f'    **Val** Loss {losses.mean():.5f} | MAE {maes.mean():.5f}')
    return losses.mean()


if __name__ == '__main__':
    main()

    # Skip teardown to avoid hangs.
    os._exit(0)
