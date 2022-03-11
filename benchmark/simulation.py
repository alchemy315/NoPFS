"""Train ResNet-50 quickly."""

import argparse
import random
import os
import os.path
import time
import statistics
import pickle
import warnings
import functools
import io

import torch
import torchvision
import apex
import numpy as np
import PIL

# try:
#     import hdmlp
#     import hdmlp.lib.torch
# except ImportError:
#     hdmlp = None

import sys
sys.path.append("/home/zzp/code/NoPFS/main")

import hdmlp
import hdmlp.lib.torch


def get_args():
    """Get arguments from the command line."""
    # argprase用于从命令行中传入参数
    parser = argparse.ArgumentParser(description='Train ResNet-50. Fast.') # 描述
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory') #type是要传入的参数的数据类型  help是该参数的提示信息
    
    '''
    如上可在命令行输入python resnet50.py 123,此时123即传入output-dir中(假设没有--)
    --表示该参数为可选参数，传入时需要python resnet50.py --output-dir = 123,如此可以增加可读性
    required表示该参数必须要
    '''

    parser.add_argument('--job-id', type=str, default=None,
                        help='Job identifier')
    parser.add_argument('--print-freq', type=int, default=None,
                        help='Frequency for printing batch info')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    '''default指默认值'''
    
    parser.add_argument('--no-eval', default=False, action='store_true',
                        help='Do not evaluate on validation set each epoch')

    '''
    action表示若给出了该变量(不用赋值),就将其置为true(action=store_true)/false(action=store_false)
    对布尔类型的参数有效.在没有使用该变量时使用default的值
    '''

    parser.add_argument('--save-stats', default=False, action='store_true',
                        help='Save all performance statistics to files')

    # Data/training details.
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing data to train with')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['imagenet', 'imagenet-22k'],
                        help='Dataset to use')

    '''choices表示取值只能为其中的值'''

    parser.add_argument('--synth-data', default=False, action='store_true',
                        help='Use synthetic data')
    parser.add_argument('--no-augmentation', default=False, action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--batch-size', type=int, default=120,
                        help='Per-GPU batch size')
    parser.add_argument('--epochs', type=int, default=90,
                        help='Number of epochs to train for')
    parser.add_argument('--drop-last', default=False, action='store_true',
                        help='Drop last small mini-batch')

    # Optimization.
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--start-lr', type=float, default=0.1,
                        help='Initial learning rate for warmup')
    parser.add_argument('--base-batch', type=int, default=256,
                        help='Base batch size for learning rate scaling')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum')
    parser.add_argument('--warmup-epochs', type=int, default=10,
                        help='Number of epochs to warm up')
    parser.add_argument('--decay-epochs', type=int, nargs='+',
                        default=[30, 60, 80],
                        help='Epochs at which to decay the learning rate')
    parser.add_argument('--decay-factors', type=float, nargs='+',
                        default=[0.1, 0.1, 0.1],
                        help='Factors by which to decay the learning rate')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='Label smoothing factor')
    parser.add_argument('--decay', type=float, default=0.0001,
                        help='L2 weight decay')

    # Performance.
    parser.add_argument('--dist', default=False, action='store_true',
                        help='Do distributed training')
    parser.add_argument('-r', '--rendezvous', type=str, default='file',
                        help='Distributed initialization scheme (file, tcp)')
    parser.add_argument('--fp16', default=False, action='store_true',
                        help='Use FP16/AMP training')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of workers for reading samples')
    parser.add_argument('--no-cudnn-bm', default=False, action='store_true',
                        help='Do not do benchmarking to select cuDNN algos')
    parser.add_argument('--no-prefetch', default=False, action='store_true',
                        help='Do not use fast prefetch pipeline')
    parser.add_argument('--hdmlp', default=False, action='store_true',
                        help='Use HDMLP for I/O')
    parser.add_argument('--hdmlp-config-path', type=str,
                        help='Config path for HDMLP')
    parser.add_argument('--hdmlp-lib-path', type=str, default=None,
                        help='Library path for HDMLP')
    parser.add_argument('--hdmlp-stats', default=False, action='store_true',
                        help='Save HDMLP statistics every epoch')
    parser.add_argument('--channels-last', default=False, action='store_true',
                        help='Use channels-last memory order')
    parser.add_argument('--dali', default=False, action='store_true',
                        help='Use DALI for I/O and data augmentation')
    parser.add_argument('--bucket-cap', type=int, default=25,
                        help='Communication max bucket size (in MB)')
    
    # Customize
    parser.add_argument('--dist-rank', type=int, required=True,
                        help='The rank of all nodes') # test 加入的全局秩
    parser.add_argument('--dist-size', type=int, default=2,
                        help='The rank of all nodes') # test node数量
    parser.add_argument('--file-name', type=str,
                        help='The path of ddp init') # 初始化路径
    parser.add_argument('--backend', type=str, default='nccl',
                        help='The backend of ddp init') # 初始化路径
    
    # 貌似上述参数名中的-都会在变量中变为_
    return parser.parse_args() # 该值类似于字典结构，可以使用.变量名来进行访问


class Logger: # 日志类
    """Simple logger that saves to a file and stdout."""

    def __init__(self, out_file, is_primary):
        """Save logging info to out_file."""
        self.is_primary = is_primary
        if is_primary: # 若首次
            if os.path.exists(out_file): # 且已经有日志文件
                print("the log file has crashed")
                # raise ValueError(f'Log file {out_file} already exists') # 报错 # test for deleting
            self.log_file = open(out_file, 'w') # 没有就新建
        else:
            self.log_file = None # 清空

    def log(self, message):
        """Log message."""
        if self.is_primary:
            # Only the primary writes the log.
            self.log_file.write(message + '\n')
            self.log_file.flush() # 刷新缓冲区，立即写入
            print(message, flush=True)

    def close(self):
        """Close the log."""
        if self.is_primary:
            self.log_file.close()


class AverageTracker:
    """Keeps track of the average of a value."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear the tracker."""
        self.vals = []

    def update(self, val, n=1):
        """Add n copies of val to the tracker."""
        if n == 1:
            self.vals.append(val)
        else:
            self.vals.extend([val]*n)

    def mean(self):
        """Return the mean."""
        if not self.vals:
            return float('nan')
        return statistics.mean(self.vals)

    def latest(self):
        """Return the latest value."""
        if not self.vals:
            return float('nan')
        return self.vals[-1]

    def save(self, filename):
        """Save data to a file."""
        with open(filename, 'a') as fp:
            fp.write(','.join([str(v) for v in self.vals]) + '\n')


@torch.jit.script
def _mean_impl(data, counts):
    """Internal scripted mean implementation."""
    return data.sum() / counts.sum()


class AverageTrackerDevice:
    """Keep track of the average of a value.

    This is optimized for storing the results on device.

    """

    def __init__(self, n, device, allreduce=True):
        """Track n total values on device.

        allreduce: Perform an allreduce over scaled values before
        computing mean.

        """
        self.n = n # len(data_loader),maybe按照batch分好类了
        self.device = device # get_cuda_device()
        self.allreduce = allreduce # args.dist
        self.last_allreduce_count = None
        self.saved_mean = None
        self.reset()

    def reset(self):
        """Clear the tracker."""
        self.data = torch.empty(self.n, device=self.device)
        self.counts = torch.empty(self.n, device='cpu', pin_memory=True)
        self.cur_count = 0
        # For caching results.
        self.last_allreduce_count = None
        self.saved_mean = None

    @torch.no_grad() # 表示不需要计算梯度，也不会进行反向传播
    def update(self, val, count=1.0):
        """Add val and associated count to tracker."""
        if self.cur_count == self.n:
            raise RuntimeError('Updating average tracker past end')
        self.data[self.cur_count] = val # loss,前向和标签的差值
        self.counts[self.cur_count] = count # samples.size(0),该batch的数量
        self.cur_count += 1

    @torch.no_grad()
    def mean(self):
        """Return the mean.

        This will be a device tensor.

        """
        if self.cur_count == 0:
            return float('nan')
        if self.cur_count == self.last_allreduce_count:
            return self.saved_mean
        valid_data = self.data.narrow(0, 0, self.cur_count)
        valid_counts = self.counts.narrow(0, 0, self.cur_count).to(self.device)
        scaled_vals = valid_data * valid_counts
        if self.allreduce:
            scaled_vals = allreduce_tensor(scaled_vals)
        mean = _mean_impl(scaled_vals, valid_counts).item()
        self.last_allreduce_count = self.cur_count
        self.saved_mean = mean
        return mean


def get_num_gpus():
    """Number of GPUs on this node."""
    return torch.cuda.device_count()


def get_local_rank(required=False): # 获取本地进程的rank
    return 0 # 在本地的索引
    """Get local rank from environment."""
    if 'MV2_COMM_WORLD_LOCAL_RANK' in os.environ:
        # The local rank of a process on a node within its job. The local rank ranges from 0,1 ... N-1 on a node with N processes running on it.
        return int(os.environ['MV2_COMM_WORLD_LOCAL_RANK'])
    if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    if 'SLURM_LOCALID' in os.environ:
        return int(os.environ['SLURM_LOCALID'])
    if required:
        raise RuntimeError('Could not get local rank')
    return 0


def get_local_size(required=False):
    return 1 # 本地只有一台GPU
    """Get local size from environment."""
    if 'MV2_COMM_WORLD_LOCAL_SIZE' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_LOCAL_SIZE']) # The number of ranks from this job that are running on this node
    if 'OMPI_COMM_WORLD_LOCAL_SIZE' in os.environ: 
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE']) # The number of ranks from this job that are running on this node
    if 'SLURM_NTASKS_PER_NODE' in os.environ: # 每个节点的任务数
        return int(os.environ['SLURM_NTASKS_PER_NODE'])
    if required:
        raise RuntimeError('Could not get local size')
    return 1


def get_world_rank(required=False):
    return int(os.environ['get_world_rank'])
    """Get rank in world from environment."""
    if 'MV2_COMM_WORLD_RANK' in os.environ: # os.enciron为环境变量的字典
        return int(os.environ['MV2_COMM_WORLD_RANK']) # The MPI rank of this process in current MPI job
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_RANK'])
    if 'SLURM_PROCID' in os.environ:
        return int(os.environ['SLURM_PROCID'])
    if required:
        raise RuntimeError('Could not get world rank')
    return 0


def get_world_size(required=False):
    return int(os.environ['get_world_size'])
    """Get world size from environment."""
    if 'MV2_COMM_WORLD_SIZE' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_SIZE'])
    if 'OMPI_COMM_WORLD_SIZE' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_SIZE'])
    if 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS'])
    if required:
        raise RuntimeError('Could not get world size')
    return 1

def find_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.


def initialize_dist(init_file, rendezvous='file', backend='nccl'): # 分布式初始化方式为tcp或file(默认)
    """Initialize PyTorch distributed backend."""
    torch.cuda.init()   # 在需要c端时显式进行初始化
    torch.cuda.set_device(get_local_rank()) # 本机上不同进程的get_local_rank()不同，因此分配的gpu号也不同

    init_file = os.path.abspath(init_file) # 转为绝对路径

    if rendezvous == 'tcp':
        dist_url = None
        node_id = get_world_rank() # 全局进程的序号
        num_nodes = get_world_size() # 分布式数量
        if node_id == 0: # 头号机器
            import socket
            ip = socket.gethostbyname(socket.gethostname()) # 获取ip地址
            port = find_free_port() # 获取端口
            dist_url = "tcp://{}:{}".format(ip, port) # 组装socket
            with open(init_file, "w") as f: # 在当前文件下创建init_工作序号
                f.write(dist_url) # 写入自己的url
        else: # 若不是零号机器
            while not os.path.exists(init_file): # 就等到有该文件后
                time.sleep(1)
            time.sleep(1)
            with open(init_file, "r") as f: # 读取0号机器的url
                dist_url = f.read()
        print("dist_url = ",dist_url,"\nnode_id = ",node_id,"\nnum_nodes=",num_nodes) # test
        torch.distributed.init_process_group(backend, init_method=dist_url, rank=node_id, world_size=num_nodes)
        print("process_group finish init") # test
        '''
        通信后端：gloo / NCCL
        初始化方法：tcp通信 / 共享文件系统 进行初始化
        rank: 当前进程的序号，且优先级
        world_size: 训练的所有进程数
        '''
    elif rendezvous == 'file':
        torch.distributed.init_process_group(
            backend='nccl', init_method=f'file://{init_file}',
            rank=get_world_rank(), world_size=get_world_size())
    else:
        raise NotImplementedError(f'Unrecognized scheme "{rendezvous}"')

    torch.distributed.barrier() # 同步
    # Ensure the init file is removed.
    if get_world_rank() == 0 and os.path.exists(init_file): # 用完后删除
        os.unlink(init_file)


def get_cuda_device():
    """Get this rank's CUDA device."""
    # return torch.device('cpu') # test
    return torch.device(f'cuda:{get_local_rank()}')


def get_job_id():
    """Return the resource manager job ID, if any"""
    if 'SLURM_JOBID' in os.environ:     # SLURM是用于Linux和Unix内核系统的免费，开源的任务调度工具，被世界范围内的超级计算机和计算机群广泛采用
        return os.environ['SLURM_JOBID']
    if 'LSB_JOBID' in os.environ:   # LSF是分布资源管理的工具，用来调度、监视、分析联网计算机的负载
        return os.environ['LSB_JOBID']
    return None


def allreduce_tensor(t):
    """Allreduce and average tensor t."""
    rt = t.clone().detach()
    torch.distributed.all_reduce(rt)
    rt /= get_world_size()
    return rt


def group_weight_decay(net, decay_factor, skip_list):
    """Set up weight decay groups.

    skip_list is a list of module names to not apply decay to.

    """
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        if any([pattern in name for pattern in skip_list]):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0},
            {'params': decay, 'weight_decay': decay_factor}]


def get_learning_rate_schedule(optimizer, args):
    """Return the learning rate schedule for training."""
    # Determine the target LR if needed.
    if args.lr is None: # 若没有给定学习率
        global_batch_size = args.batch_size * get_world_size() # 进行计算
        batch_factor = max(global_batch_size // args.base_batch, 1.0)
        args.lr = args.start_lr * batch_factor
    if args.warmup_epochs:
        target_warmup_factor = args.lr / args.start_lr
        warmup_factor = target_warmup_factor / args.warmup_epochs

    def lr_sched(epoch):
        factor = 1.0
        if args.warmup_epochs:
            if epoch > 0:
                if epoch < args.warmup_epochs:
                    factor = warmup_factor * epoch
                else:
                    factor = target_warmup_factor
        for step, decay in zip(args.decay_epochs, args.decay_factors):
            if epoch >= step:
                factor *= decay
        return factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched) # 根据epoch不同学习率变化的函数


@torch.jit.script
def _mean_stdev_impl(x, mean, stdev):
    """Internal scripted mean/stdev implementation."""
    return x.sub_(mean).div_(stdev)


class PrefetchTransform:
    """Apply set of transforms when prefetching.

    This handles conversion to float/tensor, and normalization.

    """

    def __init__(self, mean, stdev):
        self.mean = torch.Tensor([x*255 for x in mean]).cuda().view(1, 3, 1, 1)
        self.stdev = torch.Tensor([x*255 for x in stdev]).cuda().view(1, 3, 1, 1)

    def __call__(self, img):
        return _mean_stdev_impl(img, self.mean, self.stdev)


class RandomDataset(torch.utils.data.Dataset):
    """Dataset that just returns a random tensor for debugging."""

    def __init__(self, sample_shape, dataset_size, label=True, pil=False,
                 transform=None):
        super().__init__()
        self.sample_shape = sample_shape
        self.dataset_size = dataset_size
        self.label = label
        self.transform = transform
        self.pil = pil
        if pil:
            # Create a full image that will be decoded.
            rand_img = torch.rand(sample_shape)
            rand_img = torchvision.transforms.functional.to_pil_image(rand_img)
            buf = io.BytesIO()
            rand_img.save(buf, format='jpeg')
            self.d = buf.getvalue()
        else:
            self.d = torch.rand(sample_shape)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        d = self.d
        if self.pil:
            d = PIL.Image.open(io.BytesIO(d)).convert('RGB')
        if self.transform is not None:
            d = self.transform(d)
        if self.label:
            return d, 0
        else:
            return d


# Adapted from torchvision DatasetFolder.
class CachedImageFolder(torchvision.datasets.VisionDataset):
    """Like a regular ImageFolder, but supports caching metadata."""

    def __init__(self, root,
                 loader=torchvision.datasets.folder.pil_loader,
                 transform=None, target_transform=None,
                 is_valid_file=None):
        super().__init__(root, transform=transform,
                         target_transform=target_transform)

        if is_valid_file is None:
            extensions = torchvision.datasets.folder.IMG_EXTENSIONS
        else:
            extensions = None

        # Load cache if present.
        # TODO: Handle creating cache with multiple processes.
        cache_file = os.path.join(root, 'filelist.pickle')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                classes, class_to_idx, samples = pickle.load(f)
        else:
            classes, class_to_idx = self._find_classes(self.root)
            samples = torchvision.datasets.folder.make_dataset(
                self.root, class_to_idx, extensions, is_valid_file)
            if len(samples) == 0:
                raise RuntimeError(f'Found zero files in {self.root} subdirs')

            # Cache data.
            with open(cache_file, 'wb') as f:
                pickle.dump((classes, class_to_idx, samples), f)

        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, directory):
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


# Adapted from:
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/image_classification/dataloaders.py#L319
class PrefetchWrapper:
    """Prefetch ahead and perform preprocessing on GPU."""

    def __init__(self, data_loader, transform, dali=False):
        self.data_loader = data_loader
        self.transform = transform
        self.dali = dali
        # Will perform transforms on a separate CUDA stream.
        self.stream = torch.cuda.Stream()  # cuda流并行
        # Simplifies set_epoch.
        if hasattr(data_loader, 'sampler'):
            self.sampler = data_loader.sampler

    @staticmethod  # 静态方法，即可以直接通过类进行引用
    def prefetch_loader(data_loader, transform, stream, dali):
        """Actual iterator for loading."""
        first = True
        sample, target = None, None
        for next_data in data_loader:
            if dali:
                next_sample = next_data[0]["data"]
                next_target = next_data[0]["label"].squeeze(-1).long()
            else:
                next_sample, next_target = next_data

            with torch.cuda.stream(stream):
                next_sample = next_sample.cuda(non_blocking=True).float()
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

    def __iter__(self): # 用于enumerate使用
        return PrefetchWrapper.prefetch_loader(
            self.data_loader, self.transform, self.stream, self.dali)

    def __len__(self):
        return len(self.data_loader)

    def reset(self):
        if hasattr(self.data_loader, 'reset'):
            self.data_loader.reset()


def fast_collate(memory_format, batch, pin=False):
    imgs = [img[0] for img in batch] # 抓取每个sample的第一个维度，这里为通道(尚不支持channel_last)
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64) # 同样抓取每个sample的第二个维度
    #w = imgs[0].size[0]
    #h = imgs[0].size[1]
    w, h = 224, 224
    tensor = torch.zeros((len(imgs), 3, h, w), # 在内存中创建全为0的同batch相当的tensor,pin表示是否锁在内存中
                         dtype=torch.uint8,
                         pin_memory=pin).contiguous(
                             memory_format=memory_format)
    # imags为tensor的列表，每个tensor有两维
    for i, img in enumerate(imgs): # 对于每个切片的tensor
        nump_array = np.asarray(img, dtype=np.uint8) # 同np.array的区别，asarray在必要时才会进行复制
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1) # 补充维度在第三维
        nump_array = np.rollaxis(nump_array, 2) # 将第三维滚动到第一维
        # Suppress warnings.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # 过滤忽略级别的警告
            tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets # 暂时理解为做了一个变换


def create_dali_pipeline(batch_size, num_threads, device_id, data_dir, crop, size,
                       shard_id, num_shards, dali_cpu=False, is_training=True):
    try:
        from nvidia.dali.pipeline import Pipeline
        import nvidia.dali.types as types
        import nvidia.dali.fn as fn
    except ImportError:
        raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this mode.")

    pipeline = Pipeline(batch_size, num_threads, device_id, seed=12 + get_world_rank())
    with pipeline:
        images, labels = fn.file_reader(file_root=data_dir,
                                        shard_id=get_world_rank(),
                                        num_shards=get_world_size(),
                                        random_shuffle=is_training,
                                        pad_last_batch=True,
                                        name="Reader")
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        if is_training:
            images = fn.image_decoder_random_crop(images,
                                                  device=decoder_device, output_type=types.RGB,
                                                  device_memory_padding=device_memory_padding,
                                                  host_memory_padding=host_memory_padding,
                                                  random_aspect_ratio=[0.8, 1.25],
                                                  random_area=[0.1, 1.0],
                                                  num_attempts=100)
            images = fn.resize(images,
                               device=dali_device,
                               resize_x=crop,
                               resize_y=crop,
                               interp_type=types.INTERP_TRIANGULAR)
            mirror = fn.random.coin_flip(probability=0.5)
        else:
            images = fn.image_decoder(images,
                                      device=decoder_device,
                                      output_type=types.RGB)
            images = fn.resize(images,
                               device=dali_device,
                               size=size,
                               mode="not_smaller",
                               interp_type=types.INTERP_TRIANGULAR)
            mirror = False

        images = fn.crop_mirror_normalize(images.gpu(),
                                          dtype=types.FLOAT,
                                          output_layout="CHW",
                                          crop=(crop, crop),
                                          mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                          std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                          mirror=mirror)
        labels = labels.gpu()
        pipeline.set_outputs(images, labels)
    return pipeline


def setup_dali(args):
    try:
        from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    except ImportError:
        raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this mode.")

    # ImageNet sizes
    crop_size = 224
    val_size = 256

    if args.dataset == 'imagenet-22k':
        data_dir = args.data_dir
    else:
        data_dir = os.path.join(args.data_dir, 'train')

    pipe = create_dali_pipeline(batch_size=args.batch_size,
                                num_threads=args.workers,
                                device_id=get_local_rank(),
                                data_dir=data_dir,
                                crop=crop_size,
                                size=val_size,
                                dali_cpu=False,
                                shard_id=get_world_rank(),
                                num_shards=get_world_size(),
                                is_training=True)
    pipe.build()
    train_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

    # Getting number of samples from dataset name
    if args.dataset == 'imagenet':
        num_samples = 1281167
    elif args.dataset == 'imagenet-22k':
        num_samples = 14197103

    return train_loader, num_samples


@torch.no_grad()
def accuracy(output, target, topk=(1, 5)):
    """Compute accuracy@k for given top-ks."""
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    results = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        # Scaling by 100.0 / batch_size done in AverageTrackerDevice.
        results.append(correct_k)
    return results

def init_allreduce(time_gap, tensor_size, path='/home/zzp/code/NoPFS/output/log/allreduce-gloo-180'):
    with open(path, 'r') as f:
        allreduce = f.readlines()                       # 读取记录的trace
    times = []
    tensor_dimension = []
    index = 0 # 行索引
    while True:
        assert(allreduce[index][0]=="{" or allreduce[index][0]=="}") # 改行为{或{}时表示还未结束
        index+=1
        assert(allreduce[index][0:5]=="time:")
        times.append(float(allreduce[index][5:-1])) # 记录时间
        index+=1
        assert(allreduce[index][0:17]=="tensor dimension:")
        tensor_dimension.append(int(allreduce[index][17:-1])) # 维度
        index+=1
        assert(allreduce[index][0:12]=="tensor size:")
        tensor_size.append(int(allreduce[index][12:-1])) # 长度,TODO尚未兼容多维tensor
        index+=1
        if allreduce[index][0:2]!="}{": break
    time_gap.append(0) # 第一次睡眠0s
    for i in range(len(times)-1):
        time_gap.append(times[i+1]-times[i]) # 计算睡眠间隔

def train(args, train_loader, net, scaler, criterion, optimizer,
          log, transform, gaps, tensor_size, index):
    """Perform one epoch of training."""
    global temp
    global t
    net.train()
    losses = AverageTrackerDevice(len(train_loader), get_cuda_device(),allreduce=args.dist)
    batch_times = AverageTracker()
    data_times = AverageTracker()
    end_time = time.perf_counter()
    for batch, data in enumerate(train_loader):  # 关键取数据的位置,batch指batch的序号,data为数据数组
        if temp!=71:
            for i in range(6): # 每个step只有6次allreduce
                with torch.autograd.profiler.record_function("forward+a part of backward"):
                    print("sleep ", gaps[index])
                    if gaps[index]<0:
                        print(gaps[index])
                        print("******************************************************")
                    time.sleep(max(gaps[index],0))
                # tensor = torch.randn(tensor_size[index]).to(get_cuda_device()) # 构造tensor
                with torch.autograd.profiler.record_function("backward"):
                    print("allreduce",tensor_size[index])
                    torch.distributed.all_reduce(t[i],op=torch.distributed.ReduceOp.SUM) # 进行allreduce
                index += 1
        else:
            if True:
                for i in range(6): # 每个step只有6次allreduce
                    print("sleep ", gaps[index])
                    if gaps[index]<0:
                        print(gaps[index])
                        print("******************************************************")
                    time.sleep(max(gaps[index],0))
                    # tensor = torch.randn(tensor_size[index]).to(get_cuda_device()) # 构造tensor
                    print("allreduce",tensor_size[index])
                    torch.distributed.all_reduce(t[i],op=torch.distributed.ReduceOp.SUM) # 进行allreduce
                    index += 1
             
            print("trace recording")
        temp+=1
        # time.sleep(0.2) # 每次step后都有0.2s的同步cudaDeviceSynchronize


    log.log(f'    **Train** Loss {losses.mean():.5f}')
    if args.primary and args.save_stats:
        batch_times.save(
            os.path.join(args.output_dir, f'stats_batch_{args.job_id}.csv'))
        data_times.save(
            os.path.join(args.output_dir, f'stats_data_{args.job_id}.csv'))
    return index


def validate(args, validation_loader, net, criterion, log):
    """Validate on the given dataset."""
    net.eval()
    losses = AverageTrackerDevice(len(validation_loader), get_cuda_device(),
                                  allreduce=args.dist)
    top1s = AverageTrackerDevice(len(validation_loader), get_cuda_device(),
                                 allreduce=args.dist)
    top5s = AverageTrackerDevice(len(validation_loader), get_cuda_device(),
                                 allreduce=args.dist)
    with torch.no_grad():
        for samples, targets in validation_loader:
            samples = samples.to(get_cuda_device(), non_blocking=True)
            targets = targets.to(get_cuda_device(), non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                output = net(samples)
                loss = criterion(output, targets)

            losses.update(loss, samples.size(0))
            top1, top5 = accuracy(output, targets)
            top1s.update(top1, 100.0 / samples.size(0))
            top5s.update(top5, 100.0 / samples.size(0))
    log.log(f'    **Val** Loss {losses.mean():.5f} |'
            f' Top-1 {top1s.mean():.5f}% |'
            f' Top-5 {top5s.mean():.5f}%')
    return losses.mean()

# python /home/zzp/code/NoPFS/benchmark/resnet50.py --output-dir=/home/zzp/code/NoPFS/output --data-dir=/home/zzp/data/imagenet --job-id=1 --hdmlp --hdmlp-lib-path=/home/zzp/code/NoPFS/libhdmlp/build/libhdmlp.so --hdmlp-config-path=/home/zzp/code/NoPFS/libhdmlp/data/hdmlp_C.cfg --no-eval --epochs=2 --dist --r=tcp
def main():
    """Manage training."""
    args = get_args() # 获取参数
    
    # test 使用全局变量修改
    os.environ['get_world_size'] = str(args.dist_size)
    os.environ['get_world_rank'] = str(args.dist_rank)

    # Determine whether this is the primary rank.
    args.primary = get_world_rank(required=args.dist) == 0 # dist为是否进行分布式训练,若全局序号为0那么为master,只有master才会打印日志

    # Seed RNGs.
    '''
    这里的随机数用于数据和模型初始化参数的随机固定
    '''
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) # 为所有的GPU设置种子

    # Get job ID.
    # 检测是否存在分布式环境，若有则进行jobid的获取
    if args.job_id is None:
        args.job_id = get_job_id()
        if args.job_id is None:
            raise RuntimeError('No job ID specified')
    
    # 检测单机上的节点任务数是否超过当前机器的gpu个数，若超过则报错
    if args.dist:
        if get_local_size() > get_num_gpus():
            raise RuntimeError(
                'Do not use more ranks per node than there are GPUs')

        initialize_dist(args.file_name, args.rendezvous, args.backend)
    else: # 非分布式
        if get_world_size() > 1:
            print('Multiple processes detected, but --dist not passed',
                  flush=True)

    if not args.no_cudnn_bm:
        torch.backends.cudnn.benchmark = True
        # 会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
    if args.channels_last:
        if args.hdmlp:  # 当前hdmlp不支持通道维度在末尾的情况
            raise RuntimeError(
                'Not currently supporting channels-last with HDMLP')
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format # 将内存中tensor变为连续分布的格式.有时一个tensor在内存中并不连续存放,view就不可用

    # Set up output directory.
    if args.primary: # primary指是否为序号为0的机器,即master
        if not os.path.isdir(args.output_dir): # 若输出路径不是目录
            os.makedirs(args.output_dir)  # 就创建

    if args.hdmlp and not hdmlp:   # 检查hdmlp包在不在
        raise RuntimeError('HDMLP not available')

    log = Logger(os.path.join(args.output_dir, f'log_{args.job_id}.txt'),
                 args.primary)

    # Set up the model.
    if args.dataset == 'imagenet':
        num_classes = 1000      # imagenet有1000个类别
    elif args.dataset == 'imagenet-22k':
        num_classes = 21841     # imagenet-22k有21841个类别
    else:
        raise ValueError('Unknown dataset', args.dataset)   #只支持以上两个数据库
    net = torchvision.models.resnet50(pretrained=False,
                                      num_classes=num_classes,
                                      zero_init_residual=True)
    net = net.to(get_cuda_device()) # test将网络在对应GPU上训练
    net = net.to(memory_format=memory_format) # 规定内存tensor的内存格式
    if args.dist:
        net = torch.nn.parallel.DistributedDataParallel( # 多机多卡的分布式训练
            net,
            device_ids=[get_local_rank()],
            output_device=get_local_rank(),
            find_unused_parameters=True,
            bucket_cap_mb=args.bucket_cap) # 交流的最大缓存大小
    # Gradient scaler for AMP.
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16) # AMP自动混合精度，即自由变换tensor元素的类型， 在训练最开始之前实例化一个GradScaler对象
    # Loss function.
    if args.label_smoothing > 0.0: # 标签光滑处理，防止过拟合,如将one-hot [0,1] -> [0.05,0.95]
        raise RuntimeError('Label smoothing not yet supported')
    else:
        criterion = torch.nn.CrossEntropyLoss().to(get_cuda_device())
    # Optimizer.
    optimizer = torch.optim.SGD( # test原apex.optimizers.FusedSGD
        group_weight_decay(net, args.decay, ['bn']),
        args.start_lr,
        args.momentum)
    # Set up learning rate schedule.
    scheduler = get_learning_rate_schedule(optimizer, args) # 根据epoch不同学习率变化的函数

    # Record the current best validation loss during training.
    best_loss = float('inf') # 初始化为正无穷

    # TODO: Support checkpointing at some point if we care.

    # Set up data loaders.
    imagenet_means = [0.485, 0.456, 0.406]
    imagenet_stdevs = [0.229, 0.224, 0.225]
    # Build transforms.
    if not args.no_augmentation: # 若需要图像升级
        if args.no_prefetch: # 不使用prefetch
            train_transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop((224, 224)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    imagenet_means, imagenet_stdevs)])
            if not args.no_eval: # 若要在每个epoch测试准确率
                validation_transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(256),
                    torchvision.transforms.CenterCrop((224, 224)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        imagenet_means, imagenet_stdevs)]) # 则验证集也需要进行变换
        else: # 使用prefetch
            # Prefetcher will handle other transforms.
            train_transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop((224, 224)),
                torchvision.transforms.RandomHorizontalFlip()]) # 这里少进行一些预处理
            if not args.no_eval:
                validation_transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(256),
                    torchvision.transforms.CenterCrop((224, 224))])
    else: # 无需图像升级
        train_transform = None
        if not args.no_eval:
            validation_transform = None

    data_dir = args.data_dir
    if args.dataset != 'imagenet-22k':
        data_dir = os.path.join(data_dir, 'train') # 期望的imagenet数据路径形如.../train,imagenet-22k不需要train

    if args.hdmlp: # 采用hdmlp
        if args.hdmlp_stats: # 若要保存hdmlp每个epoch的统计信息
            os.environ['HDMLPPROFILING'] = '1'

        if args.synth_data: # 若使用合成数据
            raise RuntimeError('Synthetic data not supported with HDMLP') # 暂不支持
        collate_fn = functools.partial(fast_collate, memory_format,pin=True)
        '''
        原型fast_collate(memory_format, batch, pin=False),对batch做了一些变换,如下分析collate_fn(batch)即可
        '''
        '''
        functools.partial用于给定函数的部分参数,在对于其生成函数调用时只需要补齐剩余变量即可
        如对于f(a,b),我们可以令g=functools.partial(f,a=1),这样g(2)相当于f(1,2).用于回调函数时可以控制传入函数的参数值
        '''
        hdmlp_train_job = hdmlp.Job( # 创建job,此时仅构造了Job类还未初始化
            data_dir,
            args.batch_size * get_world_size(),
            args.epochs,
            'uniform',
            args.drop_last,
            transforms=[
                hdmlp.lib.transforms.ImgDecode(),
                hdmlp.lib.transforms.RandomResizedCrop(224),
                hdmlp.lib.transforms.RandomHorizontalFlip(),
                hdmlp.lib.transforms.HWCtoCHW()],
            seed=args.seed,
            config_path=args.hdmlp_config_path,
            libhdmlp_path=args.hdmlp_lib_path)
        train_dataset = hdmlp.lib.torch.HDMLPImageFolder( # 将job绑定在一个数据集上,在Folder中Job会被初始化
            data_dir,
            hdmlp_train_job,
            filelist=os.path.join(args.data_dir, 'hdmlp_files.pickle'))
        '''
        每个进程创建一个Job, Job再构建一个HDMLPImageFolder对象，其中
        子类 -> HDMLPImageFolder -> HDMLPDatasetFolder -> HDMLPVisionDataset -> PrefetchWrapper -> 父类
        HDMLPImageFolder构造函数调用HDMLPDatasetFolder构造函数，其中调用job.setup()进行job的初始化
        接着再调用libhdmlp/src/utils/hdmlp_api.cpp中的setup()初始化方法
        setup()中依次进行了jobid的分配、为该job_id创建prefetcher对象
        prefetcher对象中按照给定的各个等级的线程配置，初始化各种信息(如元数据、与其他node进行通信)，
        然后为staging buffer创建执行StagingBufferPrefetcher::prefetch的线程进行预取
        为其他存储等级创建PrefetcherBackend::prefetch线程进行预取(该线程先于staging buffer层线程创建)
        其中PrefetcherBackend::prefetch不断循环(死循环)读取数据到其区域内,
        StagingBufferPrefetcher::prefetch则不断取数据到staging buffer中
        '''
        train_loader = hdmlp.lib.torch.HDMLPDataLoader( # 继续讲dataset包装为dataloader
            train_dataset, collate_fn=collate_fn)
        if not args.no_eval:
            # TODO: Update to use faster HDMLP pipeline.
            hdmlp_validation_job = hdmlp.Job(  # 对验证集做同样操作
                os.path.join(args.data_dir, 'val'),
                args.batch_size * get_world_size(),
                args.epochs,
                'uniform',
                args.drop_last,
                seed=args.seed,
                config_path=args.hdmlp_config_path,
                libhdmlp_path=args.hdmlp_lib_path)
            validation_dataset = hdmlp.lib.torch.HDMLPImageFolder(
                os.path.join(args.data_dir, 'val'),
                hdmlp_validation_job,
                filelist=os.path.join(args.data_dir, 'hdmlp_files.pickle'),
                transform=validation_transform)
            validation_loader = hdmlp.lib.torch.HDMLPDataLoader(
                validation_dataset, collate_fn=collate_fn)
            num_val_samples = len(validation_dataset)
        else:
            num_val_samples = 0
        num_train_samples = len(train_dataset)
    elif args.dali: # 采用dali
        train_loader, num_train_samples = setup_dali(args)
        if not args.no_eval:
            raise NotImplementedError('DALI is currently only supported for training')
        num_val_samples = 0
    else: # dali和hdmlp都不适用
        # Set up loaders.
        if args.synth_data: # 如果使用合成数据(方便存在内存中)
            if args.dataset == 'imagenet':
                print("get one")
                num_samples = 20582 # test完整数量为1281167
            elif args.dataset == 'imagenet-22k':
                num_samples = 14197103
            train_dataset = RandomDataset((3, 256, 256), num_samples,
                                          pil=not args.no_augmentation,
                                          transform=train_transform) # 构造随机数据集
            num_train_samples = len(train_dataset)
            num_val_samples = 0
            if not args.no_eval:
                validation_dataset = RandomDataset(
                    (3, 256, 256), num_samples,
                    pil=not args.no_augmentation,
                    transform=validation_transform)
                num_val_samples = num_samples
        else: #否则
            train_dataset = CachedImageFolder(data_dir, transform=train_transform)
            num_train_samples = len(train_dataset)
            num_val_samples = 0
            if not args.no_eval:
                validation_dataset = CachedImageFolder(os.path.join(
                    args.data_dir, 'val'), transform=validation_transform)
                num_val_samples = len(validation_dataset)
        if args.dist: # 使用分布式训练
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=get_world_size(),
                rank=get_world_rank())
            if not args.no_eval:
                validation_sampler = torch.utils.data.distributed.DistributedSampler(
                    validation_dataset, num_replicas=get_world_size(),
                    rank=get_world_rank())
        else:   # 不使用
            train_sampler = torch.utils.data.RandomSampler(train_dataset)
            if not args.no_eval:
                validation_sampler = torch.utils.data.RandomSampler(
                    validation_dataset)
        if args.no_prefetch or args.no_augmentation:
            collate_fn = None
        else:
            collate_fn = functools.partial(fast_collate, memory_format)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            num_workers=args.workers, pin_memory=True,
            sampler=train_sampler, drop_last=args.drop_last,
            collate_fn=collate_fn)
        if not args.no_eval:
            validation_loader = torch.utils.data.DataLoader(
                validation_dataset, batch_size=args.batch_size,
                num_workers=args.workers, pin_memory=True,
                sampler=validation_sampler, drop_last=args.drop_last,
                collate_fn=collate_fn)
        else:
            validation_loader = None
    if not args.no_prefetch: # 若使用预取
        if args.dali:   # 使用dali
            transforms = None
        else: # 使用hdmlp
            transforms = PrefetchTransform(
                imagenet_means, imagenet_stdevs)
        train_loader = PrefetchWrapper(
            train_loader, transforms, dali=args.dali)

        transforms = None
        if not args.no_eval:
            validation_loader = PrefetchWrapper(
                validation_loader, PrefetchTransform(
                    imagenet_means, imagenet_stdevs), dali=args.dali)
    else:
        transforms = PrefetchTransform(imagenet_means, imagenet_stdevs)

    # Estimate reasonable print frequency as every 5%.
    if args.print_freq is None:
        args.print_freq = len(train_loader) // 20

    # Log training configuration.
    log.log(str(args))
    # log.log(str(net))
    log.log(f'Using {get_world_size()} processes')
    log.log(f'Global batch size is {args.batch_size*get_world_size()}'
            f' ({args.batch_size} per GPU)')
    log.log(f'Training data size: {num_train_samples}')
    if args.no_eval:
        log.log('No validation')
    else:
        log.log(f'Validation data size: {num_val_samples}')
    log.log(f'Starting learning rate: {args.start_lr}'
            f' | Target learning rate: {args.lr}'
            f' | Warmup epochs: {args.warmup_epochs}')

    # test allreduce
    gaps = []
    tensor_size = []
    init_allreduce(gaps,tensor_size,"/home/zzp/code/NoPFS/output/log/180-181-nccl-216") # 初始化时间间隔和发送的大小
    index = 0
    begin_time = time.perf_counter()

    # Train.
    log.log('Starting training at ' +
            time.strftime('%Y-%m-%d %X', time.gmtime(time.time())))
    epoch_times = AverageTracker()
    train_start_time = time.perf_counter()

    with torch.profiler.profile(activities=[ torch.profiler.ProfilerActivity.CPU , torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
        for epoch in range(args.epochs):
            start_time = time.perf_counter()
            if args.dist and hasattr(train_loader, 'sampler'):# hasattr用户判断对象是否含有相应属性
                train_loader.sampler.set_epoch(epoch) # 设置新的epoch的随机数
                if not args.no_eval:
                    validation_loader.sampler.set_epoch(epoch)

            log.log(f'==>> Epoch={epoch:03d}/{args.epochs:03d} '
                    f'Elapsed={int(time.perf_counter()-train_start_time)} s '
                    f'(avg epoch time: {epoch_times.mean():5.3f} s, current epoch: {epoch_times.latest():5.3f} s) '
                    f'[learning_rate={scheduler.get_last_lr()[0]:6.4f}]')

            index = train(args, train_loader, net, scaler, criterion, optimizer, log, transforms, gaps ,tensor_size, index)
            
            '''
            train(args, train_loader, net, scaler, criterion, optimizer, log, transforms) # PrefetchWrapper object
            -> for batch, data in enumerate(train_loader):  # 关键取数据的位置
            -> PrefetchWrapper.prefetch_loader(self.data_loader, self.transform, self.stream, self.dali)
            -> for next_data in data_loader: # hdmlp.lib.torch.hdmlpdataloader.HDMLPDataLoader object
            -> samples, labels = self.dataset[0:iter_batch_size] # Dataset HDMLPImageFolder
            -> folder_label, sample = self.job.get()
            -> get()等待数据装入到staging buffer后指定偏移进行读取
            '''
            if not args.no_eval:
                val_loss = validate(args, validation_loader, net, criterion, log)
                if val_loss < best_loss:
                    best_loss = val_loss
                    log.log(f'New best loss: {best_loss:.5f}')
            scheduler.step()

            if hasattr(train_loader, 'reset'):
                train_loader.reset()
            if not args.no_eval:
                if hasattr(validation_loader, 'reset'):
                    validation_loader.reset()

            epoch_times.update(time.perf_counter() - start_time)

            # Store HDMLP statistics every epoch
            if args.primary and args.hdmlp and args.hdmlp_stats:
                stat_path = os.path.join(
                    args.output_dir, f'stats_hdmlp_{args.job_id}.pkl')
                with open(stat_path, 'wb') as fp:
                    pickle.dump(hdmlp_train_job.get_metrics(), fp)
    prof.export_chrome_trace('/home/zzp/code/NoPFS/output/log/resnet_profile.json') #
    end_time = time.perf_counter()
    log.log(f'==>> Done Elapsed={int(time.perf_counter()-train_start_time)} s '
            f'(avg epoch time: {epoch_times.mean():5.3f} s '
            f'current epoch: {epoch_times.latest():5.3f} s)')
    print("total spend ",end_time-begin_time)
    a = input("press ctrl+c to quit") # test

temp = 1
t1 = torch.ones(3102696).to(get_cuda_device())
t2 = torch.ones(7875584).to(get_cuda_device())
t3 = torch.ones(7417344).to(get_cuda_device())
t4 = torch.ones(6755584).to(get_cuda_device())
t5 = torch.ones(405824).to(get_cuda_device())
t6 = torch.ones(161).to(get_cuda_device())
t = [t1,t2,t3,t4,t5,t6]

if __name__ == '__main__':
    main()
    
    # Skip teardown to avoid hangs
    os._exit(0)


# 模拟
# 1node1gpu/mini - 150s


# 真实
# 1node1gpu/mini - 155s