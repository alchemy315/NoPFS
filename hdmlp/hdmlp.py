import ctypes
import pathlib
from sys import platform
from typing import Optional, List
from .lib.transforms import transforms
import numpy as np


class Job:

    DISTR_SCHEMES = {'uniform': 1}

    def __init__(self,
                 dataset_path: str,                                         # 数据集的路径
                 batch_size: int,                                           # batch大小
                 epochs: int,                                               # epoch
                 distr_scheme: str,                                         # eg 'uniform',随机分布类型
                 drop_last_batch: bool,                                     # 是否去掉最后一个小batch
                 transforms: Optional[List[transforms.Transform]] = None,   # 预处理变换
                 seed: Optional[int] = None,                                # 随机数种子
                 config_path: Optional[str] = None,                         # hdmlp配置文件,确定了存储等级以及每个等级的线程个数
                 libhdmlp_path: Optional[str] = None,                       # linhdmlp的代码,以下自己生成
                 filesystem_backend: Optional[str] = "filesystem",
                 hdf5_data_name: Optional[str] = None,
                 hdf5_target_name: Optional[str] = None,
                 collate_data: Optional[bool] = False):
        libname = self._get_lib_path(libhdmlp_path)                         # 获取DLL的路径
        self.config_path = self._get_config_path(config_path)
        self.hdmlp_lib = ctypes.CDLL(libname)                               # 加载动态链接库到python中
        # hdmlp_lib的方法见\hdmlp\libhdmlp\src\utils\hdmlp_api.cpp
        self.hdmlp_lib.get_next_file_end.restype = ctypes.c_ulonglong
        self.hdmlp_lib.get_staging_buffer.restype = ctypes.c_void_p
        self.dataset_path = dataset_path                                    # 记录信息
        self.batch_size = batch_size
        self.epochs = epochs
        if distr_scheme not in self.DISTR_SCHEMES:
            raise ValueError("Distribution scheme {} not supported".format(distr_scheme))
        self.distr_scheme = self.DISTR_SCHEMES[distr_scheme]
        self.drop_last_batch = drop_last_batch
        self.transforms = [] if transforms is None else transforms
        self.transformed_size = 0
        self.trans_w, self.trans_h, self.trans_c = None, None, None
        if transforms is not None:
            self._get_transformed_size()
        self.seed = seed
        self.filesystem_backend = filesystem_backend
        self.hdf5_data_name = hdf5_data_name or ''
        self.hdf5_target_name = hdf5_target_name or ''
        self.buffer_p = None    # Job在初始化时尚未分配buffer_p
        self.buffer_offset = 0
        self.job_id = None      # 也尚未分配Job_id
        self.collate_data = collate_data

    def _get_lib_path(self, configured_path) -> str:        # 从libhdmlp获取lib的路径
        if configured_path is None:     # 若创建jjob时没有给定libhdmlp
            folder = pathlib.Path(__file__).parent.parent.absolute()
            library_name = "libhdmlp.so"
            if platform == "darwin":
                library_name = "libhdmlp.dylib"
            path = folder / library_name
        else:
            path = pathlib.Path(configured_path)
        if not path.exists():
            raise EnvironmentError("Couldn't find library at location {}".format(path))
        return str(path)

    def _get_config_path(self, configured_path) -> str:     # 从config获取路径
        if configured_path is None:
            path = pathlib.Path(__file__).parent.absolute() / "data" / "hdmlp.cfg"
        else:
            path = pathlib.Path(configured_path)
        if not path.exists():
            raise EnvironmentError("Couldn't find configuration at location {}".format(path))
        return str(path)

    def _get_transformed_size(self):
        w, h, c = self.get_transformed_dims()
        out_size = self.transforms[-1].get_output_size(w, h, c)
        if out_size == transforms.Transform.UNKNOWN_SIZE:
            raise ValueError("Can't determine the output size after applying the transformations")
        self.transformed_size = out_size

    def get_transformed_dims(self):
        if self.trans_w is None or self.trans_h is None or self.trans_c is None:
            w, h, c = transforms.Transform.UNKNOWN_DIMENSION, transforms.Transform.UNKNOWN_DIMENSION, transforms.Transform.UNKNOWN_DIMENSION
            for transform in self.transforms:
                w, h, c = transform.get_output_dimensions(w, h, c)
            self.trans_w, self.trans_h, self.trans_c = w, h, c
        return self.trans_w, self.trans_h, self.trans_c

    def setup(self):
        cpp_transform_names = [transform.__class__.__name__ for transform in self.transforms]
        cpp_transform_names_arr = (ctypes.c_wchar_p * len(cpp_transform_names))()
        cpp_transform_names_arr[:] = cpp_transform_names
        transform_arg_size = sum(sum(ctypes.sizeof(arg) for arg in transform.arg_types) for transform in self.transforms)
        transform_args_arr = (ctypes.c_byte * transform_arg_size)()
        transform_args_arr_p = ctypes.cast(ctypes.pointer(transform_args_arr), ctypes.c_void_p)
        for transform in self.transforms:
            arg_types = transform.arg_types
            args = transform.get_args()
            for type, arg in zip(arg_types, args):
                p = ctypes.cast(transform_args_arr_p, ctypes.POINTER(type))
                p[0] = arg
                transform_args_arr_p.value += ctypes.sizeof(type)
        job_id = self.hdmlp_lib.setup(ctypes.c_wchar_p(self.dataset_path),      # 安装libhdmlp中的部分
                                      ctypes.c_wchar_p(self.config_path),
                                      self.batch_size,
                                      self.epochs,
                                      self.distr_scheme,
                                      ctypes.c_bool(self.drop_last_batch),
                                      self.seed,
                                      cpp_transform_names_arr,
                                      transform_args_arr,
                                      self.transformed_size,
                                      len(cpp_transform_names),
                                      ctypes.c_wchar_p(self.filesystem_backend),
                                      ctypes.c_wchar_p(self.hdf5_data_name),
                                      ctypes.c_wchar_p(self.hdf5_target_name),
                                      ctypes.c_bool(self.collate_data))
        buffer = self.hdmlp_lib.get_staging_buffer(job_id)  # 获取staging_buffer
        self.job_id = job_id
        self.buffer_p = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)) # buffer_p为cpp中char类型指针,指向staging buffer

    def destroy(self):
        self.hdmlp_lib.destroy(self.job_id)

    # 返回一个数据和标签
    def get(self, num_items = 1, decode_as_np_array=False, np_array_shape=None, np_array_type=ctypes.c_float, is_string_label=True, label_shape=None,
            fixed_label_len=None):
        labels = []
        file_end = self.hdmlp_lib.get_next_file_end(self.job_id) # 获取下一个需要去除的文件末端,即已经被放入到staging buffer的文件末端
        if file_end < self.buffer_offset: # buffer_offset是数据开始的位置,如果文件末尾再上次结束的前面(即该不等式)，那么说明已经过了一次地址循环
            self.buffer_offset = 0 # 因此重置为0
        self.label_distance = self.hdmlp_lib.get_label_distance(self.job_id) # 获取最大的标签大小
        label_offset = 0
        for i in range(num_items): # num_items为需要的标签数量
            prev_label_offset = label_offset # 记录老的label_offset
            if self.label_distance == 0: # 如果最大的label大小为0
                if fixed_label_len: # 但却要求固定label长度
                    # Work-around for when labels can contain nul bytes.
                    label_offset = fixed_label_len # 那还是给他分配固定长度的空间
                else: # 没有规定固定label长度
                    while self.buffer_p[self.buffer_offset + label_offset] != b'\x00': # 把非零的部分都加入进来
                        label_offset += 1
            else: # 若最大label不为0
                label_offset += self.label_distance - 1 # 拓展空间
            if is_string_label: # 若label是string
                label = self.buffer_p[self.buffer_offset + prev_label_offset:self.buffer_offset + label_offset] # 中间的部分即为label
                if label[-1] == 0: # 若末尾是0
                    label = label[:label.find(0)] # 就去掉
                labels.append(label.decode('utf-8')) # 加入labels
            else: # 若不然
                label = np.ctypeslib.as_array(ctypes.cast(ctypes.cast(self.buffer_p, ctypes.c_void_p).value + self.buffer_offset + prev_label_offset, ctypes.POINTER(ctypes.c_float)),
                                              label_shape) # 转换类型
                labels.append(label) # 接上
            label_offset += 1
        if decode_as_np_array:
            file = np.ctypeslib.as_array(ctypes.cast(ctypes.cast(self.buffer_p, ctypes.c_void_p).value + self.buffer_offset + label_offset, ctypes.POINTER(np_array_type)),
                                         np_array_shape)
        else:
            file = self.buffer_p[self.buffer_offset + label_offset:file_end] # label 后面接着的就是file
        self.buffer_offset = file_end # 为下次做准备
        if num_items == 1: # 若只有一个数据
            labels = labels[0] # 那么只取第一个
        return labels, file

    def length(self):
        return self.hdmlp_lib.length(self.job_id)

    def get_node_id(self):
        return self.hdmlp_lib.get_node_id(self.job_id)

    def get_no_nodes(self):
        return self.hdmlp_lib.get_no_nodes(self.job_id)

    def get_batch_size(self):
        return self.batch_size

    def get_num_epochs(self):
        return self.epochs

    def get_drop_last_batch(self):
        return self.drop_last_batch

    def get_transforms(self):
        return self.transforms

    def get_metrics(self):
        metrics = {
            "stall_time": [],
            "augmentation_time": [],
            "read_times": [],
            "read_locations": []
        }
        stall_time_size = self.hdmlp_lib.get_metric_size(self.job_id, "stall_time", 0, 0)
        if stall_time_size == 0:
            print("No metrics acquired during run, did you set HDMLPPROFILING to 1?")
        else:
            self.hdmlp_lib.get_stall_time.restype = ctypes.POINTER(ctypes.c_double * stall_time_size)
            metrics["stall_time"] = [e for e in self.hdmlp_lib.get_stall_time(self.job_id).contents]
            if self.transforms:
                prefetcher_threads = self.hdmlp_lib.get_metric_size(self.job_id, "augmentation_time", 0, 0)
                for i in range(prefetcher_threads):
                    num_elems = self.hdmlp_lib.get_metric_size(self.job_id, "augmentation_time_thread", i, 0)
                    self.hdmlp_lib.get_augmentation_time.restype = ctypes.POINTER(ctypes.c_double * num_elems)
                    metrics["augmentation_time"].append([e for e in self.hdmlp_lib.get_augmentation_time(self.job_id, i).contents])
            storage_classes = self.hdmlp_lib.get_metric_size(self.job_id, "read_times", 0, 0)
            for i in range(storage_classes):
                class_read_times = []
                class_read_locations = []
                num_threads = self.hdmlp_lib.get_metric_size(self.job_id, "read_times_threads", i, 0)
                for j in range(num_threads):
                    num_elems = self.hdmlp_lib.get_metric_size(self.job_id, "read_times_threads_elem", i, j)
                    self.hdmlp_lib.get_read_times.restype = ctypes.POINTER(ctypes.c_double * num_elems)
                    self.hdmlp_lib.get_read_locations.restype = ctypes.POINTER(ctypes.c_int * num_elems)
                    if num_elems == 0:
                        class_read_times.append([])
                        class_read_locations.append([])
                    else:
                        class_read_times.append([e for e in self.hdmlp_lib.get_read_times(self.job_id, i, j).contents])
                        class_read_locations.append([e for e in self.hdmlp_lib.get_read_locations(self.job_id, i, j).contents])
                metrics["read_times"].append(class_read_times)
                metrics["read_locations"].append(class_read_locations)
        return metrics
