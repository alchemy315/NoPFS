import torch
import math


class HDMLPDataLoader(object):

    def __init__(self, dataset, collate_fn=None, label_type=torch.int64):
        self.dataset = dataset  # 数据库中含有绑定的job
        self.dataset_size = len(self.dataset)
        self.job = self.dataset.get_job()  # 获取
        self.n = self.job.get_no_nodes()
        self.node_id = self.job.get_node_id()
        self.global_batch_size = self.job.get_batch_size()
        self.num_epochs = self.job.get_num_epochs()
        # Counts are set to replicate strategy of Sampler.cpp
        self.node_local_batch_size = math.ceil(self.global_batch_size / self.n) # 局部batch大小计算为全局batch大小/机器数
        self.local_batch_size = min(max(self.global_batch_size - self.node_id * self.node_local_batch_size, 0), self.node_local_batch_size)
        self.drop_last_batch = self.job.get_drop_last_batch()
        self.batch_offset = 0
        self.collate_fn = collate_fn
        self.label_type = label_type

    def __iter__(self):
        return self

    def __next__(self):  # core
        iter_batch_size = self.local_batch_size
        if self.batch_offset > self.dataset_size:   # 若已经读取的batch超过大小
            self.batch_offset = 0   # 归零
            raise StopIteration     # 停止迭代
        elif self.batch_offset + self.global_batch_size > self.dataset_size: # 若剩下一个全局batch
            if self.drop_last_batch:    # 且设置为丢弃最后一个batch
                self.batch_offset = 0   # 就直接归零
                raise StopIteration     # 停止迭代
            else:   # 最后一个仍然需要取出
                # Replicate Sampler.cpp, i.e. nodes iterate still node_local_batch_size in the last batch, unless their offset is higher than the file size
                iter_batch_size = min(max((self.dataset_size - self.batch_offset) - self.node_local_batch_size * self.node_id, 0), self.node_local_batch_size)
                if iter_batch_size == 0:
                    self.batch_offset = 0
                    raise StopIteration

        self.batch_offset += self.global_batch_size # 增加batch偏移量

        samples, labels = self.dataset[0:iter_batch_size] # core
        if isinstance(labels, list):
            # Batch mode
            labels = torch.tensor([label for label in labels],
                                  dtype=self.label_type, pin_memory=True)
            samples = samples.pin_memory()
            return samples, labels
        else:
            if self.collate_fn is None:
                # Use "standard" HDMLP collation.
                try:
                    sample_list = [torch.as_tensor(samples)]
                except RuntimeError:
                    sample_list = [samples]
                label_list = [torch.as_tensor(labels)]
                for i in range(iter_batch_size - 1):
                    sample, label = self.dataset[0]
                    label_list.append(torch.as_tensor(label))
                    try:
                        sample_list.append(torch.as_tensor(sample))
                    except RuntimeError:
                        sample_list.append(sample)

                try:
                    sample_list = torch.stack(sample_list, 0)
                except:
                    pass

                return sample_list, torch.stack(label_list, 0)
            else:
                batch = [(samples, labels)]
                for i in range(iter_batch_size - 1):
                    batch.append(self.dataset[0])
                return self.collate_fn(batch)

    def __len__(self):
        
        if self.drop_last_batch:
            return len(self.dataset) // self.global_batch_size
        else:
            return math.ceil(len(self.dataset) / self.global_batch_size)
        '''
        return len(self.dataset)
        '''