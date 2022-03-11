from stats.Statistics import *
from dataset.Dataset import *
from tqdm.auto import tqdm

class PolicyExecutor:

    def __init__(self, policies, dataset: Dataset, N, storage_classes, p, t, d, b_fs, b_c, beta, c,
                 aggregate_stats: bool):
        self.policies = {}
        self.batches = {}
        node_local_batches = dataset.get_node_local_batches()
        file_sizes = dataset.get_file_sizes()
        for name, policy_dict in policies.items():
            start = time.perf_counter()
            policy = policy_dict['class']
            args = policy_dict['opts']
            stats = Statistics(aggregate_stats, N)
            try:
                self.policies[name] = policy(node_local_batches, file_sizes, N, storage_classes, p, t, d, b_fs, b_c, beta, c, stats, **args)
                # policies may change node_local_batches
                self.batches[name] = self.policies[name].node_local_batches
            except:
                # Policies can raise an exception when they don't support a certain setting
                continue
            print('Setup policy', name, 'done', time.perf_counter() - start)
        self.file_sizes = file_sizes
        self.N = N
        self.c = c

    def run(self):
        global_times = {name: policy.get_stats().init_stall for name, policy in self.policies.items()}
        all_stats = {p : None for p in self.policies.keys()}
        for policy_name, policy in self.policies.items():
            stats = policy.get_stats()
            print("[Starting Compute Simulation - Policy {}]".format(policy_name))
            node_local_batches = self.batches[policy_name]
            for batch_no in tqdm(range(len(node_local_batches))):
                batch = node_local_batches[batch_no]
                node_local_times = [global_times[policy_name]] * self.N
                node_id = 0
                node_offsets = [0] * self.N
                node_batch_done = [False] * self.N
                while not all(node_batch_done):
                    batch_size = len(node_local_batches[batch_no][node_id])
                    if batch_size == 0:
                        node_batch_done[node_id] = True
                    offset = node_offsets[node_id]
                    if node_batch_done[node_id]:
                        node_id = (node_id + 1) % self.N
                        continue
                    request_time = node_local_times[node_id]
                    response_time, immediate = policy.request(batch_no, offset, node_id,
                                                                    request_time)
                    if immediate:
                        node_local_times[node_id] = request_time
                        stall_time = 0
                    else:
                        node_local_times[node_id] = response_time
                        stall_time = response_time - request_time
                    node_local_times[node_id] = node_local_times[node_id] + self._get_compute_time(policy_name, node_id, batch_no, offset)
                    stats.add_stall_time(node_id, batch_no, offset, batch_size, stall_time)
                    stats.add_exec_time(node_id, batch_no, offset, batch_size, node_local_times[node_id])
                    node_offsets[node_id] += 1
                    if node_offsets[node_id] == len(batch[node_id]):
                        node_batch_done[node_id] = True

                global_times[policy_name] = max(node_local_times)  # Barrier at end of batch
            all_stats[policy_name] = stats
        return all_stats


    def _get_compute_time(self, policy_name, node_id, batch_no, batch_offset):
        node_local_batches = self.batches[policy_name]
        file_id = node_local_batches[batch_no][node_id][batch_offset]
        file_size = self.file_sizes[file_id]
        return file_size / self.c

    @staticmethod
    def run_multiple_nodes(policy,
                           dataset_config,
                           E,
                           B,
                           Nmax,
                           node_distr_scheme,
                           drop_last_iter,
                           storage_classes,
                           p,
                           t,
                           d,
                           b_fs,
                           b_c,
                           beta,
                           c,
                           aggregated_stats,
                           node_increase = "exponential"):
        i = 1
        node_stats = {}
        while i <= Nmax:
            ds = Dataset(dataset_config, E, B, i, node_distr_scheme, drop_last_iter, None)
            print("[Simulating {} nodes]".format(i))
            policy_executor = PolicyExecutor(policy,
                                             ds,
                                             i,
                                             storage_classes,
                                             p,
                                             t,
                                             d,
                                             b_fs,
                                             b_c,
                                             beta,
                                             c,
                                             aggregated_stats)
            stats = policy_executor.run()
            policy_stats = next(iter(stats.values()))
            node_stats[i] = policy_stats

            if node_increase == "exponential":
                i *= 2
            else:
                i += 1
        return node_stats
