from typing import List, Dict
import matplotlib.pyplot as plt
import pandas as pd
from bisect import bisect_left
from IPython.display import display
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import pickle


class Statistics:

    def __init__(self, aggregate_per_batch: bool, N: int):
        self.aggregate = aggregate_per_batch
        self.stall_times = []
        self.exec_times = []
        self.reads = [[[] for _ in range(4)] for _ in range(N)] # reads for the different options (0 = staging pool, 1 = PFS, 2 = local, 3 = remote)
        self.prefetch_reads = [[[] for _ in range(4)] for _ in range(N)] # reads for the different options (0 = staging pool, 1 = PFS, 2 = local, 3 = remote)
        self.N = N
        self.init_stall = 0
        if self.aggregate:
            self.batch_stall_times = {}
            self.batch_exec_times = {}

    def add_stall_time(self, node_id, batch_no, batch_offset, batch_size, stall_time):
        if self.aggregate:
            if batch_no not in self.batch_stall_times:
                self.batch_stall_times[batch_no] = stall_time / batch_size / self.N
            else:
                self.batch_stall_times[batch_no] += stall_time / batch_size / self.N
        else:
            self.stall_times.append(stall_time)

    def add_init_stall_time(self, node_id, stall_time):
        """
        Adds initial stall time (e.g. for preparation work that stalls whole pipeline)
        """
        self.init_stall = stall_time
        self.add_stall_time(node_id, 0, 0, 1, stall_time)


    def get_stall_times(self) -> List[float]:
        if self.aggregate:
            return list(self.batch_stall_times.values())
        else:
            return self.stall_times

    def add_exec_time(self, node_id, batch_no, batch_offset, batch_size, exec_time):
        if self.aggregate:
            if batch_no not in self.batch_exec_times:
                self.batch_exec_times[batch_no] = exec_time
            else:
                self.batch_exec_times[batch_no] = max(self.batch_exec_times[batch_no], exec_time)
        else:
            self.exec_times.append(exec_time)

    def get_exec_times(self) -> List[float]:
        if self.aggregate:
            return list(self.batch_exec_times.values())
        else:
            return self.exec_times

    def get_run_time(self):
        return max(self.get_exec_times())

    def add_read(self, node_id, time, option):
        self.reads[node_id][option].append(time)

    def add_prefetch_read(self, node_id, time, option):
        self.prefetch_reads[node_id][option].append(time)

    def get_pfs_statistics(self, end_time, no_buckets = 50) -> Dict[float, int]:
        stats = {}
        combined_reads = []
        for node_id in range(self.N):
            combined_list = self.reads[node_id][1] + self.prefetch_reads[node_id][1]
            combined_list.sort()
            combined_reads.append(combined_list)
        bucket_range = end_time / no_buckets
        list_pointers = [0] * self.N
        t = 0
        while t < end_time:
            stats[t] = 0
            for i in range(self.N):
                pointer = list_pointers[i]
                while pointer < len(combined_reads[i]) and combined_reads[i][pointer] <= t:
                    pointer += 1
                if pointer < len(combined_reads[i]) and combined_reads[i][pointer] < t + bucket_range:
                    stats[t] += 1
                list_pointers[i] = pointer
            t += bucket_range

        return stats

    def get_pfs_clients(self, time, node_id):
        clients = 1
        delta = 0.1
        for i in range(self.N):
            if i == node_id:
                continue
            if len(self.reads[i]) > 1:
                pfs_reads = self.reads[i][1]
                if pfs_reads and pfs_reads[-1] > time - delta:
                    k = bisect_left(pfs_reads, time)
                    if k < len(pfs_reads) and pfs_reads[k] > time - delta or k + 1 < len(pfs_reads) and pfs_reads[k + 1] < time + delta:
                        clients += 1
                        continue
            if len(self.prefetch_reads[i]) > 1:
                pfs_prefetch_reads = self.prefetch_reads[i][1]
                if pfs_prefetch_reads and pfs_prefetch_reads[-1] > time - delta:
                    k = bisect_left(pfs_prefetch_reads, time)
                    if k < len(pfs_prefetch_reads) and pfs_prefetch_reads[k] > time - delta or k + 1 < len(pfs_prefetch_reads) and pfs_prefetch_reads[k + 1] < time + delta:
                        clients += 1
                        continue
        return clients

    def get_read_statistics(self, end_time, no_buckets = 50) -> List[Dict]:
        stats = {}
        aggregated_stats = [{} for _ in range(4)] # Contains dict per option
        total_reads = [0, 0, 0, 0]
        for node_id in range(self.N):
            for option in range(4):
                self.reads[node_id][option].sort()
                total_reads[option] += len(self.reads[node_id][option])
        all_reads = sum(total_reads)
        for option in range(4):
            total_reads[option] /= all_reads
        print(total_reads)
        bucket_range = end_time / no_buckets
        list_pointers = [[0 for _ in range(4)] for _ in range(self.N)]
        t = 0
        while t < end_time:
            stats[t] = {}
            for i in range(self.N):
                for option in range(4):
                    stats[t][option] = 0
                    pointer = list_pointers[i][option]
                    option_reads = self.reads[i][option]
                    while pointer < len(option_reads) and option_reads[pointer] <= t:
                        stats[t][option] += 1
                        pointer += 1
                    list_pointers[i][option] = pointer
            t += bucket_range

        for time, option_values in stats.items():
            overall_reads = sum(option_values.values())
            for option in option_values:
                if overall_reads != 0:
                    aggregated_stats[option][time] = option_values[option] / overall_reads
                else:
                    aggregated_stats[option][time] = 0

        return aggregated_stats

    @staticmethod
    def table_run_time(policy_stats, aggregate):
        display(pd.DataFrame([[n, v.get_run_time()] for n, v in policy_stats.items()], columns=["Policy", "Runtime (sec.)"]))

    @staticmethod
    def plot_exec_time(policy_stats, aggregate, run=None, include_legend=True):
        new_dict = {}
        for policy_name, stats in policy_stats.items():
            if policy_name not in new_dict:
                new_dict[policy_name] = {}
            new_dict[policy_name]['exec_time'] = stats.get_exec_times()
            end_time = stats.get_exec_times()[-1]
            read_statistics = stats.get_read_statistics(end_time)
            for option_no, option_dict in enumerate(read_statistics):
                avg = 0
                for val in option_dict.values():
                    avg += val
                avg /= len(option_dict.values())
                new_dict[policy_name][option_no] = avg
        pickle.dump(new_dict, open("data/{}".format(run), "wb"))
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.ylabel("Execution Time (s)")
        plt.xlabel("Batch No" if aggregate else "Iteration")
        i = 0
        for policy_name, stats in policy_stats.items():
            ls = ['-', '--', '-.', ':'][i % 4]
            plt.plot(stats.get_exec_times(), label=policy_name, linestyle=ls)
            i += 1

        axins = zoomed_inset_axes(ax, 2, loc='center', bbox_to_anchor=(0,0), borderpad=3)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

        if include_legend:
            ax.legend(fontsize='x-small')

    @staticmethod
    def plot_stall_time(policy_stats, aggregate):
        plt.figure()
        plt.title("Stall Time")
        plt.ylabel("Stall Time (sec.)")
        plt.xlabel("Batch No" if aggregate else "Iteration")
        for policy_name, stats in policy_stats.items():
            plt.plot(stats.get_stall_times(), label=policy_name)
        plt.legend()

    @staticmethod
    def plot_pfs_reads(policy_stats, aggregate):
        plt.figure()
        plt.title("PFS Reads")
        plt.ylabel("Number of reads")
        plt.xlabel("Seconds")
        for policy_name, stats in policy_stats.items():
            end_time = stats.get_exec_times()[-1]
            pfs_statistics = stats.get_pfs_statistics(end_time)
            plt.plot(list(pfs_statistics.keys()), list(pfs_statistics.values()), label=policy_name)
        plt.legend()

    @staticmethod
    def plot_read_fraction(policy_stats, aggregate):
        for policy_name, stats in policy_stats.items():
            plt.figure()
            plt.title("Reads served from - {}".format(policy_name))
            plt.ylabel("Fraction")
            plt.xlabel("Seconds")
            end_time = stats.get_exec_times()[-1]
            read_statistics = stats.get_read_statistics(end_time)
            for option_no, option_dict in enumerate(read_statistics):
                if option_no == 0:
                    label = "Staging Pool"
                elif option_no == 1:
                    label = "PFS"
                elif option_no == 2:
                    label = "Local"
                elif option_no == 3:
                    label = "Remote"

                plt.plot(list(option_dict.keys()), list(option_dict.values()), label=label)
            plt.legend()
    @staticmethod
    def create_plots(policy_stats, aggregate, run):
        Statistics.table_run_time(policy_stats, aggregate)
        Statistics.plot_exec_time(policy_stats, aggregate, run)
        #Statistics.plot_stall_time(policy_stats, aggregate)
        #Statistics.plot_pfs_reads(policy_stats, aggregate)
        #Statistics.plot_read_fraction(policy_stats, aggregate)

    @staticmethod
    def plot_strong_scaling(node_stats, title):
        plt.figure()
        plt.title("Strong Scaling - {}".format(title))
        plt.ylabel("Run time (sec.)")
        plt.yscale("log", basey=2)
        plt.xlabel("Nodes")
        plt.xscale("log", basex=2)
        run_times = [stats.get_run_time() for stats in node_stats.values()]
        plt.plot(list(node_stats.keys()), run_times)
