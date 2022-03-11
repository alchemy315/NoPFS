#include "../../include/utils/MetadataStore.h"
#include <algorithm>
#include <climits>


MetadataStore::MetadataStore(int networkbandwidth_clients, // NOLINT(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
                             int networkbandwidth_filesystem,
                             std::map<int, int>* pfs_bandwidths,
                             std::vector<std::map<int, int>>* storage_level_bandwidths,
                             std::vector<int>* no_threads) {
    interp_pfs_bandwidth = interpolate_map(n, pfs_bandwidths);      // 插值获取带宽值
    for (int unsigned long i = 0; i < no_threads->size(); i++) {    // 对于每个存储等级
        int threads = (*no_threads)[i];     // 获取线程数
        std::map<int, int>* storage_level_bandwidth = &(*storage_level_bandwidths)[i];
        interp_storage_level_bandwidths.push_back(interpolate_map(threads, storage_level_bandwidth) / threads);
    }   // 把每个等级线程数都插值进入
    this->networkbandwidth_clients = networkbandwidth_clients;
    this->networkbandwidth_filesystem = networkbandwidth_filesystem;
}

void MetadataStore::set_no_nodes(int no_nodes) {
    this->n = no_nodes;
}

void MetadataStore::insert_cached_file(int storage_level, int file_id) {
    std::lock_guard<std::shared_timed_mutex> writer_lock(file_locations_mutex);
    file_locations[file_id] = storage_level;
}

int MetadataStore::get_storage_level(int file_id) {
    std::shared_lock<std::shared_timed_mutex> reader_lock(file_locations_mutex);
    if (file_locations.count(file_id) != 0) {
        return file_locations[file_id];
    } else {
        return 0;
    }
}

void MetadataStore::store_planned_locations(std::vector<int>::iterator& start,
                                            std::vector<int>::iterator& end,
                                            int storage_level) {
  std::lock_guard<std::shared_timed_mutex> lock(planned_file_locations_mutex);
  for (auto i = start; i < end; ++i) {
    planned_file_locations[*i] = storage_level;
  }
}

int MetadataStore::get_planned_storage_level(int file_id) {
  std::shared_lock<std::shared_timed_mutex> lock(planned_file_locations_mutex);
  if (planned_file_locations.count(file_id) != 0) {
    return planned_file_locations[file_id];
  } else {
    return -1;
  }
}

double MetadataStore::interpolate_map(int key_val, std::map<int, int>* map) {   // 对其中离key的值最近的左右两个插值获取数值
    if (map->count(key_val) > 0) {  // 若有键对应的值
        return (*map)[key_val];     // 直接返回
    } else {                        // 否则
        int lb = 0;                 // 下界
        int ub = INT_MAX;           // 上界
        int min_nodes = INT_MAX;
        int max_nodes = 0;
        for (auto& pair : *map) {
            if (pair.first > lb && pair.first < key_val) {  // 若下界可变大
                lb = pair.first;                            // 就变大
            } else if (pair.first < ub && pair.first > key_val) {   // 若下界可变小
                ub = pair.first;                                    // 就变小
            }
            if (pair.first < min_nodes) {   // 若可以就更新最小值
                min_nodes = pair.first;
            }
            if (pair.first > max_nodes) {   // 若可以更新最大值
                max_nodes = pair.first;
            }
        }
        if (lb == 0) {                      // 最值直接返回
            return (*map)[min_nodes];
        } else if (ub == INT_MAX) {
            return (*map)[max_nodes];
        } else {
            // Interpolate
            int lb_bandwidth = (*map)[lb];
            int ub_bandwidth = (*map)[ub];
            return lb_bandwidth + (double) (key_val - lb) * (ub_bandwidth - lb_bandwidth) / (ub - lb);  // 进行插值
        }
    }
}

/**
 * Returns prefetching options, ordered according to the bandwidth.
 * Option 0 is PFS, 1 remote read and 2 local read. See the performance model for details
 * @param local_storage_level In which local storage level the file is available, 0 if not
 * @param remote_storage_level In which remote storage level the file is available, 0 if not
 * @param options Array with 3 elements, options will be put into this array
 */
void MetadataStore::get_option_order(int local_storage_level, int remote_storage_level, int* options) {
    double pfs_speed = std::min(interp_pfs_bandwidth / n, networkbandwidth_filesystem);
    double local_speed = 0.0;
    double remote_speed = 0.0;
    if (local_storage_level != 0) {
        local_speed = interp_storage_level_bandwidths[local_storage_level];
    }
    if (remote_storage_level != 0) {
        remote_speed = std::min(interp_storage_level_bandwidths[remote_storage_level], networkbandwidth_clients);
    }
    double speeds[] = {pfs_speed, remote_speed, local_speed};
    options[0] = 0;
    options[1] = 1;
    options[2] = 2;
    std::sort(options, options + 3, [&speeds](int& a, int& b) {
        if (speeds[a] == speeds[b]) {
            // Prefer local over remote over PFS in case of ties
            return (a == OPTION_LOCAL) || (a == OPTION_REMOTE && b != OPTION_LOCAL);
        }
        return speeds[a] > speeds[b];
    });
}
