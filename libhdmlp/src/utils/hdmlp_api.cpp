#include <iostream>
#include "../../include/storage/FileSystemBackend.h"
#include "../../include/prefetcher/Prefetcher.h"
#include "../../include/utils/hdmlp_api.h"


int setup(wchar_t* dataset_path,            // 数据库路径
          wchar_t* config_path,             // 配置我文件路径
          int batch_size,
          int epochs,
          int distr_scheme,                 // 随机分布
          bool drop_last_batch,
          int seed,
          wchar_t** transform_names,
          char* transform_args,
          int transform_output_size,
          int transform_len,
          wchar_t* filesystem_backend,
          wchar_t* hdf5_data_name,
          wchar_t* hdf5_target_name,
          bool collate_data) {
    int job_id = 0;
    while (job_id < PARALLEL_JOBS_LIMIT) {      //若还没有超过上限
        if (!used_map[job_id]) {                //寻找没有被使用的job_id
            used_map[job_id] = true;
            break;
        } else {
            job_id++;
        }
    }
    if (job_id == PARALLEL_JOBS_LIMIT) {        //若超出上限
        throw std::runtime_error("Maximal parallel jobs exceeded");
    }
    //Prefetcher* pf[PARALLEL_JOBS_LIMIT];
    pf[job_id] = new Prefetcher(dataset_path,       // 在对应的job_id创建prefetchr，里面进行了全部内容
                                config_path,
                                batch_size,
                                epochs,
                                distr_scheme,
                                drop_last_batch,
                                seed,
                                job_id,
                                transform_names,
                                transform_args,
                                transform_output_size,
                                transform_len,
                                filesystem_backend,
                                hdf5_data_name,
                                hdf5_target_name,
                                collate_data);
    return job_id;
}

char* get_staging_buffer(int job_id) {
    char* staging_buffer = pf[job_id]->get_staging_buffer();
    return staging_buffer;
}

int get_node_id(int job_id) {
    return pf[job_id]->get_node_id();
}

int get_no_nodes(int job_id) {
    return pf[job_id]->get_no_nodes();
}

int length(int job_id) {
    return pf[job_id]->get_dataset_length();
}

unsigned long long int get_next_file_end(int job_id) {
    pf[job_id]->notify_data_consumed(consumed_until[job_id]); // 通知prefetch访问新的数据，此时旧的数据就可以退出
    unsigned long long int file_end = pf[job_id]->get_next_file_end();
    consumed_until[job_id] = file_end; 
    return file_end;
}

int get_metric_size(int job_id, wchar_t* kind, int index, int subindex) {
    std::wstring wskind(kind);
    std::string skind(wskind.begin(), wskind.end());
    if (skind == "stall_time") {
        if (pf[job_id]->metrics != nullptr) {
            return pf[job_id]->metrics->stall_time.size();
        } else {
            return 0;
        }
    } else if (skind == "augmentation_time") {
        return pf[job_id]->metrics->augmentation_time.size();
    } else if (skind == "augmentation_time_thread") {
        return pf[job_id]->metrics->augmentation_time[index].size();
    } else if (skind == "read_times") {
        return pf[job_id]->metrics->read_times.size();
    } else if (skind == "read_times_threads") {
        return pf[job_id]->metrics->read_times[index].size();
    } else if (skind == "read_times_threads_elem") {
        return pf[job_id]->metrics->read_times[index][subindex].size();
    }
    return 0;
}

double* get_stall_time(int job_id) {
    return pf[job_id]->metrics->stall_time.data();
}

double* get_augmentation_time(int job_id, int thread_id) {
    return pf[job_id]->metrics->augmentation_time[thread_id].data();
}

double* get_read_times(int job_id, int storage_class, int thread_id) {
    return pf[job_id]->metrics->read_times[storage_class][thread_id].data();
}

int* get_read_locations(int job_id, int storage_class, int thread_id) {
    return pf[job_id]->metrics->read_locations[storage_class][thread_id].data();
}

int get_label_distance(int job_id) {
    return pf[job_id]->largest_label_size;
}

void destroy(int job_id) {
    delete pf[job_id];
    used_map[job_id] = false;
}
