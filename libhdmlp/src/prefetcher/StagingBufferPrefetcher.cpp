#include <iostream>
#include <thread>
#include <cstring>
#include "../../include/prefetcher/StagingBufferPrefetcher.h"
#include "../../include/utils/Metrics.h"

StagingBufferPrefetcher::StagingBufferPrefetcher(char* staging_buffer, unsigned long long int buffer_size, int node_id, int no_threads,
                                                 Sampler* sampler, StorageBackend* backend, PrefetcherBackend** pf_backends,
                                                 MetadataStore* metadata_store, DistributedManager* distr_manager,
                                                 TransformPipeline** transform_pipeline, int transform_output_size, Metrics* metrics,
                                                 bool collate_data) {
    this->buffer_size = buffer_size;
    this->staging_buffer = staging_buffer;
    this->node_id = node_id;
    this->no_threads = no_threads;
    this->sampler = new Sampler(*sampler);
    this->backend = backend;
    this->pf_backends = pf_backends;
    this->metadata_store = metadata_store;
    this->distr_manager = distr_manager;
    this->transform_pipeline = transform_pipeline;
    this->metrics = metrics;
    if (transform_pipeline != nullptr || collate_data) {
        batch_size = sampler->get_node_local_batch_size();
        unsigned long max_file_size = 0;
        for (int i = 0; i < backend->get_length(); i++) {
            unsigned long size = backend->get_file_size(i);
            int label_size = backend->get_label_size(i) + 1;
            if (size > max_file_size) {
                max_file_size = size;
            }
            if (label_size > largest_label_size) {
                largest_label_size = label_size;
            }
        }
        if (transform_pipeline != nullptr) {
          transform_buffers = new char*[no_threads];
          for (int i = 0; i < no_threads; i++) {
            transform_buffers[i] = new char[max_file_size];
          }
          this->transform_output_size = transform_output_size;
        }
    }
    this->collate_data = collate_data;
    global_iter_done = new bool[no_threads]();
}

StagingBufferPrefetcher::~StagingBufferPrefetcher() {
    delete sampler;
    delete[] global_iter_done;
    if (transform_pipeline != nullptr) {
        for (int i = 0; i < no_threads; i++) {
            delete[] transform_buffers[i];
        }
        delete[] transform_buffers;
    }
}

void StagingBufferPrefetcher::prefetch(int thread_id) {
    while (true) {
        std::vector<int> curr_access_string;
        sampler->get_node_access_string(node_id, curr_access_string); // 获取访问序列
        int access_string_size = curr_access_string.size(); // 获取访问长度
        int inserted_until = 0;
        bool do_transform = transform_pipeline != nullptr;
        bool profiling = metrics != nullptr;
        std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2;
        while (true) { // 不断循环
            std::unique_lock<std::mutex> crit_section_lock(prefetcher_mutex); // 上锁
            while (waiting_for_consumption) { 
                consumption_waiting_cond_var.wait(crit_section_lock); // 若正在使用数据就等待
            }
            int j = prefetch_offset; // 本次预取的目标
            if (j == 0) { // 若为首次prefetch
                curr_iter_file_ends.resize(access_string_size);
                curr_iter_file_ends_ready.resize(access_string_size);
                for (int i = 0; i < access_string_size; i++) {
                    curr_iter_file_ends_ready[i] = false; // 初始化
                }
            }
            prefetch_offset += 1; // 记录

            if (j >= access_string_size) { // 若访问完成
                break; // 就退出死循环
            }
            int file_id = curr_access_string[j]; // 获取当前要访问的数据id
            unsigned long file_size = backend->get_file_size(file_id);// 获取其数据大小
            int label_size = backend->get_label_size(file_id);// 及其标签的大小
            unsigned long entry_size = file_size + label_size + 1; // 总的大小,最后+1分割每个组合的占位符
            /*
            entry_size 指数据加标签的大小
            staging_buffer_pointer 指当前插入数据到staging buffer的位置
            read_offset 指当前在使用的数据(即替换不能超过此处)
            */
            if (do_transform) { // 如果要做变换
                // Batch mode, i.e. we fetch batch_size consecutive labels / samples
                entry_size = transform_output_size + label_size + 1; // 则大小改为变换后的大小
                if (j % batch_size == 0) { // 如果是一个batch的开头第一个数据
                    // If drop_last is false, can have smaller batches
                    curr_batch_size = std::min(access_string_size - j, batch_size); // 可以针对最后一个小batch进行缩小
                    // We're starting a new batch, need to check if there is enough space
                    while (staging_buffer_pointer < read_offset && staging_buffer_pointer + curr_batch_size * (transform_output_size + largest_label_size) >= read_offset) { // 若需要填入的batch覆盖到了待读的数据
                        // Prevent overwriting of non-read data
                        waiting_for_consumption = true;
                        read_offset_cond_var.wait(crit_section_lock);// 就停下等待
                    }
                }
            } else if (collate_data) { // 如果使用合成文件
              // Batch mode without transforming.
              if (j % batch_size == 0) { // 同理
                curr_batch_size = std::min(access_string_size - j, batch_size);
                while (staging_buffer_pointer < read_offset
                       && staging_buffer_pointer + curr_batch_size*(file_size + largest_label_size) >= read_offset) {
                  waiting_for_consumption = true;
                  read_offset_cond_var.wait(crit_section_lock);
                }
              }
            } else { // 如果需要读取数据
                while (staging_buffer_pointer < read_offset && staging_buffer_pointer + entry_size >= read_offset) { // 同理
                    // Prevent overwriting of non-read data
                    waiting_for_consumption = true;
                    read_offset_cond_var.wait(crit_section_lock);
                }
            }
            // 前面处理了每个batch第一个数据的情况
            // 以下为其余情况

            unsigned long long int local_staging_buffer_pointer;
            int batch_offset = 0;
            if (do_transform) {
                if (j % batch_size == batch_size - 1 || j == access_string_size - 1) { // 在读取该batch最后一个数据时
                    if (staging_buffer_pointer + (curr_batch_size + batch_size) * (transform_output_size + largest_label_size) > buffer_size) { // 如果当前正在装的batch(即正在装最后一个数据的batch)和下一个batch的大小超过总的大小
                        staging_buffer_pointer = 0;// 下一个batch就从头开始装
                        while (batch_size * (transform_output_size + largest_label_size) >= read_offset) {// 且开始等待,等到从开始的空间足够使用开始
                            waiting_for_consumption = true;
                            read_offset_cond_var.wait(crit_section_lock);
                        }
                    }

                    local_staging_buffer_pointer = staging_buffer_pointer;// 记录起点
                    staging_buffer_pointer += curr_batch_size * (transform_output_size + largest_label_size);//记录重点
                } else { // 不是batch的最后一个元素
                    local_staging_buffer_pointer = staging_buffer_pointer; // 正常装入
                }
                batch_offset = j % batch_size; // 记录batch内部序号
            } else if (collate_data) { // 同理
              if (j % batch_size == batch_size - 1
                  || j == access_string_size - 1) {
                if (staging_buffer_pointer + (curr_batch_size+batch_size)*(file_size+largest_label_size) > buffer_size) {
                  staging_buffer_pointer = 0;
                  while (batch_size * (file_size + largest_label_size) >= read_offset) {
                    waiting_for_consumption = true;
                    read_offset_cond_var.wait(crit_section_lock);
                  }
                }
                local_staging_buffer_pointer = staging_buffer_pointer;
                staging_buffer_pointer += curr_batch_size * (file_size+largest_label_size);
              } else {
                local_staging_buffer_pointer = staging_buffer_pointer;
              }
              batch_offset = j % batch_size;
            } else {
                if (staging_buffer_pointer + entry_size > buffer_size) {
                    // Start again at beginning of array
                    staging_buffer_pointer = 0;
                    // Ensure that overwriting is not possible after reset of pointer
                    while (entry_size >= read_offset) {
                        waiting_for_consumption = true;
                        read_offset_cond_var.wait(crit_section_lock);
                    }
                }
                local_staging_buffer_pointer = staging_buffer_pointer;
                staging_buffer_pointer += entry_size;
            }
            //以上占好了位置
            int curr_local_batch_size = curr_batch_size;

            if (waiting_for_consumption) {
                waiting_for_consumption = false;
                consumption_waiting_cond_var.notify_all();
            }
            crit_section_lock.unlock();

            backend->fetch_label(file_id, staging_buffer + local_staging_buffer_pointer + batch_offset * largest_label_size); // 指定位置存放标签
            if (do_transform || collate_data) {
                // Fill remaining bytes with zero bytes
                for (unsigned long long k = local_staging_buffer_pointer + batch_offset * largest_label_size + label_size + 1;
                    k < local_staging_buffer_pointer + (batch_offset + 1) * largest_label_size; k++) {
                    staging_buffer[k] = 0;
                }//填0
            }
            if (!do_transform) {
              if (collate_data) {
                fetch(file_id,
                      staging_buffer + local_staging_buffer_pointer + curr_local_batch_size*largest_label_size + batch_offset*file_size,
                      thread_id);
              } else {
                fetch(file_id, staging_buffer + local_staging_buffer_pointer + label_size + 1, thread_id); // 取文件
              }
            } else {
                fetch(file_id, transform_buffers[thread_id], thread_id);
                if (profiling) {
                    t1 = std::chrono::high_resolution_clock::now();
                }
                transform_pipeline[thread_id]->transform(transform_buffers[thread_id], file_size,
                        staging_buffer + local_staging_buffer_pointer + curr_local_batch_size * largest_label_size + batch_offset * transform_output_size);
                if (profiling) {
                    t2 = std::chrono::high_resolution_clock::now();
                    metrics->augmentation_time[thread_id].emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count());
                }
            }
            std::unique_lock<std::mutex> staging_buffer_lock(staging_buffer_mutex);
            // Check if all the previous file ends were inserted to the queue. If not, don't insert, but only set
            // curr_iter_file_ends / curr_iter_file_ends_ready s.t. another thread will insert it
            if (!do_transform) {
              if (collate_data) {
                curr_iter_file_ends[j] = local_staging_buffer_pointer + curr_local_batch_size*largest_label_size + (batch_offset + 1) * file_size;
              } else {
                curr_iter_file_ends[j] = local_staging_buffer_pointer + entry_size;
              }
            } else {
                curr_iter_file_ends[j] = local_staging_buffer_pointer + curr_local_batch_size * largest_label_size + (batch_offset + 1) * transform_output_size;
            }
            curr_iter_file_ends_ready[j] = true; // 标记为准备好了
            bool all_prev_inserted = true; // 是否当前文件之前所有都准备好了
            for (int k = inserted_until; k < j; k++) { // 进行检查
                if (!curr_iter_file_ends_ready[k]) { // 若存在没有准备好的
                    all_prev_inserted = false; // 就置0
                    break; // 中断循环
                } else {
                    inserted_until = k; // 记录最大的连续准备好的k
                }
            }
            if (all_prev_inserted) { // 若确实全部准备好了
                // Also insert file_ends from faster threads
                int k = j;
                bool inserted = false;
                while (k < access_string_size && curr_iter_file_ends_ready[k]) {
                    if ((!do_transform && !collate_data) || k % batch_size == batch_size - 1 || k == access_string_size - 1) {
                        file_ends.push_back(curr_iter_file_ends[k]); // 将其插入的file_ends提供给staging buffer prefetcher使用
                        inserted = true;
                    }
                    k++;
                }
                if (inserted) {
                    staging_buffer_cond_var.notify_one();
                }
            }
            staging_buffer_lock.unlock();
        }
        bool all_threads_done = true;

        // Advance batch when all threads are done with the current one
        std::unique_lock<std::mutex> crit_section_lock(prefetcher_mutex);
        global_iter_done[thread_id] = true;
        for (int i = 0; i < no_threads; i++) { // 检查是否所有线程都完成
            if (!global_iter_done[i]) {
                all_threads_done = false;
            }
        }
        if (all_threads_done) { // 若是
            sampler->advance_batch(); // 则重新打乱数据
            //void Sampler::advance_batch() { shuffle_sequence(access_sequence); }
            prefetch_batch += 1; // 进行下一个batch
            prefetch_offset = 0; // 偏移归零
            for (int i = 0; i < no_threads; i++) {
                global_iter_done[i] = false;
            }
            batch_advancement_cond_var.notify_all();
        } else {
            batch_advancement_cond_var.wait(crit_section_lock);
        }

        if (prefetch_batch >= sampler->epochs) {
            break;
        }

        crit_section_lock.unlock();
    }
}

void StagingBufferPrefetcher::fetch(int file_id, char* dst, int thread_id) {
    std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2;
    bool profiling = metrics != nullptr;
    int remote_storage_level = distr_manager->get_remote_storage_class(file_id); // 获取远处的等级
    int local_storage_level = metadata_store->get_storage_level(file_id);   // 获取本地的等级
    int option_order[3];
    metadata_store->get_option_order(local_storage_level, remote_storage_level, option_order); // 通过等级进行评估从哪里获取数据
    if (profiling) {
        t1 = std::chrono::high_resolution_clock::now();
    }

    if (option_order[0] == OPTION_REMOTE) { // 若从其他node取
        if (distr_manager->fetch(file_id, dst, thread_id)) {//若从其它节点取到了数据
            if (profiling) {
                t2 = std::chrono::high_resolution_clock::now();
                metrics->read_locations[0][thread_id].emplace_back(OPTION_REMOTE);
                metrics->read_times[0][thread_id].emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count());
            }
            return; // 就完成
        } else if (profiling) { // 否则记录
            // Track unsuccesful remote fetches as well
            metrics->read_locations[0][thread_id].emplace_back(-1);
            metrics->read_times[0][thread_id].emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count());
        }
    }
    // 若首先在本地取，或者上一步没有在远处取到
    if (option_order[0] == OPTION_LOCAL || (option_order[0] == OPTION_REMOTE && option_order[1] == OPTION_LOCAL)) {
        pf_backends[local_storage_level - 1]->fetch(file_id, dst); // 就在本地取数据
        if (profiling) { // 记录
            t2 = std::chrono::high_resolution_clock::now();
            metrics->read_locations[0][thread_id].emplace_back(OPTION_LOCAL);
            metrics->read_times[0][thread_id].emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count());
        }
    } else { // 否则
        int planned_storage_level = metadata_store->get_planned_storage_level(file_id); // 查看该数据是否应该被缓存
        if (planned_storage_level != -1) { // 若是
          // File is meant to be cached, but we are ahead of the prefetcher.
          // Use the current thread to help out.
          pf_backends[planned_storage_level - 1]->fetch_and_cache(file_id, dst);// 则帮助prefetcher取回并缓存
        } else { // 若不是
          backend->fetch(file_id, dst); // 则从后端取
        }
        if (profiling) {
          // File is uncached, so both options hit the PFS.
          t2 = std::chrono::high_resolution_clock::now();
          metrics->read_locations[0][thread_id].emplace_back(OPTION_PFS);
          metrics->read_times[0][thread_id].emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count());
        }
    }
}

void StagingBufferPrefetcher::advance_read_offset(unsigned long long int new_offset) {
    std::unique_lock<std::mutex> lock(prefetcher_mutex);
    read_offset = new_offset;
    read_offset_cond_var.notify_one();
}

unsigned long long int StagingBufferPrefetcher::get_next_file_end() {
    std::unique_lock<std::mutex> staging_buffer_lock(staging_buffer_mutex); // 加锁
    while (file_ends.empty()) { // 若空了
        staging_buffer_cond_var.wait(staging_buffer_lock); // 就等待
    }
    unsigned long long int file_end = file_ends.front(); // 去除最前的元素
    file_ends.pop_front(); // 并移除
    return file_end;
}
