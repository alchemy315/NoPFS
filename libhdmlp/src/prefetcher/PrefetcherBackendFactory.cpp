#include <iostream>
#include "../../include/prefetcher/PrefetcherBackendFactory.h"
#include "../../include/prefetcher/MemoryPrefetcher.h"
#include "../../include/prefetcher/FileSystemPrefetcher.h"
#include "../../include/utils/Metrics.h"

PrefetcherBackend* PrefetcherBackendFactory::create(const std::string& prefetcher_backend,      // 后端名称
                                                    std::map<std::string, std::string>& backend_options, // 对应的可选项map
                                                    unsigned long long int capacity, // 容量
                                                    std::vector<int>::iterator start,
                                                    std::vector<int>::iterator end,
                                                    StorageBackend* storage_backend,
                                                    MetadataStore* metadata_store,
                                                    int storage_level, int job_id, int node_id, Metrics* metrics) {
    PrefetcherBackend* pfb;
    if (prefetcher_backend == "memory") { // 若为内存层
        pfb = new MemoryPrefetcher(backend_options, start, end, capacity, storage_backend, metadata_store,
                                   storage_level, true, metrics); // 创建内存层的prefetcher
    } else if (prefetcher_backend == "filesystem") { // 若为文件系统层
        pfb = new FileSystemPrefetcher(backend_options, start, end, capacity, storage_backend, metadata_store,
                                       storage_level, job_id, node_id, metrics); // 创建文件系统层
    } else {
        throw std::runtime_error("Unsupported prefetch backend");
    }
    return pfb;
}
