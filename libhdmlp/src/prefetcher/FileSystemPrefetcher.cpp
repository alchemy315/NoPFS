#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <iostream>
#include "../../include/prefetcher/FileSystemPrefetcher.h"
#include "../../include/utils/MetadataStore.h"

FileSystemPrefetcher::FileSystemPrefetcher(std::map<std::string, std::string>& backend_options, // 存放了路径
                                           std::vector<int>::iterator prefetch_start,
                                           std::vector<int>::iterator prefetch_end,
                                           unsigned long long int capacity, StorageBackend* backend,
                                           MetadataStore* metadata_store, int storage_level, int job_id, int node_id, Metrics* metrics) :
        MemoryPrefetcher(backend_options, prefetch_start, prefetch_end, capacity, backend, metadata_store, storage_level, false, metrics) {// 先利用内存的版本，但不分配内存上的控件，即只构造了file_ends的数组
    path = backend_options["path"]; // 获取配置中的路径
    if (path == "env") { // 若路径为env
      char* env_path = std::getenv("HDMLP_FILESYSTEM_PATH"); // 从环境变量中获取
      if (env_path == nullptr) {
        throw std::runtime_error("HDMLP_FILESYSTEM_PATH not set");
      }
      path = env_path;
    }
    if (path.back() != '/') {
        path += '/'; // 为末尾加上/
    }
    struct stat base_dir{};
    int j = stat(path.c_str(), &base_dir); //test 查看数字
    if(j != 0){
        throw std::runtime_error("Configured file system prefetching path doesn't exist");
    }
    if(!S_ISDIR(base_dir.st_mode)){
        throw std::runtime_error("Configured file system prefetching path isn't a directory");
    }
    // test将以下判断拆开了
    // if (!(stat(path.c_str(), &base_dir) == 0 && S_ISDIR(base_dir.st_mode))) { // NOLINT(hicpp-signed-bitwise)
    //     throw std::runtime_error("Configured file system prefetching path doesn't exist or isn't a directory");
    // }

    path = path + std::to_string(job_id) + "_" + std::to_string(node_id); // 创建0_0的文件作为缓存

    if ((fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, (mode_t) 0600)) < 0) { // NOLINT(hicpp-signed-bitwise)
        throw std::runtime_error("Error opening file for prefetching");
    }
    if (lseek(fd, capacity - 1, SEEK_SET) < 0) {
        throw std::runtime_error("Error seeking file for prefetching");
    }
    if (write(fd, "", 1) != 1) {
        throw std::runtime_error("Error writing to file for prefetching");
    }
    buffer = static_cast<char*>(mmap(nullptr, capacity, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)); // NOLINT(hicpp-signed-bitwise) // 分配一些磁盘上的空间进行读写,作为缓存
    if (buffer == MAP_FAILED) {
        throw std::runtime_error("Error while mmapping file");
    }
}

FileSystemPrefetcher::~FileSystemPrefetcher() {
    munmap(buffer, capacity);
    close(fd);
    unlink(path.c_str());
}