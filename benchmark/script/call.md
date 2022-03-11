# docker版本

## 数据集
* imagenet2021:位于138node/139node的/disk内，分为三种imagenet_mini/imagenet_mid/imagenet_whole规格
* 挂载命令
* sudo sshfs -o nonempty,exec,allow_other zzp@10.0.2.138:/disk /home/zzp/code/NoPFS/data

## 动态库
* 位于/home/zzp/code/NoPFS/libhdmlp/build中
* libhdmlp.so 常规库
* libhdmlp_no_local.so 不使用本地存储
* libhdmlp_docker.so 在docker中环境使用(前两个不行)

## hdmlp配置
* 位于/home/zzp/code/NoPFS/libhdmlp/data中
* .cfg文件

## 单机
python /home/zzp/code/NoPFS/benchmark/resnet50_docker.py \
--job-id=1 --hdmlp --no-eval --epochs=2 --hdmlp-stats \
--output-dir=/home/zzp/code/NoPFS/output/log \
--hdmlp-lib-path=/home/zzp/code/NoPFS/libhdmlp/build/libhdmlp_docker.so \
--data-dir=/home/zzp/code/NoPFS/data2/imagenet_mini \
--hdmlp-config-path=/home/zzp/code/NoPFS/libhdmlp/data/hdmlp_docker.cfg \
--dist --r=tcp --dist-rank=0 --dist-size=1

## 多机
python /home/zzp/code/NoPFS/benchmark/resnet50_docker.py \
--job-id=1 --hdmlp --no-eval --epochs=2 --hdmlp-stats \
--output-dir=/home/zzp/code/NoPFS/output/log \
--hdmlp-lib-path=/home/zzp/code/NoPFS/libhdmlp/build/libhdmlp.so \
--data-dir=/home/zzp/code/NoPFS/data/imagenet_mini \
--hdmlp-config-path=/home/zzp/code/NoPFS/configure/hdmlp_docker.cfg \
--dist --r=tcp --dist-rank=1 --dist-size=3

## 缓冲
python /home/zzp/code/NoPFS/benchmark/resnet50.py \
--job-id=1 --no-eval --epochs=2 --hdmlp-stats \
--output-dir=/home/zzp/code/NoPFS/output/log \
--hdmlp-lib-path=/home/zzp/code/NoPFS/libhdmlp/build/libhdmlp.so \
--data-dir=/home/zzp/code/NoPFS/data2/imagenet_mini \
--hdmlp-config-path=/home/zzp/code/NoPFS/configure/hdmlp_docker.cfg \
--dist --r=tcp --dist-rank=0 --dist-size=1


## 选项
--dist --r=tcp 采用多机训练
--hdmlp-stats 打印hdmlp状态

## 编译hdmlp(注意在docker内和外的lib不通用)
cd /home/zzp/code/NoPFS/libhdmlp/build && cmake .. && make -j

## 激活conda(docker不适用)
source activate py36

