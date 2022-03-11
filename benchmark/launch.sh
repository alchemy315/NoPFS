#!/bin/sh
sshfs -o nonempty,exec,allow_other zzp@10.0.2.180:/home/zzp/code/NoPFS /home/zzp/code/NoPFS
sshfs -o nonempty,exec,allow_other zzp@10.0.2.138:/disk /home/zzp/code/NoPFS/data

# source activate py36
echo 180start
python /home/zzp/code/NoPFS/benchmark/resnet50.py \
--job-id=1 --hdmlp --no-eval --batch-size=60 --epochs=2 --hdmlp-stats \
--output-dir=/home/zzp/code/NoPFS/output/log \
--hdmlp-lib-path=/home/zzp/code/NoPFS/libhdmlp/build/libhdmlp.so \
--data-dir=/home/zzp/code/NoPFS/data/imagenet_mini \
--hdmlp-config-path=/home/zzp/code/NoPFS/libhdmlp/data/hdmlp.cfg \
--dist --r=tcp --dist-rank=1 --dist-size=2 \
--file-name=/home/zzp/code/NoPFS/data/init_1 \
--backend=nccl
# 299s
# echo "yes"