# 运行resnet50.py
container_name=torchtest
image_name=torchtest
main_node=10.0.2.180 # NoPFS的宿主机
data_node=10.0.2.138
data_path=/disk

base=$(cd ../../..;pwd)
epochs=1
batch_size=60
# 将所有机器进行初始化,包括挂载和容器的创建
echo "----------------------------------init sshfs and docker----------------------------------"
for host in `cat machines`; do
    echo start $host

    # 运行挂载脚本(只需要挂载在宿主机上即可)
    echo "< copy mount.sh >"
    if [ $host != $main_node ]; then
        ssh $host "scp $main_node:$base/NoPFS/benchmark/test/mount.sh $base/mount.sh && cd $base && sh $base/mount.sh" 
        ssh $host "rm -f $base/mount.sh"
    else
        sshfs -o nonempty,exec,allow_other $USER@$data_node:$data_path $base/NoPFS/data
    fi
    echo "< done >"

    # 创建容器# --gpus=all \
    echo "< create container >"
    ssh $host "sudo docker run --rm --cap-add=SYS_ADMIN \
    --privileged \
    --security-opt seccomp=unconfined \
    --runtime=nvidia \
    --ulimit memlock=-1 \
    --net=host \
    -v $base/NoPFS/:$base/NoPFS/ \
    -it -d \
    --name ${container_name} \
    ${image_name}:latest "
    echo "<done>"
done

# x=1
# for i in {0..$(($x-1))}; do 
# echo $i 
# done

# 主机编译libhdmlp
echo "< compiling hdmlp >"
echo "cd $base/NoPFS/libhdmlp && source ~/.bashrc && make clean && make all > /dev/null" > $base/NoPFS/benchmark/test/temp/compile.sh
ssh $main_node "sudo docker exec -i ${container_name} /bin/bash $base/NoPFS/benchmark/test/temp/compile.sh"
echo "< done >"

# 开始有hdmlp的1/2/3机器的1/2/4/8卡训练
job_id=0
echo "this is a sheet" > $base/NoPFS/benchmark/test/config-sheet # job-id与各个配置的对应表
for dataset in imagenet; do
    for world_size in 1 2 3; do
        for local_size in 1 2 4 8; do
            job_no=0 # 用于记录开了多少个任务
            world_rank=0
            echo "< create and run job script >"
            echo "dataset = ${dataset} / world_size = ${world_size} / local_size = ${local_size}"
            for host in `cat machines | head -n $world_size`; do
                ssh $host "cd $base/NoPFS/benchmark/test/script && mkdir -p dataset-${dataset}/world_size-${world_size}/local_size-${local_size}/world_rank-${world_rank} && chmod -R 777 dataset-${dataset}/world_size-${world_size}/local_size-${local_size}/world_rank-${world_rank}"     # 在script中创建脚本目录 world_size/local_size
                ssh $host "cd $base/NoPFS/output/log && mkdir -p dataset-${dataset}/world_size-${world_size}/local_size-${local_size}/world_rank-${world_rank} && chmod -R 777 dataset-${dataset}/world_size-${world_size}/local_size-${local_size}/world_rank-${world_rank}"                  # 创建日志的目录
                
                # 生成脚本并执行
                for ((local_rank=0; local_rank < $local_size; local_rank++)); do
                    ssh $host "cd $base/NoPFS/benchmark/test/script/dataset-${dataset}/world_size-${world_size}/local_size-${local_size}/world_rank-${world_rank} && mkdir -p local_rank-${local_rank} && chmod 777 local_rank-${local_rank}" # 同理创建local_rank的文件
                    ssh $host "cd $base/NoPFS/output/log/dataset-${dataset}/world_size-${world_size}/local_size-${local_size}/world_rank-${world_rank} && mkdir -p local_rank-${local_rank} && chmod 777 local_rank-${local_rank}" #日志先写入日期
                    echo "export hdmlp_path=$base/NoPFS && \
                    nohup python $base/NoPFS/benchmark/resnet50.py \
                    --job-id=${job_id} --no-eval --hdmlp-stats --dist --r=tcp --backend=nccl --print-freq=1 --save-stats \
                    --hdmlp \
                    --hdmlp-lib-path=$base/NoPFS/libhdmlp/build/libhdmlp.so \
                    --hdmlp-config-path=$base/NoPFS/libhdmlp/data/hdmlp.cfg \
                    --dataset=${dataset} --epochs=${epochs} --batch-size=${batch_size} \
                    --data-dir=$base/NoPFS/data/${dataset} \
                    --output-dir=$base/NoPFS/output/log/dataset-${dataset}/world_size-${world_size}/local_size-${local_size}/world_rank-${world_rank}/local_rank-${local_rank} \
                    --world-rank=${world_rank} --world-size=${world_size} \
                    --local-rank=${local_rank} --local-size=${local_size} \
                    --init-path=$base/NoPFS/data/init_${job_id} \
                    --profiler-path=$base/NoPFS/output/log/dataset-${dataset}/world_size-${world_size}/local_size-${local_size}/world_rank-${world_rank}/local_rank-${local_rank}/trace \
                    >> $base/NoPFS/output/log/dataset-${dataset}/world_size-${world_size}/local_size-${local_size}/world_rank-${world_rank}/local_rank-${local_rank}/${job_id}.log 2>&1 & \
                    " > $base/NoPFS/benchmark/test/script/dataset-${dataset}/world_size-${world_size}/local_size-${local_size}/world_rank-${world_rank}/local_rank-${local_rank}/${job_id}.sh
                   
                    job_no=$((${job_no}+1))
                    if [ $job_no == $(($local_size*$world_size)) ];then # 最后一个脚本进行wait
                        echo "< waiting job >"
                        echo "wait" >> $base/NoPFS/benchmark/test/script/dataset-${dataset}/world_size-${world_size}/local_size-${local_size}/world_rank-${world_rank}/local_rank-${local_rank}/${job_id}.sh
                    else
                        echo "create next"
                    fi

                    # 运行脚本
                    chmod u+x $base/NoPFS/benchmark/test/script/dataset-${dataset}/world_size-${world_size}/local_size-${local_size}/world_rank-${world_rank}/local_rank-${local_rank}/${job_id}.sh # 修改权限
                    ssh $host "sudo docker container exec -i ${container_name} /bin/bash $base/NoPFS/benchmark/test/script/dataset-${dataset}/world_size-${world_size}/local_size-${local_size}/world_rank-${world_rank}/local_rank-${local_rank}/${job_id}.sh"
                done # 单机上每个脚本都已经创建好运行了
                wait
                world_rank=$((${world_rank}+1)) # 此时换到下一个机器上
            done # 所有机器的每个脚本都创建并运行了，即该job已经运行了
            cd $base/NoPFS/benchmark/test && echo "job_id = ${job_id} / dataset = ${dataset} / world_size = ${world_size} / local_size = ${local_size}" >> config-sheet
            job_id=$((job_id+1)) # 即需要换到下一个job上
            echo "<done>"
        done
    done
done # 创建了容器和各自的训练脚本未删除
echo "----------------------------------work done----------------------------------"