# 运行resnet50.py
node_num=1
container_name=zzp6
image_name=zzp2
main_node=10.0.2.180 # NoPFS的宿主机
data_node=10.0.2.138
data_path=/disk
dataset=imagenet
rank=0
base=$(cd ../../..;pwd)
echo "----------------------------------work begin----------------------------------"
for host in `cat machines | head -n $node_num`; do
    echo start $host

    # 运行挂载脚本(只需要挂载在宿主机上即可)
    echo "-------------------------------copy mount file--------------------------------"
    if [ $host != $main_node ]; then
        ssh $host "scp $main_node:$base/NoPFS/benchmark/test/mount.sh $base/mount.sh && cd $base && sh $base/mount.sh" 
        ssh $host "rm -f $base/mount.sh"
    else
        sshfs -o nonempty,exec,allow_other $USER@$data_node:$data_path $base/NoPFS/data
    fi
    echo "done"

    # 创建容器# --gpus=all \
    echo "-------------------------------create container-------------------------------"
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
    
    # 生成脚本
    echo "-------------------------------create job script------------------------------"
    echo "nohup python $base/NoPFS/benchmark/resnet50.py \
    --job-id=1 --hdmlp --no-eval --batch-size=60 --epochs=2 --hdmlp-stats \
    --output-dir=$base/NoPFS/output/cache \
    --hdmlp-lib-path=$base/NoPFS/libhdmlp/build/libhdmlp.so \
    --print-freq=1 \
    --save-stats \
    --dataset=${dataset} \
    --data-dir=$base/NoPFS/data/imagenet_mini \
    --hdmlp-config-path=$base/NoPFS/libhdmlp/data/hdmlp.cfg \
    --dist --r=tcp --dist-rank=${rank} --dist-size=${node_num} \
    --file-name=$base/NoPFS/data/init_1 \
    --profiler-path=$base/NoPFS/output/trace/${host}.trace \
    --backend=nccl \
    > $base/NoPFS/output/log/${host}.log 2>&1 & \
    " | sudo tee $base/NoPFS/benchmark/test/${host}.sh
    sudo chmod 777 $base/NoPFS/benchmark/test/${host}.sh # 修改权限
    
    # 运行脚本
    echo "----------------------------------run job-------------------------------------"
    ssh $host "sudo docker container exec -i ${container_name} /bin/bash $base/NoPFS/benchmark/test/${host}.sh"

    # 删除
    echo "---------------------------------delete file----------------------------------"
    # ssh $host sudo docker stop $container_name              # 删除容器
    sudo rm /home/zzp/code/NoPFS/benchmark/test/${host}.sh  # 删除任务脚本
    # echo "done"
    rank=$((rank+1))

done

# 创建了容器和各自的训练脚本未删除

echo "----------------------------------work done----------------------------------"
