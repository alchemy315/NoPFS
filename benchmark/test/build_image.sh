# 将torch文件发往目标机器并进行编译
image_name=torchtest
node_num=3
main_node=10.0.2.180 # 将文件从这个机器发送docker到其他机器并编译

base=$(cd ../../..;pwd)
echo $base
echo "-----------------------------work begin-----------------------------"
echo "packing torch-test"
ssh $main_node "cd $base/NoPFS && tar -zcvf torch-test.tar.gz torch-test > /dev/null" # 打包torch-test文件夹
echo "done"
for host in `cat machines | head -n $node_num`; do
    echo start $host

    if [ $host != $main_node ]; then
        echo "send mount.sh"
        ssh $host "scp ${main_node}:${base}/NoPFS/benchmark/test/mount.sh $base/mount.sh > /dev/null" 
        echo "done"
        echo "run mount.sh"
        ssh $host "cd $base && sh mount.sh"
        echo "done"
        ssh $host "rm -f $base/mount.sh"
    fi
    echo "receive package"
    ssh $host "cd $base && cp NoPFS/torch-test.tar.gz $base && tar zxvf torch-test.tar.gz > /dev/null && rm -f torch-test.tar.gz" # 复制打包文件移出并解压
    echo "done"

    echo "build image"
    ssh $host "cd $base/torch-test && date > $host.build.log && nohup sudo docker build -t $image_name . >> $base/torch-test/${host}.build.log 2>&1 &"
    echo "hanging on"
done
ssh $main_node "sudo rm -f $base/NoPFS/torch-test.tar.gz" # 删除打包文件
echo "-----------------------------work done-----------------------------"

# 每个机器的/home/zzp/code/torch/$host.build.log即为输出日志
# ssh $host cat $base/10* | sed '$!N;$!D'
