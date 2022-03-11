# 删除临时文件,包括mount.sh torch-test.tar.gz和torch-test文件夹
node_num=3

base=$(cd ../../..;pwd)
rm -f $base/NoPFS/torch-test.tar.gz
for host in `cat machines | head -n $node_num`; do
    echo "start $host"

    echo "delete mount.sh"
    ssh $host "cd $base && rm -f $base/mount.sh"
    echo "done"

    echo "delete torch-test.tar.gz"
    ssh $host "cd $base && rm -f $base/torch-test.tar.gz"
    echo "done"

    echo "delete torch-test directory"
    ssh $host "cd $base && rm -rf torch-test"
    echo "done"
    
done