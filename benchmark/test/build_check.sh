# 检查image是否构建完成
node_num=3

base=$(cd ../../..;pwd)
for host in `cat machines | head -n $node_num`; do
    echo "check ${host}"
    ssh $host "cat $base/torch-test/$host* | sed '$!N;$!D'"
done