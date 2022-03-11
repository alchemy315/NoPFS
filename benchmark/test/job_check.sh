# 检查image是否构建完成,位于/home/zzp/code/NoPFS/output/log/*
node_num=3

base=$(cd ../../..;pwd)
for host in `cat machines | head -n $node_num`; do
    echo "check ${host}"
    ssh $host "cat $base/NoPFS/output/log/${host}* | sed '$!N;$!D'"
done