# 清除log和生成的脚本
# 注意：该脚本会全部清除所有生成的数据，仅在测试中使用
base=$(cd ../../..;pwd)
rm -rf $base/NoPFS/benchmark/test/script/dataset-imagenet*

rm -rf $base/NoPFS/output/log/dataset-imagenet*

rm -rf $base/NoPFS/benchmark/test/config-sheet

rm -rf $base/NoPFS/benchmark/test/temp/*

for host in `cat machines`; do
    ssh $host "sudo docker stop torchtest"
done