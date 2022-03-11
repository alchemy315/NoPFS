# 在base中调用,如/home/zzp/code上
main_node=10.0.2.180
data_node=10.0.2.138
data_path=/disk
# user=zzp

base=$(pwd)
# source ~/.bashrc
mkdir -p NoPFS
sshfs -o nonempty,exec,allow_other $USER@$main_node:$base/NoPFS $base/NoPFS
sshfs -o nonempty,exec,allow_other $USER@$data_node:$data_path $base/NoPFS/data

# sshfs -o nonempty,exec,allow_other zzp@10.0.2.138:/disk /home/zzp/code/NoPFS/data