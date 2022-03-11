set -x
rank=0
num_world=3

for each in 10.0.2.180 10.0.2.181 10.0.2.182; do
	echo start rank $rank
	ssh $each "nohup /home/zzp/anaconda3/envs/py36/bin/python \
	/home/zzp/code/NoPFS/main/benchmark/resnet50.py \
	--job-id=1 --hdmlp --no-eval --batch-size=60 --epochs=2 --hdmlp-stats \
	--output-dir=/home/zzp/code/NoPFS/output/log \
	--hdmlp-lib-path=/home/zzp/code/NoPFS/libhdmlp/build/libhdmlp.so \
	--data-dir=/home/zzp/code/NoPFS/data/imagenet_mini \
	--hdmlp-config-path=/home/zzp/code/NoPFS/main/libhdmlp/data/hdmlp.cfg \
	--dist --r=tcp --dist-rank=${rank} --dist-size=${num_world} \
	--file-name=/home/zzp/code/NoPFS/data/init_1 \
	--backend=nccl \
	1>/home/zzp/code/NoPFS/output/log/${each}.log 2>&1 &"
	rank=$((rank+1))

	if [ ${rank} == ${num_world} ]; then
		break
	fi
done
