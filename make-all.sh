set -x
rank=0
num_world=3

for each in 10.0.2.180 10.0.2.181 10.0.2.182; do
	echo start make $rank
	ssh $each "cd /home/zzp/code/NoPFS/libhdmlp && make clean && make all"
	rank=$((rank+1))

	if [ ${rank} == ${num_world} ]; then
		break
	fi
done
