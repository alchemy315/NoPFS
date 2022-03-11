# name = simulation.py 
# name = resnet50.py
for each in 10.0.2.180 10.0.2.181 10.0.2.182; do
	ssh $each "ps ux |grep resnet50.py  |grep -v grep | awk '{print \$2}'|xargs kill -9"
done
