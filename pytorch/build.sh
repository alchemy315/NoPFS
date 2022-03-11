base=$(pwd)
DEBUG=1 date > 10.0.2.180.build.log && nohup /opt/conda/bin/python setup.py develop --cmake >> 10.0.2.180.build.log 2>&1 & 