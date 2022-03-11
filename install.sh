#!/bin/bash
yes | pip install --no-deps torchvision
yes | sudo apt-get install sshfs
# apex
cd /home/zzp/code/NoPFS/thrid_party/apex
yes | sudo python  setup.py install --cpp_ext --cuda_ext
# opencv
cd /home/zzp/code/NoPFS/thrid_party/opencv
yes | sudo apt-get install software-properties-common
yes | sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
yes | sudo apt update
yes | sudo apt-get install cmake pkg-config
yes | sudo apt-get install build-essential libgtk2.0-dev libavcodec-dev
yes | sudo apt-get install libavformat-dev libjpeg.dev libtiff4.dev libswscale-dev
yes | sudo apt-get install libjasper-dev
sudo rm -rf build
mkdir build
cd build
sudo   cmake -D WITH_TBB=ON -D WITH_EIGEN=ON -D OPENCV_GENERATE_PKGCONFIG=ON  -D BUILD_DOCS=ON -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF  -D WITH_OPENCL=OFF -D WITH_CUDA=OFF -D BUILD_opencv_gpu=OFF -D BUILD_opencv_gpuarithm=OFF -D BUILD_opencv_gpubgsegm=O -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
sudo make install
# openmpi
cd /home/zzp/code/NoPFS/thrid_party/openmpi-4.1.1
sudo ./configure --prefix="/usr/local/openmpi"
sudo make
sudo make install
# libconfig
cd /home/zzp/code/NoPFS/thrid_party/libconfig-1.7.3
sudo ./configure --prefix="/usr/local"
sudo make
sudo make install
# hdf5
cd /home/zzp/code/NoPFS/thrid_party/hdf5-1.13.0
sudo ./configure --prefix=/usr/local/hdf5 --enable-cxx --enable-build-mode=production # 这里需要安装cxx的接口
sudo make
sudo make install
# 环境变量
echo "export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig\n\
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib\n\
export PATH=$PATH:/usr/local/openmpi/bin\n\
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/openmpi/lib\n\
export PATH=$PATH:/usr/local/libconfig/bin\n\
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/libconfig/lib\n\
export PATH=$PATH:/usr/local/hdf5/bin\n\
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/hdf5/lib" | sudo tee -a ~/.bashrc
echo "/usr/local/lib" | sudo tee -a /etc/ld.so.conf.d/opencv.conf
echo "/usr/local/hdf5/lib" | sudo tee -a /etc/ld.so.conf.d/h5.conf
source ~/.bashrc
sudo ldconfig
# hdf5补一个包
yes | sudo apt-get install libhdf5-dev
