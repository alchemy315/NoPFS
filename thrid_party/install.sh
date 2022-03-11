# 脚本版
# 1.加速更新
# echo -e "\
# channels:\n\
#   - defaults\n\
# show_channel_urls: true\n\
# default_channels:\n\
#   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main\n\
#   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r\n\
#   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2\n\
# custom_channels:\n\
#   conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
#   msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
#   bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
#   menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
#   pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
#   pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
#   simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" | sudo tee ~/.condarc

# 2.卸载原生的totch
# yes | conda uninstall pytorch -y
# yes | pip uninstall torch

# 3.pytorch编译安装
# cd /home/zzp/code/NoPFS/torch/pytorch
# yes | pip install typing_extensions
# export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
# sudo DEBUG=1 /opt/conda/bin/python setup.py develop --cmake

# 4.安装torchvision
yes | pip install --no-deps torchvision

# 5.安装sshfs
sudo apt-get -y install sshfs

# 6.安装apex
cd /home/zzp/code/NoPFS/main/thrid_party/apex
sudo /opt/conda/bin/python setup.py install --cpp_ext --cuda_ext

# 7.opencv安装
sudo apt-get -y install software-properties-common
sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
sudo apt update
sudo apt-get -y install cmake pkg-config
sudo apt-get -y install build-essential libgtk2.0-dev libavcodec-dev
sudo apt-get -y install libavformat-dev libjpeg.dev libtiff4.dev libswscale-dev
sudo apt-get -y install libjasper-dev
cd /home/zzp/code/NoPFS/thrid_party/opencv
sudo rm -rf build
mkdir build
cd build
sudo mkdir /usr/local/opencv
sudo   cmake -D WITH_TBB=ON -D WITH_EIGEN=ON -D OPENCV_GENERATE_PKGCONFIG=ON  -D BUILD_DOCS=ON -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF  -D WITH_OPENCL=OFF -D WITH_CUDA=OFF -D BUILD_opencv_gpu=OFF -D BUILD_opencv_gpuarithm=OFF -D BUILD_opencv_gpubgsegm=O -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local /opencv ..
sudo make install

# 8. 安装mpich
sudo mkdir /usr/local/mpich
cd /home/zzp/code/NoPFS/thrid_party/mpich-3.4.3
autoreconf -ivf
sudo ./configure --prefix=/usr/local/mpich
sudo make
sudo make install

# 9.安装libconfig
cd /home/zzp/code/NoPFS/thrid_party/libconfig-1.7.3
sudo mkdir /usr/local/libconfig
sudo ./configure --prefix="/usr/local/libconfig"
sudo apt-get -y install texinfo
sudo make
sudo make install

# 10.安装hdf5
cd /home/zzp/code/NoPFS/thrid_party/hdf5-1.13.0
sudo mkdir /usr/local/hdf5
sudo apt-get -y install libhdf5-dev
sudo ./configure --prefix=/usr/local/hdf5 --enable-cxx --enable-build-mode=production
sudo make
sudo make install

# 11.环境变量
echo -e "\nexport PKG_CONFIG_PATH=\$PKG_CONFIG_PATH:/usr/local/opencv/lib/pkgconfig\n\
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/opencv/lib\n\
export PATH=\$PATH:/usr/local/mpich/bin\n\
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/mpich/lib\n\
export PATH=\$PATH:/usr/local/libconfig/bin\n\
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/libconfig/lib\n\
export PATH=\$PATH:/usr/local/hdf5/bin\n\
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/hdf5/lib" | sudo tee -a ~/.bashrc
source ~/.bashrc

# 12.动态链接库
echo "/usr/local/opencv/lib" | sudo tee /etc/ld.so.conf.d/opencv.conf
echo "/usr/local/hdf5/lib" | sudo tee /etc/ld.so.conf.d/h5.conf
sudo ldconfig

# 13. 再次安装pytorch
# export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
# sudo DEBUG=1 /opt/conda/bin/python setup.py develop --cmake

# 其他python包
pip install Pillow
pip install numpy
pip install dataclasses
pip install PyYAML
pip install typing_extensions
