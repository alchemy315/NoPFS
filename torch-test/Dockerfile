# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# cd /home/zzp/code/NoPFS/torch && sudo docker build -t zzp2 .
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel AS devel

# ARG USERNAME=web_server
ARG USERNAME=zzp

# ENV http_proxy=http://10.74.176.8:11080
# ENV https_proxy=http://10.74.176.8:11080
# ENV no_proxy=localhost,127.0.0.1,kwaidc.com
# ENV CC=/usr/bin/gcc
# ENV CXX=/usr/bin/g++

COPY sources.list /etc/apt/sources.list

RUN apt-get update -y && \
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      vim gdb git wget tar unzip curl clang-format libboost-all-dev apt-utils&& \
      rm -rf /var/lib/apt/lists/* && \
apt-get update -y && DEBIAN_FRONTEND="noninteractive" apt-get install -y zsh google-perftools openssh-server \
  cloc \
  check \
  sudo \
  libtbb-dev \
  lsof \
  cmake && \
echo "${USERNAME}	ALL=(ALL) NOPASSWD:ALL" >>/etc/sudoers && \
echo "root:1234" | chpasswd && \
useradd --uid=1001 --shell=/bin/bash ${USERNAME} && \
ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo 'Asia/Shanghai' >/etc/timezone && \
sed -i 's/#Port 22/Port 22333/g' /etc/ssh/sshd_config

EXPOSE 22333/tcp
EXPOSE 22333/udp

RUN mkdir /home/${USERNAME} && \
chown -R 1001:1001 /home/${USERNAME} && \
service ssh restart

USER 1001:1001
WORKDIR /home/${USERNAME}
RUN mkdir .ssh && \
echo 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQDogir2qzcJazWaQPniVhhXlPwV1lakJZYAsqetEASG85wulddYlbNbJRQwfIGSgYK9Uo1mMer4IK3WdkSIOpsdkFv5ja3zAmThFk14kEezwkUbK20+zdEJL4TWI1gNW4st6I3wlf1zRum0dSxfIvDH0si4be8DYsFSCl1R0IZZpymjSdjhOy434ATKoL23vMZC86ZbkJ3ZleBR8YggHFI7+OcCqip4u32BVLQAViMRykCmXQl8VGTtEwwseqPC3Ab84ZZoO7VE32Z6ARO6YNjtCW030NsaZVjuH+cO0IVV1vDGZIikfQTevlVkNNC42eoaaFz7RmWzIT3hdQawOvhkUPLrUVYD1+Sy+WPGLsUvTPFrazx014HHrE5FARSXsMNhXhQ/Kl4/rK1lDdUiIS/kmvnDgeA3oPBWc/Y7lkNvdeqSMGR6gWBB36ZF6EFodxsBWzxNX8QwTd2Cq0MsZKYh/rdfwTOYDXX+3WEvlQfPiwj5aaeeaCKkZl0zQB4pprYk4wOupZ1wJNjcguWj17EwkrRompT8U+1yREShHoiRTdWo/N+qJzlCdPGeLrlo+LFRFccg/7O06ladAUhQGiJqa0x1yGtWeLSpia2v3mag8v5eB7q5DTufl9tOp0mbDRcQFkLPu0M2oM7KVflVHclIvOAfhvzrXLal379mGPclMQ== Email' >>.ssh/authorized_keys && \
chmod 600 .ssh/authorized_keys


# RUN http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890  git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf && ~/.fzf/install


COPY fzf /home/${USERNAME}/.fzf
# RUN /home/${USERNAME}/.fzf/install

## ssh
RUN alias sudo='sudo PATH="$PATH" HOME="$HOME" LD_LIBRARY_PATH="$LD_LIBRARY_PATH"' && \
sudo service ssh restart && \
sudo chown -R ${USERNAME}:${USERNAME} /opt/conda/ && \
sudo apt-get -y install sshfs

# 加速更新
RUN echo "\
channels:\n\
  - defaults\n\
show_channel_urls: true\n\
default_channels:\n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main\n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r\n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2\n\
custom_channels:\n\
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" | sudo tee ~/.condarc

# # 安装apex/opencv
COPY apex /home/${USERNAME}/apex
RUN cd /home/${USERNAME}/apex && sudo /opt/conda/bin/python setup.py install --cpp_ext --cuda_ext && \
sudo apt-get -y install software-properties-common && \
sudo apt-get -y install cmake pkg-config && \
sudo apt-get -y install build-essential libgtk2.0-dev libavcodec-dev && \
sudo apt-get -y install libavformat-dev libjpeg.dev libtiff4.dev libswscale-dev && \
sudo apt-get -y install texinfo && \
sudo apt-get -y install libhdf5-dev && \
yes | sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main" && \
sudo apt-get -y update && sudo apt-get -y upgrade && \
sudo apt-get -y install libjasper-dev
# yes | sudo apt update && yes | sudo apt upgrade \
# # 安装opencv
COPY opencv /home/${USERNAME}/opencv
RUN cd /home/${USERNAME}/opencv && sudo rm -rf build && sudo mkdir build && cd build && \
sudo mkdir /usr/local/opencv && \
sudo cmake -D WITH_TBB=ON -D WITH_EIGEN=ON -D OPENCV_GENERATE_PKGCONFIG=ON  -D BUILD_DOCS=ON -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF  -D WITH_OPENCL=OFF -D WITH_CUDA=OFF -D BUILD_opencv_gpu=OFF -D BUILD_opencv_gpuarithm=OFF -D BUILD_opencv_gpubgsegm=O -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local /opencv .. && \
sudo make install

# # 安装mpich
COPY mpich-3.4.3 /home/${USERNAME}/mpich
RUN sudo mkdir /usr/local/mpich && \
cd /home/${USERNAME}/mpich && \
sudo ./configure --prefix=/usr/local/mpich --disable-fortran  --with-device=ch4:ofi && \
sudo autoreconf -ivf && \
sudo make && \ 
sudo make install

# # 安装libconfig
COPY libconfig-1.7.3 /home/${USERNAME}/libconfig
RUN cd /home/${USERNAME}/libconfig && \
sudo mkdir /usr/local/libconfig && \
sudo ./configure --prefix="/usr/local/libconfig" && \
sudo make && \
sudo make install
# texinfo提前到前面安装了

# # 安装hdf5
COPY hdf5-1.13.0 /home/${USERNAME}/hdf5
RUN cd /home/${USERNAME}/hdf5 && \
sudo mkdir /usr/local/hdf5 && \
sudo ./configure --prefix=/usr/local/hdf5 --enable-cxx --enable-build-mode=production && \
sudo make && \
sudo make install
# libhdf5提前到前面装

# # 环境变量
RUN echo "export PKG_CONFIG_PATH=\$PKG_CONFIG_PATH:/usr/local/opencv/lib/pkgconfig\n\
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/opencv/lib\n\
export PATH=\$PATH:/usr/local/mpich/bin\n\
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/mpich/lib\n\
export PATH=\$PATH:/usr/local/libconfig/bin\n\
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/libconfig/lib\n\
export PATH=\$PATH:/usr/local/hdf5/bin\n\
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/hdf5/lib" | sudo tee -a ~/.bashrc && \
echo "/usr/local/opencv/lib" | sudo tee /etc/ld.so.conf.d/opencv.conf && \
echo "/usr/local/hdf5/lib" | sudo tee /etc/ld.so.conf.d/h5.conf && \
echo "user_allow_other" | sudo tee -a /etc/fuse.conf

# # 删除原生mpi
RUN cd /usr/bin/ && \
sudo rm -rf mpiCC* mpic* mpiexec* mpif* mpirun* ompi* osh* && \
cd /etc/alternatives && \
sudo rm -rf *mpi* && \
cd /usr/lib/x86_64-linux-gnu && \
sudo rm -rf libmpi* && \
sudo rm -rf /usr/lib/x86_64-linux-gnu/openmpi

# 编译hdmlp
# RUN cd /home/zzp/code/NoPFS/libhdmlp && \
# make clean && make all
