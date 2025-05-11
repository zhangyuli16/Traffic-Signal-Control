FROM ubuntu:18.04

# 避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 安装必要的包
RUN apt-get update && apt-get install -y \
    software-properties-common \
    python3.5 \
    python3-pip \
    wget \
    git \
    xvfb \
    libglu1-mesa-dev \
    libgl1-mesa-dev \
    libosmesa6-dev \
    xorg-dev \
    libx11-dev \
    libxcursor-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxi-dev

# 添加SUMO仓库
RUN add-apt-repository ppa:sumo/stable
RUN apt-get update && apt-get install -y sumo sumo-tools sumo-doc

# 设置SUMO环境变量
ENV SUMO_HOME=/usr/share/sumo

# 安装Python依赖
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow==1.12.0
RUN pip3 install \
    numpy \
    scipy \
    matplotlib \
    pandas

# 工作目录
WORKDIR /app

# 设置入口点
ENTRYPOINT ["/bin/bash"]