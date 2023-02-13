FROM ubuntu:latest
MAINTAINER <zhendong.peng@mobvoi.com>
ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i s@/archive.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g /etc/apt/sources.list
RUN apt-get update && apt-get install -y git cmake wget build-essential
RUN git clone https://github.com/duj12/wenet.git /home/wenet   # 将git仓库地址更新成自有代码
# 下面的模型文件是开源的，可以下载下来试运行， Xmov模型文件联系dujing获取
ARG model=20210618_u2pp_conformer_libtorch.tar.gz
RUN wget -P /home https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell2/$model
RUN tar -xzf /home/$model -C /home
ARG build=/home/wenet/runtime/libtorch/build
RUN cmake -B $build -DCMAKE_BUILD_TYPE=Release -DGRAPH_TOOLS=ON && cmake --build $build
