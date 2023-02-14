FROM ubuntu:latest
MAINTAINER <zhendong.peng@mobvoi.com>
ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i s@/archive.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g /etc/apt/sources.list
RUN apt-get update && apt-get install -y git cmake wget build-essential
RUN git clone http://git.xmov.ai/dujing/asr-online.git /home/asr-online   #将仓库代码拉到/home/asr-online

RUN mkdir -p /home/asr-online/resource
# 下面的模型文件是开源的，可以下载下来试运行
##ARG model=20210618_u2pp_conformer_libtorch.tar.gz
##RUN wget -P /home https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell2/$model
##RUN tar -xzf /home/$model -C /home/asr-online/resource

#直接从内部共享盘下载文件S:\users\dujing\asr-online\resource
#将此文件夹下的ASR和LM都复制到/home/asr-online/resource


ARG build=/home/asr-online/libtorch/build
RUN mkdir -p $build && cd $build
RUN cmake -B $build -DCMAKE_BUILD_TYPE=Release -DGRAPH_TOOLS=ON && cmake --build $build

#编译好之后，cd到/home/asr-online/libtorch，运行run_asr_server.sh
#然后打开libtorch/web/templates/index.html, 将ip替换为启动服务的服务器ip,即可开始流式识别。