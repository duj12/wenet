FROM ubuntu:latest
MAINTAINER <dujing@xmov.ai>
ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i s@/archive.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g /etc/apt/sources.list
RUN apt-get update && apt-get install -y git cmake wget build-essential
WORKDIR /workspace
COPY asr-online /workspace/asr-online
ARG build=/workspace/asr-online/libtorch/build
RUN mkdir -p $build && cd $build && cmake -DCMAKE_BUILD_TYPE=Release -DGRAPH_TOOLS=ON -DGPU=ON .. && cmake --build .