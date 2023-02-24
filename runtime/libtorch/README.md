# 使用说明
与WeNet官方有所区别，我们针对模型架构等做了一些调整，核心推理代码也有所改变。
因此部署过程也需要做出相应的调整，需要使用内部的代码https://git.xmov.ai/dujing/asr-online。

## 一、推荐：本地编译作为服务端，构建websocket流式服务
步骤如下：
```shell
cd libtorch
mkdir build && cd build && cmake .. && cmake --build .
```
  
如果需要编译GPU版本
```shell
mkdir build && cd build && cmake -DGPU=ON .. && cmake --build .
```

然后将模型资源从S:\users\dujing\asr-online\resource拷贝到当前项目根目录（默认为asr-online）下
  
再在libtorch目录下运行
```shell
bash ./run_asr_server.sh
```  
最后用浏览器打开libtorch/web/templates/index.html, 将ip替换为启动服务的服务器ip,即可开始流式识别。

## 二、构建docker镜像进行部署，Docker也可以作为服务端
步骤如下：

1.克隆代码（由于是私库，所以放到dockerfile中克隆会引起不必要的麻烦）
```shell
# `pwd`=asr-online/libtorch 当前路径为libtorch下
cd docker 
# 首先在机器上配置好公司内部的git账号，才能正常下载
# 这里确定docker/asr-online 为代码临时保存路径，方便直接打包到镜像中
git clone git@git.xmov.ai:dujing/asr-online.git asr-online
```
2.构建容器 
```shell
# Dockerfile中将运行路径设置为/workspace/asr-online
# Dockerfile默认编译GPU版本，如需编译CPU版本，则需要在Dockerfile中删除”-DGPU=ON“编译选项
docker build --no-cache -t wenet:latest .
```
3.准备所需的资源
```shell
cd ../..
# `pwd`=asr-online, 当前路径回到asr-online根目录
mkdir -p resource #所需资源放置在根目录下比较合适
# 所需模型资源放置在共享路径S:\users\dujing\asr-online\resource
cp -r <your_resource_dir> resource
```
4.启动容器，并映射路径.
```shell
#`pwd`=asr-online 
docker run --rm -v $PWD/resource:/workspace/asr-online/resource -it wenet bash
```
5.容器中直接测试
```shell
cd /workspace/asr-online/libtorch
export GLOG_logtostderr=1
export GLOG_v=2
#wav_path=../resource/WAV/19louzhibo.wav
wav_dir=../resource/WAV/test_xmov_youling
wav_scp=../resource/WAV/test_xmov_youling.list
find $wav_dir -name "*.wav" | awk -F"/" -v name="" \
  '{name=$NF; gsub(".wav", "", name); print name" "$0 }' | sort > $wav_scp
model_dir=../resource/ASR
./build/bin/decoder_main \
    --chunk_size 16 \
    --min_trailing_silence 800 \
    --wav_scp $wav_scp \
    --model_path $model_dir/final.zip \
    --unit_path $model_dir/units.txt 2>&1 | tee log.txt
```
6.或者映射容器端口(demo监听容器的10086,将其映射成宿主的8086端口)，在容器中启动websocket服务，供客户端调用。
```shell
docker run --rm -p 8086:10086 -v $PWD/resource:/workspace/asr-online/resource -it wenet bash
cd /workspace/asr-online/libtorch
bash ./run_asr_server.sh
```
7.最后用浏览器打开libtorch/web/templates/index.html, 将ip替换为启动服务的服务器ip:8086,即可开始流式识别。


# WeNet Server (x86) ASR Demo

**[中文版:x86 平台上使用 WeNet 进行语音识别](./README_CN.md)**

## Run with Prebuilt Docker

* Step 1. Download pretrained model(see the following link) or prepare your trained model.

[AISHELL-1](https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell/20210601_u2%2B%2B_conformer_libtorch.tar.gz)
| [AISHELL-2](https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell2/20210618_u2pp_conformer_libtorch.tar.gz)
| [GigaSpeech](https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/gigaspeech/20210728_u2pp_conformer_libtorch.tar.gz)
| [LibriSpeech](https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/librispeech/20210610_u2pp_conformer_libtorch.tar.gz)
| [Multi-CN](https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/multi_cn/20210815_unified_conformer_libtorch.tar.gz)


* Step 2. Start docker websocket server. Here is a demo.

``` sh
model_dir=$PWD/20210602_u2++_conformer_libtorch  # absolute path
docker run --rm -it -p 10086:10086 -v $model_dir:/home/wenet/model wenetorg/wenet-mini:latest bash /home/run.sh
```

* Step 3. Test with web browser. Open runtime/libtorch/web/templates/index.html in the browser directly, input your `WebSocket URL`, it will request some permissions, and start to record to test, as the following graph shows.

![Runtime web](../../docs/images/runtime_web.png)

## Run in Docker Build

We recommend using the docker environment to build the c++ binary to avoid
system and environment problems.

* Step 1. Build your docker image.

``` sh
cd docker
docker build --no-cache -t wenet:latest .
```

* Step 2. Put all the resources, like model, test wavs into a docker resource dir.

``` sh
mkdir -p docker_resource
cp -r <your_model_dir> docker_resource/model
cp <your_test_wav> docker_resource/test.wav
```

* Step 3. Start docker container.
``` sh
docker run --rm -v $PWD/docker_resource:/home/wenet/runtime/libtorch/docker_resource -it wenet bash
```

* Step 4. Testing in docker container
```
cd /home/wenet/runtime/libtorch
export GLOG_logtostderr=1
export GLOG_v=2
wav_path=docker_resource/test.wav
model_dir=docker_resource/model
./build/bin/decoder_main \
    --chunk_size -1 \
    --wav_path $wav_path \
    --model_path $model_dir/final.zip \
    --unit_path $model_dir/units.txt 2>&1 | tee log.txt
```

Or you can do the WebSocket server/client testing as described in the `WebSocket` section.

## Run with Local Build

* Step 1. Download or prepare your pretrained model.

* Step 2. Build. The build requires cmake 3.14 or above. For building, please first change to `wenet/runtime/libtorch` as your build directory, then type:

``` sh
mkdir build && cd build && cmake .. && cmake --build .
```

For building with GPU, you should turn on `GPU`:

``` sh
mkdir build && cd build && cmake -DGPU=ON .. && cmake --build .
```

* Step 3. Testing, the RTF(real time factor) is shown in the console.

``` sh
export GLOG_logtostderr=1
export GLOG_v=2
wav_path=your_test_wav_path
model_dir=your_model_dir
./build/bin/decoder_main \
    --chunk_size -1 \
    --wav_path $wav_path \
    --model_path $model_dir/final.zip \
    --unit_path $model_dir/units.txt 2>&1 | tee log.txt
```


## Advanced Usage

### WebSocket

* Step 1. Download or prepare your pretrained model.
* Step 2. Build as in `Run with Local Build`
* Step 3. Start WebSocket server.

``` sh
export GLOG_logtostderr=1
export GLOG_v=2
model_dir=your_model_dir
./build/bin/websocket_server_main \
    --port 10086 \
    --chunk_size 16 \
    --model_path $model_dir/final.zip \
    --unit_path $model_dir/units.txt 2>&1 | tee server.log
```
* Step 4. Start WebSocket client.

```sh
export GLOG_logtostderr=1
export GLOG_v=2
wav_path=your_test_wav_path
./build/websocket_client_main \
    --hostname 127.0.0.1 --port 10086 \
    --wav_path $wav_path 2>&1 | tee client.log
```

You can also start WebSocket client by web browser as described before.

Here is a demo for command line based websocket server/client interaction.

![Runtime server demo](../../../docs/images/runtime_server.gif)

### gRPC

Why grpc? You may find your answer in https://grpc.io/.
Please follow the following steps to try gRPC.

* Step 1. Download or prepare your pretrained model.
* Step 2. Build
``` sh
mkdir build && cd build && cmake -DGRPC=ON .. && cmake --build .
```
* Step 3. Start gRPC server

``` sh
export GLOG_logtostderr=1
export GLOG_v=2
model_dir=your_model_dir
./build/bin/grpc_server_main \
    --port 10086 \
    --workers 4 \
    --chunk_size 16 \
    --model_path $model_dir/final.zip \
    --unit_path $model_dir/units.txt 2>&1 | tee server.log
```

* Step 4. Start gRPC client.

```sh
export GLOG_logtostderr=1
export GLOG_v=2
wav_path=your_test_wav_path
./build/bin/grpc_client_main \
    --hostname 127.0.0.1 --port 10086 \
    --wav_path $wav_path 2>&1 | tee client.log
```

