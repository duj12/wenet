## Using CUDA based Decoders for Triton ASR Server
### Introduction
The triton model repository `model_repo` here, integrates the [CUDA WFST decoder](https://github.com/nvidia-riva/riva-asrlib-decoder) originally described in https://arxiv.org/abs/1910.10032. We take small u2++ fp16 onnx inference for online ASR as an example.

### Quick Start

```sh
# using docker image runtime/gpu/Dockerfile/Dockerfile.server
# OR you can use the following image
# docker pull soar97/triton-wenet:22.12

# pwd is runtime/gpu。onnx_model_dir包含onnx模型和TLG资源
onnx_model_dir=$(pwd)/../resource/ASR/onnx_gpu_model

# 500M的ONNX模型和520M的TLG,显存预计需要10G
# 使用自己的docker镜像wenet_server:22.12，也可以直接用soar97/triton-wenet:22.12
docker run -it --rm --name "wenet_trt_test" \
       --gpus '"device=0"' --shm-size 1g \
       -v $PWD/online_wfst:/ws/model_repo \
       -v $onnx_model_dir:/ws/onnx_model    \
       --net host wenet_server:22.12  
    
# 安装riva-asrlib-decoder, 不同于官方版本，做了流式的更新
cd /ws/model_repo/
git clone --recursive https://github.com/duj12/riva-asrlib-decoder.git
cd riva-asrlib-decoder
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

# 转换模型参数模板，运行服务端
bash /ws/model_repo/convert_start_server.sh
# 目前加载fst会卡住。如果直接运行 python3 /ws/model_repo/wenet/1/decoder.py 则是正常的。

# 如需调试，在docker中进行如下设置
apt install -y openssh-server
echo root:pwd2GPU|chpasswd
echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config
echo "Port 5000" >> /etc/ssh/sshd_config
service ssh restart

# 然后按照gpu/README.md进行客户端的操作即可
```

