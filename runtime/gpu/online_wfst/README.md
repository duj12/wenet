## Using CUDA based Decoders for Triton ASR Server
### Introduction
The triton model repository `model_repo_cuda_decoder` here, integrates the [CUDA WFST decoder](https://github.com/nvidia-riva/riva-asrlib-decoder) originally described in https://arxiv.org/abs/1910.10032. We take small conformer fp16 onnx inference for offline ASR as an example.

### Quick Start

```sh
# using docker image runtime/gpu/Dockerfile/Dockerfile.server
# OR you can use the following image
# docker pull soar97/triton-wenet:22.12

# pwd is runtime/gpu
onnx_model_dir=$(pwd)/../resource/ASR/onnx_gpu_model

#TLG_dir=$(pwd)/../resource/ASR
## cp the TLG.fst and words.fst 
#cp $TLG_dir/TLG.fst $PWD/online_wfst/model_repo/wenet/1/lang/
#cp $TLG_dir/words.txt $PWD/online_wfst/model_repo/wenet/1/lang/

# 500M的ONNX模型和520M的TLG,显存预计需要10G
docker run -it --rm --name "wenet_trt_test" \
       --gpus '"device=0"' --shm-size 1g \
       -v $PWD/online_wfst:/ws/model_repo \
       -v $onnx_model_dir:/ws/onnx_model    \
       --net host wenet_server:22.12  
    
# 安装riva-asrlib-decoder, 不同于官方版本，做了流式的更新
# cd /ws/model_repo/riva-asrlib-decoder
cd /ws/model_repo/
git clone --recursive https://github.com/duj12/riva-asrlib-decoder.git
cd riva-asrlib-decoder
#pip install nanobind
#apt install -y ninja-build
#pip install -e .[testing] -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple


# 先测试online-decoder, 需要远程连接docker环境进行开发
apt install -y openssh-server
echo root:pwd2GPU|chpasswd
echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config
echo "Port 5000" >> /etc/ssh/sshd_config
service ssh restart

# 然后按照gpu/README.md进行客户端的操作即可
```

### TODO: Performance of Small Offline ASR Model using Different Decoders

Benchmark(offline conformer model trained on Aishell1) based on Aishell1 test set with V100, the total audio duration is 36108.919 seconds.

(Note: 80 concurrent tasks, service has been fully warm up.)

|Decoding Method | decoding time(s) | WER (%) |
|----------|--------------------|-------------|
| CTC Greedy Search             | 23s | 4.97  |
| CUDA TLG 1-best (3-gram LM)   | 31s | 4.58  |
