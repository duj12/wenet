## Using CUDA based Decoders for Triton ASR Server
### Introduction
The triton model repository `model_repo_cuda_decoder` here, integrates the [CUDA WFST decoder](https://github.com/nvidia-riva/riva-asrlib-decoder) originally described in https://arxiv.org/abs/1910.10032. We take small conformer fp16 onnx inference for offline ASR as an example.

### Quick Start

首先按照https://huggingface.co/speechai/model_repo_conformer_aishell_wenet_tlg 的流程，使用官方提供的模型运行一下示例。

下面使用自己的模型进行配置和运行

```sh
# using docker image runtime/gpu/Dockerfile/Dockerfile.server
# OR you can use the following image
# docker pull soar97/triton-wenet:22.12

# pwd is runtime/gpu
# 目前只支持离线的模型
onnx_model_dir=$(pwd)/../resource/ASR/onnx_gpu_model_offline

TLG_dir=$(pwd)/../resource/ASR
# cp the TLG.fst and words.fst to $PWD/cuda_decoders/model_repo_cuda_decoder/scoring
cp $TLG_dir/TLG.fst $PWD/cuda_decoders/model_repo_cuda_decoder/scoring/1/lang/
cp $TLG_dir/words.txt $PWD/cuda_decoders/model_repo_cuda_decoder/scoring/1/lang/

# 500M的ONNX模型和520M的TLG,显存预计需要10G
docker run -it --rm --name "wenet_trt_test" \
       --gpus '"device=0"' --shm-size 1g \
       -v $PWD/cuda_decoders:/ws/model_repo \
       -v $onnx_model_dir:/ws/onnx_model    \
       --net host wenet_server:22.12  
    
# 安装riva-asrlib-decoder
cd /ws/model_repo
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 启动服务端
bash run.sh  # > run_cuda_decoder_server.log 2>&1 &

# 然后按照gpu/README.md进行客户端的操作即可
```

### TODO: Performance of Small Offline ASR Model using Different Decoders

Benchmark(offline conformer model trained on Aishell1) based on Aishell1 test set with V100, the total audio duration is 36108.919 seconds.

(Note: 80 concurrent tasks, service has been fully warm up.)

|Decoding Method | decoding time(s) | WER (%) |
|----------|--------------------|-------------|
| CTC Greedy Search             | 23s | 4.97  |
| CUDA TLG 1-best (3-gram LM)   | 31s | 4.58  |
