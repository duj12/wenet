# Xmov Streaming ASR 

  根据WeNet的训练和推理框架进行开发，可根据需要部署在不同平台。

  具体部署方法，切换到对应平台路径下，按照说明进行编译即可。

  推荐采用Libtorch的方式，编译到X86平台服务端，运行demo和一些配置参数可查看libtorch/run_asr_server.sh。其他开放的配置参数见core/decoder/params.h。
  
  具体步骤如下：
    
    cd libtorch
    mkdir build && cd build && cmake .. && cmake --build .
  
  如果需要编译GPU版本
  
    mkdir build && cd build && cmake -DGPU=ON .. && cmake --build .

  然后将模型资源从S:\users\dujing\asr-online\resource拷贝到当前项目根目录（默认为asr-online）下
  
  再在libtorch目录下运行
  
    ./run_asr_server.sh

  最后用浏览器打开libtorch/web/templates/index.html, 将ip替换为启动服务的服务器ip,即可开始流式识别。
    

  由于服务启动时对刚开始输入的音频流解码时延比较大，可以采取类似于core/bin/decoder_main.cc里面warmup的方式，对模型进行预热，提前启动服务让模型对部分静音进行解码，然后才开始接收用户语音，优化用户体验。  

  如有其他问题，联系dujing。


# Runtime on WeNet

This is the runtime of WeNet.

We are going to support the following platforms:

1. Various deep learning inference engines, such as LibTorch, ONNX, OpenVINO, TVM, and so on.
2. Various OS, such as android, iOS, Harmony, and so on.
3. Various AI chips, such as GPU, Horzion BPU, and so on.
4. Various hardware platforms, such as Raspberry Pi.
5. Various language binding, such as python and go.

Feel free to volunteer yourself if you are interested in trying out some items(they do not have to be on the list).

## Progress

For each platform, we will create a subdirectory in runtime. Currently, we have:

- [x] LibTorch: in c++, the default engine of WeNet.
- [x] OnnxRuntime: in c++, the official runtime for onnx model.
- [x] GPU: in python, powered by triton.
- [x] android: in java, it shows an APP demo.
- [ ] Language binding
  - [x] binding/python: python is the first class for binding.
  - [ ] binding/go: ongoing.


