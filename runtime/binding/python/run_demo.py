import wave
from wenetruntime.decoder import ASRModel, Decoder
import json

LOG_LEVEL=2    # 设置为1，2可以显示更多log
PRINT_JSON_RES=False    #打印每次decoder.decode输出的json串
ITN = True

def process_one_thread(t_number,
                       model, wav_file,
                       common_context_list,
                       user_context_list,
                       vad_silence_len=1000):
    """

    :param model: 初始化之后的ASR_Model, 包含了模型等公有资源
    :param wav_file: 需要解码的音频
    :param common_context_list: 公共热词列表
    :param user_context_list: 用户输入热词列表
    :param vad_silence_len: VAD拖尾静音参数，单位ms
    :return:
    """

    '''
        Decoder对象为每个解码线程私有对象
        流式识别时Decoder类初始化的几个参数推荐设置值
            continuous_decoding: 如果设为False, 则使用额外的VAD模型，当前版本推荐设置为False
            context: 随模型初始化的热词列表，即hot_words.txt文件对应的list
            vad_trailing_silence: VAD拖尾静音长度的参数（默认是1000ms）
            nbest: 返回解码结果的数量，最大支持10个候选结果，一般都设置为1
            enable_timestamp: 是否打开时间戳，为True时会返回每个字对应的时间戳
            enable_itn: 是否打开文本反正则，为True时会将汉字转成对应的阿拉伯数字
            fbank_frame_shift: 提取特征的帧率，为采样点数，取值为100表示特征提取帧率为160fps(16000/100)
                实际在解码时做了4倍下采样，所以100对应的解码帧率为40fps(16000/100/4)，160对应25fps(16000/160/4)

        关于其他decoder的可设置参数，见py/decoder.py
        '''

    # 当前也支持在线程内部创建ASRModel,但是没必要这么做
    # model = ASRModel("../../resource/ASR")
    print(f"Thread {t_number}： 创建Decoder，加载公有热词")
    decoder = Decoder(model,
                      context=common_context_list,
                      continuous_decoding=False,
                      vad_trailing_silence=1000,
                      nbest=1,
                      enable_timestamp=False,
                      enable_itn = False,
                      fbank_frame_shift = 160,
                      )

    ###################################################################
    ### Do warmup. 预热Decoder
    ###################################################################
    with wave.open("../../resource/WAV/warmup.wav", 'rb') as fin:
        assert fin.getnchannels() == 1
        wav = fin.readframes(fin.getnframes())

    print(f"Thread {t_number}： 开始预热音频解码...")
    print(f"Thread {t_number}： 预热Decoder时，在调用decode方法时如果Decoder没初始化，会在内部自动初始化" )
    # We suppose the wav is 16k, 16bits, and decode every 0.3 seconds
    interval = int(0.3 * 16000) * 2
    for _ in range(2):
        for i in range(0, len(wav), interval):
            last = False if i + interval < len(wav) else True
            chunk_wav = wav[i: min(i + interval, len(wav))]
            ans = decoder.decode(chunk_wav, last)
            result = json.loads(ans) if len(ans) > 0 else None
            if result and result["type"] == "final_result" and len(result['nbest']) > 0:
                print(f"Thread {t_number}：", result["nbest"][0]["sentence"])
        print("decoder.is_warmed: ",decoder.is_warmed())
    print(f"Thread {t_number}： 预热音频解码结束.")

    decoder.set_log_level(LOG_LEVEL)

    print(f"Thread {t_number}： 加载私有热词")
    # 此时decoder已经加载了common_context_list, 如果需要设置用户自定义list，reset一下。
    decoder.reset_user_context(user_context_list)
    decoder.set_context_score(3.0)   # 设置热词分数

    print(f"Thread {t_number}： 设定VAD静音参数")
    # Decoder已经设定好vad_silence_length, 但是如果有新用户要更改，直接再set一下。
    decoder.set_vad_trailing_silence(vad_silence_len)

    print(f"Thread {t_number}： 为单个用户重设解码器")
    #更改了热词和VAD，需要重置一下decoder
    decoder.reset_user_decoder()

    # 下面两个操作不需要重置decoder
    print(f"Thread {t_number}： 开启时间戳")
    decoder.enable_timestamp(True)
    print(f"Thread {t_number}： 开启文本反正则")
    decoder.set_itn(ITN)

    # In demo we read wave in non-streaming fashion.
    with wave.open(wav_file, 'rb') as fin:
        assert fin.getnchannels() == 1
        wav = fin.readframes(fin.getnframes())

    print(f"Thread {t_number}： 开始解码...")
    # We suppose the wav is 16k, 16bits, and decode every 0.3 seconds
    # In the asr server environment, the decode chunk is fixed to 0.3s.
    interval = int(0.3 * 16000) * 2
    for i in range(0, len(wav), interval):
        last = False if i + interval < len(wav) else True
        chunk_wav = wav[i: min(i + interval, len(wav))]
        ans = decoder.decode(chunk_wav, last)
        if PRINT_JSON_RES:
            print(ans)
        result = json.loads(ans) if len(ans) > 0 else None
        if result and result["type"] == "final_result" and len(result['nbest']) > 0:
            print(f"Thread {t_number}：", result["nbest"][0]["sentence"])
            if ITN:
                print(f"Thread {t_number}：", "itn: ", result["nbest"][0]["itn"])

    print(f"Thread {t_number}： 解码结束")


def process_wav_scp(model, wav_root, wav_scp,
                       common_context_list,
                       vad_silence_len=1000,
                       result_file=None, 
                       label_file=None,
                       wer_file=None):
    """
    用于测试多个音频文件列表的解码准确率
    """
    # 测试准确率
    decoder = Decoder(model,
                      context=common_context_list,
                      continuous_decoding=False,
                      vad_trailing_silence=vad_silence_len,
                      nbest=1,
                      enable_timestamp=False,
                      enable_itn = False,
                      fbank_frame_shift = 160,
                      )
    decoder.set_log_level(LOG_LEVEL)
    fout = None
    if result_file is not None:
        fout = open(result_file, 'w', encoding='utf-8')

    with open(wav_scp, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip().split()
            wav_name, wav_path = line[0], line[1]
            wav_file = wav_root+'/'+wav_path
            print(f"Process {wav_file}")
            # In demo we read wave in non-streaming fashion.
            with wave.open(wav_file, 'rb') as fin:
                assert fin.getnchannels() == 1
                wav = fin.readframes(fin.getnframes())
            if fout:
                fout.write(wav_name + " ")
            # WAV is 16k, 16bits, and decode every 0.3 seconds
            interval = int(0.3 * 16000) * 2
            for i in range(0, len(wav), interval):
                last = False if i + interval < len(wav) else True
                chunk_wav = wav[i: min(i + interval, len(wav))]
                ans = decoder.decode(chunk_wav, last)
                result= json.loads(ans) if len(ans)>0 else None
                if fout is not None and result and result["type"] == "final_result" and len(result['nbest']) > 0:
                    print(result["nbest"][0]["sentence"])
                    fout.write(result["nbest"][0]["sentence"])

            if fout:
                fout.write("\n")
                fout.flush()



    #测试解码准确率
    import os
    if label_file is not None and fout is not None:
        os.system(f"python3 ../../resource/tools/compute-wer.py --char=1 --v=1  "
                  f"{label_file} {result_file} > {wer_file} ; "
                  f"tail {wer_file}")



if __name__ == "__main__":

    """
    Tips: 模型资源是多个线程公用，但是每个线程的解码句柄(包含解码器的设置以及使用的热词，都是每个线程私有的)
    """

    common_context_list = []
    # 如果有公用的热词，那么先加载公用热词列表
    # 通用的热词列表文件放置在ASR资源文件夹下面
    with open("../../resource/ASR/hot_words.txt", 'r', encoding='utf-8') as f_context:
        for line in f_context:
            word = line.strip()
            common_context_list.append(word)

    #初始化模型
    model = None
    print("创建模型，为多个线程公有内存")
    model = ASRModel("../../resource/ASR", num_thread=1)
    print("创建模型，模型加载已完毕")

    ACC_test = True
    STOP_test = False
    if ACC_test:
        test_set='test_youling_farfield'
        print("下面先测试模型解码准确率...")
        wav_root = "../../resource/WAV"
        wav_scp = f"../../resource/WAV/{test_set}.scp0"
        result_file = f"../../resource/WAV/{test_set}.asr"
        label_file = f"../../resource/WAV/{test_set}.txt"
        wer_file = f"../../resource/WAV/{test_set}.wer"
        vad_silence_len=1000
        process_wav_scp(model, wav_root, wav_scp,
                           common_context_list,
                           vad_silence_len,
                           result_file,
                           label_file,
                           wer_file)
        print("测试准确率完毕")
    if STOP_test:
        print("测试输入为空，强制中断解码的情况...")
        decoder = Decoder(model,
                          context=common_context_list,
                          continuous_decoding=False,
                          vad_trailing_silence=1000,
                          nbest=1,
                          enable_timestamp=False,
                          enable_itn=False,
                          fbank_frame_shift = 160,
                          )
        ans = decoder.decode(b'', True)
        print(ans)
        print("测试中断解码完毕")

    print("下面测试多线程加载不同热词...")
    t_count = 1
    case_count = 2
    wav_files = ["../../resource/WAV/warmup.wav",
                 "../../resource/WAV/10second_sil.wav"]
    user_context_lists = [["小黄车", "抓紧上车", "三二一上链接", "魔珐", "魔块"], ["魔法", "模块"]]
    vad_silences = [1000, 2500]
    itn = [True, False]   #设定是否开启
    time_stamp = [True, False]
    """
    Demo中给了两条音频进行并行解码测试，用户热词分别是4个和1个(实际打印log为加上公有热词的数量)
    VAD静音参数为500ms和2500ms。可根据输出结果查看每个进程/线程的情况。
    """


    if t_count == 1:
        process_one_thread(0, model, wav_files[0 % case_count],
                common_context_list,
                user_context_lists[0 % case_count],
                vad_silences[0 % case_count])
    else:
        '''
        多进程示例
        '''
        # from multiprocessing import Process
        # process_list = []
        # for i in range(t_count):
        #     p = Process(target=process_one_thread, args=(i, model, wav_files[i%case_count],
        #                                common_context_list,
        #                                user_context_lists[i%case_count],
        #                                vad_silences[i%case_count])  # 实例化进程对象
        #     p.start()
        #     process_list.append(p)
        #
        # for i in process_list:
        #     p.join()

        '''
        多线程示例
        '''
        import threading
        threads = []
        for i in range(t_count):
            t = threading.Thread(target=process_one_thread,
                                 args=(i, model, wav_files[i%case_count],
                                       common_context_list,
                                       user_context_lists[i%case_count],
                                       vad_silences[i%case_count]))
            threads.append(t)

        for i in range(t_count):
            threads[i].start()
            """
            如果多个线程共享同一个Decoder对象(包含线程不安全的成员)，那么必须加锁。
            将Model和Decoder分开，Decoder为每个线程私有，那么不需要加锁就可以实现多个线程同时流式识别。
            """
            #threads[i].join()


