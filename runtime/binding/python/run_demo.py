import wave
#import wenetruntime as wenet
from wenetruntime.decoder import ASRModel, Decoder


def process_one_thread(model, wav_file,
                       user_context_list,
                       vad_silence_len=1000):
    """

    :param model: 初始化之后的ASR_Model, 包含了模型等公有资源
    :param wav_file: 需要解码的音频
    :param user_context_list: 用户输入热词列表(包含了公有热词)
    :param vad_silence_len: VAD拖尾静音参数，单位ms
    :return:
    """

    '''
        Decoder对象为每个解码线程私有对象
        流式识别时Decoder类初始化的几个参数推荐设置值
            continuous_decoding: 是否连续解码，音频流场景下必须设置为True
            context: 随模型初始化的热词列表，即hot_words.txt文件对应的list
            vad_trailing_silence: VAD拖尾静音长度的参数（默认是1000ms）
            nbest: 返回解码结果的数量，最大支持10个候选结果，一般都设置为1
            enable_timestamp: 是否打开时间戳，为True时会返回每个字对应的时间戳

        关于其他decoder的可设置参数，见py/decoder.py
        '''

    decoder = Decoder(model,
                      context=user_context_list,
                      continuous_decoding=True,
                      vad_trailing_silence=vad_silence_len,
                      nbest=1,
                      enable_timestamp=False
                      )

    # In demo we read wave in non-streaming fashion.
    with wave.open(wav_file, 'rb') as fin:
        assert fin.getnchannels() == 1
        wav = fin.readframes(fin.getnframes())

    # We suppose the wav is 16k, 16bits, and decode every 0.5 seconds
    interval = int(0.5 * 16000) * 2
    for i in range(0, len(wav), interval):
        last = False if i + interval < len(wav) else True
        chunk_wav = wav[i: min(i + interval, len(wav))]
        ans = decoder.decode(chunk_wav, last)
        print(ans)


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
    model = ASRModel("../../resource/ASR")

    import threading
    t_count = 2
    wav_files = ["../../resource/WAV/test_xmov_youling/asrtest_axiong_0001.wav",
                 "../../resource/WAV/test_xmov_youling/asrtest_axiong_0002.wav"]
    user_context_lists = [["小黄车", "抓紧上车", "三二一上链接", "公募五零"], ["公墓武林"]]
    vad_silences = [500, 800]
    """
    Demo中给了两条音频进行并行解码测试，用户热词分别是4个和1个(实际打印log为加上公有热词的数量)
    VAD静音参数为500ms和800ms。可根据输出结果查看每个线程的情况。
    """

    threads = []
    for i in range(t_count):
        t = threading.Thread(target=process_one_thread,
                             args=(model, wav_files[i], user_context_lists[i]+common_context_list, vad_silences[i]))
        threads.append(t)

    for i in range(t_count):
        threads[i].start()
        """
        如果多个线程共享同一个Decoder对象(包含线程不安全的成员)，那么必须加锁，否则各个线程的解码结果可能混在一起。
        将Model和Decoder分开，Decoder为每个线程私有，那么不需要加锁就可以实现多个线程同时流式识别。
        """
        #threads[i].join()