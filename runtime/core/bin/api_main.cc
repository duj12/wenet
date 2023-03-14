// Copyright (c) 2022  Binbin Zhang (binbzha@qq.com)
//               2023  dujing
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "api/wenet_api.h"
#include "frontend/wav.h"
#include "utils/flags.h"
#include "utils/string.h"

#include <thread>
#include "utils/thread_pool.h"

DEFINE_string(model_dir, "", "model dir path");
DEFINE_string(wav_path, "", "single wave path");
DEFINE_string(context_path, "", "hot words path");
DEFINE_string(user_context_path, "", "user hot words path");
DEFINE_bool(enable_timestamp, false, "enable timestamps");
DEFINE_int32(continuous_decoding, 1, "enable continuous_decoding");
DEFINE_int32(vad_trailing_silence, 1000, "VAD trailing silence length");
DEFINE_int32(thread_num, 1, "num of decode thread");

void decode(std::string wav_path, void* decoder){
  wenet_clear_user_context(decoder);
  if (!FLAGS_user_context_path.empty()){
      LOG(INFO) << "Reading context " << FLAGS_user_context_path;
      std::vector<std::string> contexts;
      std::ifstream infile(FLAGS_user_context_path);
      std::string context;
      while (getline(infile, context)) {
        //contexts.emplace_back(Trim(context));
        wenet_add_user_context(decoder, context.c_str());
      }
  }
    
  wenet_set_timestamp(decoder, FLAGS_enable_timestamp == true ? 1 : 0);
  wenet_set_continuous_decoding(decoder, FLAGS_continuous_decoding);
  wenet_set_vad_trailing_silence(decoder, FLAGS_vad_trailing_silence);
  //for (int i=0; i<10000; i++)   // for loop to test reset_user_decoder
  wenet_reset_user_decoder(decoder);    // to make user decoder work.
  wenet::WavReader wav_reader(wav_path);
  std::vector<int16_t> data(wav_reader.num_samples());
  for (int i = 0; i < wav_reader.num_samples(); i++) {
    data[i] = static_cast<int16_t>(*(wav_reader.data() + i));
  }

  for (int i = 0; i < 10; i++) {
    //0.4s per chunk. Return the final result when last is 1
    int interval = (0.4 * 16000) * 2;
    int last = (data.size()*2)%interval;
    int segment = (data.size()*2) / interval + int(last!=0);
    for (int j = 0; j<segment; j++){
      int start = j*interval;
      int length = j==segment-1 ? last : interval;
      wenet_decode(decoder, reinterpret_cast<const char*>(data.data())+start,
                 length, int(j==segment-1));
      const char* result = wenet_get_result(decoder);
      LOG(INFO) << i << " " << result;
    }
    wenet_reset(decoder);
  }
  wenet_free(decoder);
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  wenet_set_log_level(2);
  void* asr_model = wenet_init_resource(FLAGS_model_dir.c_str(), FLAGS_thread_num);
  std::string wav_path = FLAGS_wav_path;

  
  ThreadPool pool(FLAGS_thread_num);
  for (int i = 0; i < FLAGS_thread_num; i++) {
    void* decoder = wenet_init();
    wenet_set_decoder_resource(decoder, asr_model);
    wenet_set_vad_trailing_silence(decoder, 1000);   //set a default value
    if (!FLAGS_context_path.empty()){
      LOG(INFO) << "Reading context " << FLAGS_context_path;
      std::vector<std::string> contexts;
      std::ifstream infile(FLAGS_context_path);
      std::string context;
      while (getline(infile, context)) {
        //contexts.emplace_back(Trim(context));
        wenet_add_context(decoder, context.c_str());
      }
    }
    
    wenet_init_decoder(decoder);  // initial this decoder.
    pool.enqueue(decode, wav_path, decoder);
  }
  
  return 0;
}
