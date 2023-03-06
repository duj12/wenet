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

DEFINE_string(model_dir, "", "model dir path");
DEFINE_string(wav_path, "", "single wave path");
DEFINE_bool(enable_timestamp, false, "enable timestamps");
DEFINE_int32(continuous_decoding, 1, "enable continuous_decoding");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  wenet_set_log_level(2);
  void* asr_model = wenet_init_resource(FLAGS_model_dir.c_str());
  void* decoder = wenet_init();
  wenet_set_decoder_resource(decoder, asr_model);
  wenet_set_timestamp(decoder, FLAGS_enable_timestamp == true ? 1 : 0);
  wenet_set_continuous_decoding(decoder, FLAGS_continuous_decoding);
  wenet::WavReader wav_reader(FLAGS_wav_path);
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
  return 0;
}
