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

#ifndef API_WENET_API_H_
#define API_WENET_API_H_

#ifdef __cplusplus
extern "C" {
#endif

// Init Model
void* wenet_init_resource(const char* model_dir, int num_thread);

// free Model
void wenet_free_resource(void* model);

/** Init decoder from the file and returns the object
 *
 * @param model_dir: the model dir
 * @returns model object or NULL if problem occured
 */
void* wenet_init();

// load model to decoder
void wenet_set_decoder_resource(void* decoder, void* model);

/** Free wenet decoder and corresponding resource
 */
void wenet_free(void* decoder);

/** Reset decoder for next decoding
 */
void wenet_reset(void* decoder);

/** init decoder
 */
void wenet_init_decoder(void* decoder);

/** Decode the input wav data
 * @param data: pcm data, encoded as int16_t(16 bits)
 * @param len: data length
 * @param last: if it is the last package
 */
void wenet_decode(void* decoder, const char* data, int len, int last);

/** Get decode result in json format
 *  It returns partial result when last is 0
 *  It returns final result when last is 1

    {
      "nbest" : [{
          "sentence" : "are you okay"
          "word_pieces" : [{
              "end" : 960,
              "start" : 0,
              "word" : "are"
            }, {
              "end" : 1200,
              "start" : 960,
              "word" : "you"
            }, {
            ...}]
        }, {
          "sentence" : "are you ok"
        }],
      "type" : "final_result"
    }

    "type": final_result/partial_result
    "nbest": nbest is enabled when n > 1 in final_result
        "sentence": the ASR result
        "word_pieces": optional, output timestamp when enabled
 */
const char* wenet_get_result(void* decoder);

/** Set n-best, range 1~10
 *  wenet_get_result will return top-n best results
 */
void wenet_set_nbest(void* decoder, int n);

/** Whether to enable word level timestamp in results
    disable it when flag = 0, otherwise enable
 */
void wenet_set_timestamp(void* decoder, int flag);

/** Add one contextual biasing
 */
void wenet_add_context(void* decoder, const char* word);

/** Set contextual biasing bonus score
 */
void wenet_set_context_score(void* decoder, float score);

/** Add one contextual biasing for specific user
 */
void wenet_add_user_context(void* decoder, const char* word);

/** Clear user's context list.
 */
void wenet_clear_user_context(void* decoder);


/** Set language, has effect on the postpocessing
 *  @param: lang, could be chs/en now
 */
void wenet_set_language(void* decoder, const char* lang);

/** Set log level
 *  We use glog in wenet, so the level is the glog level
 */
void wenet_set_log_level(int level);

/** Enable continous decoding or not
 *  flag > 0: enable, otherwise disable
 */
void wenet_set_continuous_decoding(void* decoder, int flag);

/** Give access to VAD trailing silence length.
 *  the length is in ms, default = 1000 ms.
 */
void wenet_set_vad_trailing_silence(void* decoder, int length_in_ms);

/*
* Reset user defined decoder. include hotwords and vad configure
*/
void wenet_reset_user_decoder(void* decoder);

/** Whether to enable inverse text normalization in final results
    disable it when flag = 0, otherwise enable
 */
void wenet_set_itn(void* decoder, int flag);



#ifdef __cplusplus
}
#endif

#endif  // API_WENET_API_H_
