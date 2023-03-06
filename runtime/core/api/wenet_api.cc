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

#include <memory>
#include <string>
#include <vector>

#include "decoder/asr_decoder.h"
#include "decoder/torch_asr_model.h"
#include "post_processor/post_processor.h"
#include "utils/file.h"
#include "utils/json.h"
#include "utils/string.h"


class ASRModel{
 public:
  bool use_lm_symbols = false;
  explicit ASRModel(const std::string& model_dir) {
    // Resource init
    resource_ = std::make_shared<wenet::DecodeResource>();
    wenet::TorchAsrModel::InitEngineThreads();
    std::string model_path = wenet::JoinPath(model_dir, "final.zip");
    CHECK(wenet::FileExists(model_path));

    auto model = std::make_shared<wenet::TorchAsrModel>();
    model->Read(model_path);
    resource_->model = model;

    // units.txt: E2E model unit
    std::string unit_path = wenet::JoinPath(model_dir, "units.txt");
    CHECK(wenet::FileExists(unit_path));
    resource_->unit_table = std::shared_ptr<fst::SymbolTable>(
      fst::SymbolTable::ReadText(unit_path));

    std::string fst_path = wenet::JoinPath(model_dir, "TLG.fst");
    if (wenet::FileExists(fst_path)) {  // With LM
      resource_->fst = std::shared_ptr<fst::Fst<fst::StdArc>>(
          fst::Fst<fst::StdArc>::Read(fst_path));

      std::string symbol_path = wenet::JoinPath(model_dir, "words.txt");
      CHECK(wenet::FileExists(symbol_path));
      resource_->symbol_table = std::shared_ptr<fst::SymbolTable>(
          fst::SymbolTable::ReadText(symbol_path));
      use_lm_symbols = true;
    } else {  // Without LM, symbol_table is the same as unit_table
      resource_->symbol_table = resource_->unit_table;
      use_lm_symbols = false;
    }
  }
  std::shared_ptr<wenet::DecodeResource> get_resource(){return resource_;}

 private:
  std::shared_ptr<wenet::DecodeResource> resource_ = nullptr;
};


class Recognizer {
 public:
  explicit Recognizer() {
    // FeaturePipeline init
    // Here we use frame_length=400 and frame_shift=100
    feature_config_ = std::make_shared<wenet::FeaturePipelineConfig>(80, 16000, 400, 100);
    feature_pipeline_ =
        std::make_shared<wenet::FeaturePipeline>(*feature_config_);
//    // Resource init
//    resource_ = std::make_shared<wenet::DecodeResource>();
//    wenet::TorchAsrModel::InitEngineThreads();
//    std::string model_path = wenet::JoinPath(model_dir, "final.zip");
//    CHECK(wenet::FileExists(model_path));
//
//    auto model = std::make_shared<wenet::TorchAsrModel>();
//    model->Read(model_path);
//    resource_->model = model;
//
//    // units.txt: E2E model unit
//    std::string unit_path = wenet::JoinPath(model_dir, "units.txt");
//    CHECK(wenet::FileExists(unit_path));
//    resource_->unit_table = std::shared_ptr<fst::SymbolTable>(
//      fst::SymbolTable::ReadText(unit_path));
//
//    std::string fst_path = wenet::JoinPath(model_dir, "TLG.fst");
//    if (wenet::FileExists(fst_path)) {  // With LM
//      resource_->fst = std::shared_ptr<fst::Fst<fst::StdArc>>(
//          fst::Fst<fst::StdArc>::Read(fst_path));
//
//      std::string symbol_path = wenet::JoinPath(model_dir, "words.txt");
//      CHECK(wenet::FileExists(symbol_path));
//      resource_->symbol_table = std::shared_ptr<fst::SymbolTable>(
//          fst::SymbolTable::ReadText(symbol_path));
//      use_lm_symbols_ = true;
//    } else {  // Without LM, symbol_table is the same as unit_table
//      resource_->symbol_table = resource_->unit_table;
//      use_lm_symbols_ = false;
//    }

    // Context config init
    context_config_ = std::make_shared<wenet::ContextConfig>();
    decode_options_ = std::make_shared<wenet::DecodeOptions>();
    decode_options_->ctc_wfst_search_opts.length_penalty = -3.0;  // -3.0 is proper
    post_process_opts_ = std::make_shared<wenet::PostProcessOptions>();
  }

  void Reset() {
    if (feature_pipeline_ != nullptr) {
      feature_pipeline_->Reset();
    }
    if (decoder_ != nullptr) {
      decoder_->Reset();
    }
    result_.clear();
  }

  void InitDecoder() {
    CHECK(decoder_ == nullptr);
    // Optional init context graph
    if (context_.size() > 0 || user_context_.size() > 0) {
      std::vector<std::string> context; 
      if (context_.size()>0){
        for (int i=0; i<context_.size(); i++){
          context.emplace_back(context_[i]);
        }
      }
      if (user_context_.size()>0){
        for (int i=0; i<user_context_.size(); i++){
          context.emplace_back(user_context_[i]);
        }
      }
      context_config_->context_score = context_score_;
      auto context_graph =
          std::make_shared<wenet::ContextGraph>(*context_config_);
      context_graph->BuildContextGraph(context, resource_->symbol_table, use_lm_symbols_);
      resource_->context_graph = context_graph;
    }
    // PostProcessor
    if (language_ == "chs") {  // TODO(Binbin Zhang): CJK(chs, jp, kr)
      post_process_opts_->language_type = wenet::kMandarinEnglish;
    } else {
      post_process_opts_->language_type = wenet::kIndoEuropean;
    }
    resource_->post_processor =
        std::make_shared<wenet::PostProcessor>(*post_process_opts_);
    // Init decoder
    decoder_ = std::make_shared<wenet::AsrDecoder>(feature_pipeline_, resource_,
                                                   *decode_options_);
  }

  void Decode(const char* data, int len, int last) {
    using wenet::DecodeState;
    // Init decoder when it is called first time
    if (decoder_ == nullptr) {
      InitDecoder();
    }
    // Convert to 16 bits PCM data to float
    CHECK_EQ(len % 2, 0);
    feature_pipeline_->AcceptWaveform(reinterpret_cast<const int16_t*>(data),
                                      len / 2);
    if (last > 0) {
      feature_pipeline_->set_input_finished();
    }

    while (true) {
      DecodeState state = decoder_->Decode(false);
      if (state == DecodeState::kWaitFeats) {
        break;
      } else if (state == DecodeState::kEndFeats) {
        decoder_->Rescoring();
        UpdateResult(true);
        break;
      } else if (state == DecodeState::kEndpoint && continuous_decoding_) {
        decoder_->Rescoring();
        UpdateResult(true);
        decoder_->ResetContinuousDecoding();
        break;  //dujing: We should break decoding if we detect VAD's EndPoint 
      } else {  // kEndBatch
        UpdateResult(false);
      }
    }
  }

  void UpdateResult(bool final_result) {
    json::JSON obj;
    obj["type"] = final_result ? "final_result" : "partial_result";
    int nbest = final_result ? nbest_ : 1;
    obj["nbest"] = json::Array();
    for (int i = 0; i < nbest && i < decoder_->result().size(); i++) {
      json::JSON one;
      one["sentence"] = decoder_->result()[i].sentence;
      //if (final_result && enable_timestamp_) {
      // Here we need timestamp in partial_result
      if (enable_timestamp_) {
        one["word_pieces"] = json::Array();
        for (const auto& word_piece : decoder_->result()[i].word_pieces) {
          json::JSON piece;
          piece["word"] = word_piece.word;
          piece["start"] = word_piece.start;
          piece["end"] = word_piece.end;
          one["word_pieces"].append(piece);
        }
      }
      one["sentence"] = decoder_->result()[i].sentence;
      obj["nbest"].append(one);
    }
    obj["vad_state"] = (int)decoder_->GetVADState();
    result_ = obj.dump();
  }

  const char* GetResult() { return result_.c_str(); }

  void set_nbest(int n) { nbest_ = n; }
  void set_enable_timestamp(bool flag) { enable_timestamp_ = flag; }
  void AddContext(const char* word) { context_.emplace_back(word); }
  void add_user_context(const char* word) {user_context_.emplace_back(word); }
  void clear_user_context() {user_context_.clear();}
  void set_context_score(float score) { context_score_ = score; }
  void set_language(const char* lang) { language_ = lang; }
  void set_continuous_decoding(bool flag) { continuous_decoding_ = flag; }
  //Give access to VAD trailing silence length
  void set_vad_trailing_silence(const int length_in_ms) {
    decode_options_->ctc_endpoint_config.rule2.min_trailing_silence = length_in_ms;
  }

  void set_resource(std::shared_ptr<wenet::DecodeResource> resource){
    resource_ = resource;
  }
  
  bool use_lm_symbols_ = false;

 private:
  // NOTE(Binbin Zhang): All use shared_ptr for clone in the future
  std::shared_ptr<wenet::FeaturePipelineConfig> feature_config_ = nullptr;
  std::shared_ptr<wenet::FeaturePipeline> feature_pipeline_ = nullptr;
  std::shared_ptr<wenet::DecodeResource> resource_ = nullptr;
  std::shared_ptr<wenet::DecodeOptions> decode_options_ = nullptr;
  std::shared_ptr<wenet::AsrDecoder> decoder_ = nullptr;
  std::shared_ptr<wenet::ContextConfig> context_config_ = nullptr;
  std::shared_ptr<wenet::PostProcessOptions> post_process_opts_ = nullptr;

  int nbest_ = 1;
  std::string result_;
  bool enable_timestamp_ = false;
  std::vector<std::string> context_;
  std::vector<std::string> user_context_;  // the context list for specific user
  float context_score_;
  std::string language_ = "chs";
  bool continuous_decoding_ = false;
};

void* wenet_init_resource(const char* model_dir){
  ASRModel* model = new ASRModel(model_dir);
  return reinterpret_cast<void*>(model);
}

void wenet_free_resource(void* model) {
  delete reinterpret_cast<ASRModel*>(model);
}

void* wenet_init() {
  Recognizer* decoder = new Recognizer();
  return reinterpret_cast<void*>(decoder);
}

void wenet_set_decoder_resource(void* decoder, void *model){
  Recognizer* recognizer = reinterpret_cast<Recognizer*>(decoder);
  ASRModel* asr_model = reinterpret_cast<ASRModel*>(model);
  std::shared_ptr<wenet::DecodeResource> resource = asr_model->get_resource();
  recognizer->set_resource(resource);
  recognizer->use_lm_symbols_ = asr_model->use_lm_symbols;
}

void wenet_free(void* decoder) {
  delete reinterpret_cast<Recognizer*>(decoder);
}

void wenet_reset(void* decoder) {
  Recognizer* recognizer = reinterpret_cast<Recognizer*>(decoder);
  recognizer->Reset();
}

void wenet_init_decoder(void* decoder) {
  Recognizer* recognizer = reinterpret_cast<Recognizer*>(decoder);
  recognizer->InitDecoder();
}

void wenet_decode(void* decoder, const char* data, int len, int last) {
  Recognizer* recognizer = reinterpret_cast<Recognizer*>(decoder);
  recognizer->Decode(data, len, last);
}

const char* wenet_get_result(void* decoder) {
  Recognizer* recognizer = reinterpret_cast<Recognizer*>(decoder);
  return recognizer->GetResult();
}

void wenet_set_log_level(int level) {
  FLAGS_logtostderr = true;
  FLAGS_v = level;
}

void wenet_set_nbest(void* decoder, int n) {
  Recognizer* recognizer = reinterpret_cast<Recognizer*>(decoder);
  recognizer->set_nbest(n);
}

void wenet_set_timestamp(void* decoder, int flag) {
  Recognizer* recognizer = reinterpret_cast<Recognizer*>(decoder);
  bool enable = flag > 0 ? true : false;
  recognizer->set_enable_timestamp(enable);
}

void wenet_add_context(void* decoder, const char* word) {
  Recognizer* recognizer = reinterpret_cast<Recognizer*>(decoder);
  recognizer->AddContext(word);
}

void wenet_set_context_score(void* decoder, float score) {
  Recognizer* recognizer = reinterpret_cast<Recognizer*>(decoder);
  recognizer->set_context_score(score);
}

void wenet_add_user_context(void* decoder, const char* word) {
  Recognizer* recognizer = reinterpret_cast<Recognizer*>(decoder);
  recognizer->add_user_context(word);
}
void wenet_clear_user_context(void* decoder) {
  Recognizer* recognizer = reinterpret_cast<Recognizer*>(decoder);
  recognizer->clear_user_context();
}

void wenet_set_language(void* decoder, const char* lang) {
  Recognizer* recognizer = reinterpret_cast<Recognizer*>(decoder);
  recognizer->set_language(lang);
}

void wenet_set_continuous_decoding(void* decoder, int flag) {
  Recognizer* recognizer = reinterpret_cast<Recognizer*>(decoder);
  recognizer->set_continuous_decoding(flag > 0);
}

void wenet_set_vad_trailing_silence(void* decoder, int length_in_ms) {
  Recognizer* recognizer = reinterpret_cast<Recognizer*>(decoder);
  recognizer->set_vad_trailing_silence(length_in_ms);
}