// Copyright (c) 2022  Binbin Zhang(binbzha@qq.com)
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

#include <pybind11/pybind11.h>

#include "api/wenet_api.h"

namespace py = pybind11;


PYBIND11_MODULE(_wenet, m) {
  m.doc() = "wenet pybind11 plugin";  // optional module docstring
  m.def("wenet_init_resource", &wenet_init_resource, py::return_value_policy::reference,
        "wenet init model resources.");
  m.def("wenet_free_resource", &wenet_free_resource, "wenet free model resources.");
  m.def("wenet_init", &wenet_init, py::return_value_policy::reference,
        "wenet init decoder and some config");
  m.def("wenet_free", &wenet_free, "wenet free");
  m.def("wenet_set_decoder_resource", &wenet_set_decoder_resource, "wenet set decoder resource");
  m.def("wenet_reset", &wenet_reset, "wenet reset");
  m.def("wenet_init_decoder", &wenet_init_decoder, "wenet init decoder");
  m.def("wenet_decode", &wenet_decode, "wenet decode");
  m.def("wenet_get_result", &wenet_get_result, py::return_value_policy::copy,
        "wenet get result");
  m.def("wenet_set_log_level", &wenet_set_log_level, "set log level");
  m.def("wenet_set_nbest", &wenet_set_nbest, "set nbest");
  m.def("wenet_set_timestamp", &wenet_set_timestamp, "set timestamp flag");
  m.def("wenet_set_itn", &wenet_set_itn, "set inverse text normalization flag");
  m.def("wenet_add_context", &wenet_add_context, "add one context word");
  m.def("wenet_set_context_score", &wenet_set_context_score,
        "set context bonus score");
  m.def("wenet_add_user_context", &wenet_add_user_context,
        "add one context word for one user");
  m.def("wenet_clear_user_context", &wenet_clear_user_context,
        "clear user's context list");
  m.def("wenet_set_language", &wenet_set_language, "set language");
  m.def("wenet_set_continuous_decoding", &wenet_set_continuous_decoding,
        "enable continuous decoding or not");
  m.def("wenet_set_vad_trailing_silence", &wenet_set_vad_trailing_silence,
        "Set VAD trailing silence length, in ms.");

  m.def("wenet_reset_user_decoder", &wenet_reset_user_decoder,
        "reset the decoder for specific user.");
}
