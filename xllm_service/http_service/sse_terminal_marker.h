/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm-service/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cstddef>
#include <string_view>

namespace xllm_service {

// Incrementally recognizes the OpenAI SSE terminal line without retaining
// response payload. A clean HTTP EOF alone is not a semantic stream terminal.
class SseTerminalMarker final {
 public:
  void observe(const void* data, size_t length) {
    if (terminal_ || data == nullptr || length == 0) {
      return;
    }
    const auto* bytes = static_cast<const char*>(data);
    for (size_t index = 0; index < length && !terminal_; ++index) {
      observe_byte(bytes[index]);
    }
  }

  bool terminal_at_end() const { return terminal_ || awaiting_line_end_; }

 private:
  void observe_byte(char byte) {
    if (awaiting_line_end_) {
      if (byte == '\r' || byte == '\n') {
        terminal_ = true;
      } else {
        awaiting_line_end_ = false;
        at_line_start_ = byte == '\n';
      }
      return;
    }
    if (!at_line_start_) {
      at_line_start_ = byte == '\n';
      return;
    }
    if (byte == kMarker[matched_bytes_]) {
      ++matched_bytes_;
      if (matched_bytes_ == kMarker.size()) {
        matched_bytes_ = 0;
        awaiting_line_end_ = true;
        at_line_start_ = false;
      }
      return;
    }
    matched_bytes_ = 0;
    at_line_start_ = byte == '\n';
  }

  static constexpr std::string_view kMarker = "data: [DONE]";

  size_t matched_bytes_ = 0;
  bool at_line_start_ = true;
  bool awaiting_line_end_ = false;
  bool terminal_ = false;
};

}  // namespace xllm_service
