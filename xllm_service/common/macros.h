/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <utility>

namespace xllm_service {
#define XLLM_SERVICE_PROPERTY(T, property)                                    \
 public:                                                                      \
  [[nodiscard]] const T& property() const& noexcept { return property##_; }   \
  [[nodiscard]] T& property() & noexcept { return property##_; }              \
  [[nodiscard]] T&& property() && noexcept { return std::move(property##_); } \
                                                                              \
  auto property(const T& value) & -> decltype(*this) {                        \
    property##_ = value;                                                      \
    return *this;                                                             \
  }                                                                           \
                                                                              \
  auto property(T&& value) & -> decltype(*this) {                             \
    property##_ = std::move(value);                                           \
    return *this;                                                             \
  }                                                                           \
                                                                              \
  void property(const T& value) && = delete;                                  \
  void property(T&& value) && = delete;                                       \
                                                                              \
  T property##_

#define XLLM_SERVICE_UNUSED_PARAMETER(x) ((void)(x))

#if __has_attribute(guarded_by)
#define XLLM_SERVICE_GUARDED_BY(x) __attribute__((guarded_by(x)))
#else
#define XLLM_SERVICE_GUARDED_BY(x)
#endif

#define XLLM_SERVICE_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;                   \
  void operator=(const TypeName&) = delete

}  // namespace xllm_service
