include(cc_library)

cc_library(
  NAME
    request_event_recorder
  HDRS
    request_event_recorder.h
  SRCS
    request_event_recorder.cpp
  DEPS
    proto_xllm
)
