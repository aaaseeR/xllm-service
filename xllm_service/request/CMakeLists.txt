include(cc_library)

cc_library(
  NAME
    output_event_sequencer
  HDRS
    output_event_sequencer.h
  SRCS
    output_event_sequencer.cpp
  DEPS
    :common
)

cc_library(
  NAME
    request
  HDRS
    request.h
  DEPS
    :common
    :execution_hold
    :output_event_sequencer
    proto_xllm
)
