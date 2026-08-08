include(cc_library)

cc_library(
  NAME
    request
  HDRS
    request.h
  DEPS
    :common
    :execution_hold
    proto_xllm
)
