include(cc_library)

cc_library(
  NAME
    request
  HDRS
    request.h
  DEPS
    :backend_http
    :common
)
