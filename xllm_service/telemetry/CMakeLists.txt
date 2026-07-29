include(cc_library)
include(cc_test)

cc_library(
  NAME
    telemetry
  HDRS
    prometheus_metrics.h
  SRCS
    prometheus_metrics.cpp
  DEPS
    nlohmann_json::nlohmann_json
)

cc_test(
  NAME
    prometheus_metrics_test
  SRCS
    prometheus_metrics_test.cpp
  DEPS
    :telemetry
    GTest::gtest_main
    nlohmann_json::nlohmann_json
)
