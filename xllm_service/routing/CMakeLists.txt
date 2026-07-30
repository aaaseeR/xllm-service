include(cc_library)
include(cc_test)

cc_library(
  NAME
    routing
  HDRS
    routing_configuration.h
    routing_decision.h
  SRCS
    routing_configuration.cpp
    routing_decision.cpp
  DEPS
    nlohmann_json::nlohmann_json
)

cc_test(
  NAME
    routing_configuration_test
  SRCS
    routing_configuration_test.cpp
  DEPS
    :routing
    GTest::gtest_main
)

cc_test(
  NAME
    routing_decision_test
  SRCS
    routing_decision_test.cpp
  DEPS
    :routing
    GTest::gtest_main
)
