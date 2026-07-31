include(cc_library)
include(cc_test)

cc_library(
  NAME
    routing
  HDRS
    pd_compatibility.h
    routing_configuration.h
    routing_decision.h
  SRCS
    pd_compatibility.cpp
    routing_configuration.cpp
    routing_decision.cpp
  DEPS
    :common
    nlohmann_json::nlohmann_json
)

cc_test(
  NAME
    pd_compatibility_test
  SRCS
    pd_compatibility_test.cpp
  DEPS
    :routing
    GTest::gtest_main
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
