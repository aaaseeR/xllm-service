include(cc_library)
include(cc_test)

cc_library(
  NAME
    execution
  HDRS
    request_session.h
    request_session_registry.h
  SRCS
    request_session.cpp
    request_session_registry.cpp
  DEPS
    :common
    :request
    :transport
)

cc_test(
  NAME
    request_session_test
  SRCS
    request_session_test.cpp
  DEPS
    :execution
    GTest::gtest_main
)

cc_test(
  NAME
    request_session_registry_test
  SRCS
    request_session_registry_test.cpp
  DEPS
    :execution
    GTest::gtest_main
)
