include(cc_library)
include(cc_test)

cc_library(
  NAME
    runtime
  HDRS
    runtime_state.h
  SRCS
    runtime_state.cpp
)

cc_test(
  NAME
    runtime_state_test
  SRCS
    runtime_state_test.cpp
  DEPS
    :runtime
    GTest::gtest_main
)
