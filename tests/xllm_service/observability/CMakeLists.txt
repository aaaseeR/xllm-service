include(cc_test)

cc_test(
  NAME
    request_identity_test
  SRCS
    request_identity_test.cpp
  DEPS
    :request_identity
    GTest::gtest_main
    proto_xllm
)

cc_test(
  NAME
    request_event_recorder_test
  SRCS
    request_event_recorder_test.cpp
  DEPS
    :request_event_recorder
    GTest::gtest_main
    proto_xllm
)
