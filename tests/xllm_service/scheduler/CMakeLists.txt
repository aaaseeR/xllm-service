include(cc_test)

cc_test(
  NAME
    xllm_chat_parse_bridge_test
  SRCS
    xllm_chat_parse_bridge_test.cpp
  DEPS
    :xllm_chat_parse_runtime
    GTest::gtest_main
)

cc_test(
  NAME
    flow_control_queue_test
  SRCS
    flow_control_queue_test.cpp
  DEPS
    :flow_control_queue
    GTest::gtest_main
)
