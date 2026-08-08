include(cc_test)

cc_test(
  NAME
    output_event_sequencer_test
  SRCS
    output_event_sequencer_test.cpp
  DEPS
    :output_event_sequencer
    GTest::gtest_main
)

cc_test(
  NAME
    request_deadline_queue_test
  SRCS
    request_deadline_queue_test.cpp
  INCLUDES
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm
  DEPS
    :chat_template
    :request
    :request_deadline_queue
    GTest::gtest_main
    brpc-static
)
