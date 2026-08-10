include(cc_test)

cc_test(
  NAME
    sse_terminal_marker_test
  SRCS
    sse_terminal_marker_test.cpp
  DEPS
    GTest::gtest_main
)

cc_test(
  NAME
    health_response_test
  SRCS
    health_response_test.cpp
  DEPS
    :health_response
    GTest::gtest_main
)

cc_test(
  NAME
    request_execution_context_test
  SRCS
    request_execution_context_test.cpp
  DEPS
    :chat_template
    :request
    GTest::gtest_main
    brpc-static
    proto_xllm
)

cc_test(
  NAME
    request_trust_policy_test
  SRCS
    request_trust_policy_test.cpp
  DEPS
    :request_trust_policy
    GTest::gtest_main
)

cc_test(
  NAME
    usage_proto_adapter_test
  SRCS
    usage_proto_adapter_test.cpp
  DEPS
    :usage_proto_adapter
    GTest::gtest_main
)

cc_test(
  NAME
    chat_json_parser_test
  SRCS
    chat_json_parser_test.cpp
  DEPS
    :chat_json_parser
    GTest::gtest_main
)

cc_test(
  NAME
    anthropic_adapter_test
  SRCS
    anthropic_adapter_test.cpp
  DEPS
    :anthropic_adapter
    GTest::gtest_main
)

cc_test(
  NAME
    anthropic_stream_encoder_test
  SRCS
    anthropic_stream_encoder_test.cpp
  DEPS
    :anthropic_stream_encoder
    GTest::gtest_main
)
