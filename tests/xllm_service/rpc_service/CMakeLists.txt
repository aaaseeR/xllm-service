include(cc_test)

cc_test(
  NAME
    xllm_rpc_service_test
  SRCS
    rpc_service_test.cpp
  DEPS
    :disagg_generation_adapter
    :xllm_rpc_service
    gflags::gflags
    glog::glog
    GTest::gtest_main
    proto_xllm
)

cc_test(
  NAME
    first_event_recovery_client_test
  SRCS
    first_event_recovery_client_test.cpp
  DEPS
    :first_event_recovery_client
    :output_event_sequencer
    GTest::gtest_main
    brpc-static
    proto_xllm
)
