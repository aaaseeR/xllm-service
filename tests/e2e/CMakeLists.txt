include(cc_binary)

cc_binary(
  NAME
    xllm_service_mock_native_engine
  SRCS
    mock_native_engine.cpp
  DEPS
    proto::proto_rpc_service
    proto_xllm
    gflags::gflags
    glog::glog
)
target_link_libraries(xllm_service_mock_native_engine PRIVATE brpc-static)
