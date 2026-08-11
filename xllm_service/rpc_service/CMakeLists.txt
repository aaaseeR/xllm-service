include(cc_binary)
include(cc_library)

cc_library(
  NAME
  disagg_generation_adapter
  HDRS
    disagg_generation_adapter.h
  SRCS
    disagg_generation_adapter.cpp
  DEPS
    :common
    proto::proto_rpc_service
)

cc_library(
  NAME
  first_event_recovery_client
  HDRS
    first_event_recovery_client.h
  SRCS
    first_event_recovery_client.cpp
  DEPS
    :common
    :disagg_generation_adapter
    proto::proto_rpc_service
)
target_link_libraries(first_event_recovery_client PRIVATE brpc-static)

cc_library(
  NAME
  state_stream_client
  HDRS
    state_stream_client.h
  SRCS
    state_stream_client.cpp
  DEPS
    :provider_contract
    proto::proto_rpc_service
)
target_link_libraries(state_stream_client PRIVATE brpc-static)

cc_library(
  NAME
  kv_state_stream_client
  HDRS
    kv_state_stream_client.h
  SRCS
    kv_state_stream_client.cpp
  DEPS
    :provider_contract
    proto::proto_rpc_service
)
target_link_libraries(kv_state_stream_client PRIVATE brpc-static)

cc_library(
  NAME
  kv_snapshot_client
  HDRS
    kv_snapshot_client.h
  SRCS
    kv_snapshot_client.cpp
  DEPS
    :identity_key
    :provider_contract
    proto_xllm
)
target_link_libraries(kv_snapshot_client PRIVATE brpc-static)

cc_library(
  NAME
  xllm_rpc_service
  HDRS
    service.h
  SRCS
    service.cpp
  DEPS
    :common
    :disagg_generation_adapter
    :kv_shadow_index
    :scheduler
    absl::random_random
    absl::strings
    glog::glog
    proto::proto_rpc_service
    proto_xllm
    tokenizer
    chat_template
)
target_link_libraries(xllm_rpc_service PRIVATE brpc-static)

cc_binary(
  NAME 
    xllm_rpc_serving
  SRCS 
    main.cpp
  DEPS
    :xllm_rpc_service
    gflags::gflags   
)

cc_library(
  NAME
  xllm_rpc_client
  HDRS
    client.h
  SRCS
    client.cpp
  DEPS
    :common
    glog::glog
    proto::proto_rpc_service
)
target_link_libraries(xllm_rpc_client PRIVATE brpc-static)
