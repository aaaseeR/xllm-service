include(grpc_proto_library)
include(proto_library)

set(PROTOBUF_IMPORT_DIRS ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/proto)

set(XLLM_PROTO_SRCS
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/proto/anthropic.proto
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/proto/chat.proto
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/proto/common.proto
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/proto/completion.proto
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/proto/sample.proto
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/proto/text_generation.proto
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/proto/rec.proto
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/proto/provider.proto
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/proto/observability.proto
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/proto/disagg_pd.proto
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/proto/xllm_service.proto
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/proto/embedding.proto
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/proto/embedding_data.proto
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/proto/tensor.proto
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/proto/multimodal.proto
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/proto/image_generation.proto
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/proto/audio_generation.proto
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/proto/video_generation.proto
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/proto/rerank.proto
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/proto/models.proto)

foreach(XLLM_PROTO_SRC IN LISTS XLLM_PROTO_SRCS)
  if(NOT EXISTS ${XLLM_PROTO_SRC})
    message(
      FATAL_ERROR
        "Pinned third_party/xllm is not a buildable V2 protocol source: ${XLLM_PROTO_SRC} is missing"
    )
  endif()
endforeach()

proto_library(
  NAME
    proto_rpc_service
  SRCS
    xllm_rpc_service.proto
)

proto_library(
  NAME
    proto_http_service
  SRCS
    xllm_http_service.proto
)

proto_library(
  NAME
    proto_xllm
  SRCS
    ${XLLM_PROTO_SRCS}
)

target_link_libraries(proto_rpc_service PUBLIC proto_xllm)
