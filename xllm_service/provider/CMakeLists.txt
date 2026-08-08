include(cc_library)

cc_library(
  NAME
    provider_contract
  HDRS
    provider_contract.h
  SRCS
    provider_contract.cpp
  DEPS
    proto_xllm
)

cc_library(
  NAME
    provider_registry
  HDRS
    provider_adapter.h
    provider_registry.h
  SRCS
    provider_registry.cpp
  DEPS
    :provider_contract
    proto_xllm
)
